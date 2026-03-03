# Fine-Tuning PaddleOCR-VL-1.5 on cord-v2: Journey & Lessons Learned

## Goal

Fine-tune [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) (0.9B parameter vision-language model) on the [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) receipt OCR dataset using TRL's SFTTrainer with LoRA.

## Infrastructure

- **GPU**: vast.ai instance with 32GB VRAM
- **Local dev**: Windows 10, NVIDIA 940MX (2GB VRAM) — too small for this model, used only for script development
- **Framework**: TRL (SFTTrainer), PEFT (LoRA), transformers 4.55.x

## Key Decisions & Fixes

### 1. Transformers Version (4.55.x, not 5.x)

PaddleOCR-VL-1.5 was published with a custom `auto_map` for transformers 4.55. The model's `config.json` stores text model config at the top level, but transformers v5 expects it nested under `text_config`. Loading with v5 causes:

```
AttributeError: 'PaddleOCRVLConfig' object has no attribute 'text_config'
```

**Fix**: Pin `transformers>=4.55.0,<5.0.0` and use `AutoModel` (registered in the model's `auto_map`) instead of `AutoModelForImageTextToText`.

### 2. HF Cache Location on vast.ai

The vast.ai root overlay (`/`) has limited space (~4GB). The cord-v2 dataset needs ~4.3GB for caching. Default HuggingFace cache goes to `~/.cache` on root.

**Fix**: Set `HF_HOME=/dev/shm/hf_cache` (23GB shared memory mount) **before** importing any HF libraries. Order matters — if set after import, the libraries have already initialized their cache paths.

### 3. Dataset Mapping Performance

Initial `dataset.map()` with `batch_size=4, num_proc=1` took ~38 minutes.

**Fix**: Increased to `batch_size=32, num_proc=32` to leverage the 40 vCPUs on the vast.ai instance.

### 4. Output Format: Custom Text vs JSON

**First attempt**: Custom structured text format via `format_gt_parse()`:
```
menu:
  nm: -TICKET CP
  num: 901016
  cnt: 2
  price: 60.000
```

This format was completely alien to the base model, which naturally outputs raw OCR text. The structural leap was too large for LoRA fine-tuning.

**Fix**: Switched to JSON output (`json.dumps(gt_parse)`) which is a well-known format that models learn efficiently:
```json
{"menu": [{"nm": "-TICKET CP", "num": "901016", "cnt": "2", "price": "60.000"}], ...}
```

### 5. Prompt Format: Match the Base Model

**First attempt**: `"Extract and parse the document content from this image."` — a free-form prompt unfamiliar to the model.

The base model was trained with specific task keywords (`OCR:`, `table:`, `chart:`, etc.).

**Fix**: Changed to `"OCR: Parse this receipt into JSON"` — starts with the `OCR:` keyword the model recognizes.

### 6. Passing the Processor to SFTTrainer

Without explicitly passing the processor, SFTTrainer attempts to auto-load it. For `trust_remote_code` models, this auto-loading doesn't pass `trust_remote_code=True`, so it fails silently and falls back to a text-only tokenizer — losing all image processing.

**Fix**: Explicitly pass `processing_class=processor` to SFTTrainer.

### 7. Custom Data Collator (Critical)

The built-in SFTTrainer data collator had multiple issues with this custom model:

- **Image handling**: Embedding PIL images inside message content blocks caused them to be lost during tokenization. Images need to be in a separate column and processed through the model's image pipeline.
- **Chat template**: The generic collator doesn't handle PaddleOCR-VL's custom chat template properly.

**Fix**:
- Restructured dataset to have separate `images` and `messages` columns
- Wrote a custom `collate_fn` that manually calls `processor.apply_chat_template()` and `processor()` for proper image + text processing
- Added `dataset_kwargs={"skip_prepare_dataset": True}` and `remove_unused_columns=False` to SFTConfig

### 8. Label Masking — Only Train on Assistant Response (Critical)

This was the breakthrough fix. Without it, the loss was stuck at ~13-14 (near random for vocab size 103K) across all attempts.

**The problem**: Labels included ALL tokens — system tokens, user prompt, image placeholders, and the assistant response. The assistant's JSON response is a small fraction of total tokens. The loss was dominated by the model trying to predict unpredictable prompt/image tokens, drowning out any learning signal from the actual target.

**Fix**: In the collator, tokenize the prompt separately to find its length, then mask everything before the assistant response:

```python
# Tokenize prompt individually to get its length
prompt_tok = processor(text=[prompt_only], images=[[image]], return_tensors="pt")
prompt_lengths.append(prompt_tok["input_ids"].shape[1])

# Mask everything before assistant response
labels[i, :prompt_lengths[i]] = -100

# Also mask padding
labels[labels == processor.tokenizer.pad_token_id] = -100
```

**Result**: Loss dropped from stuck at ~13-14 to starting at ~9 and reaching **~2.0** within 25 steps.

## Training Configuration

```python
# Model
MODEL_NAME = "PaddlePaddle/PaddleOCR-VL-1.5"  # 0.9B params
DATASET_NAME = "naver-clova-ix/cord-v2"         # ~800 train samples

# Hyperparameters
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2   # effective batch = 4

# LoRA
LORA_R = 64
LORA_ALPHA = 32
LORA_TARGET_MODULES = "all-linear"
```

## Loss Progression Across Attempts

| Attempt | Changes | Loss After 2 Epochs |
|---------|---------|-------------------|
| 1 | Custom text format, embedded images, no processor | Stuck at ~13-14 |
| 2 | JSON format, `OCR:` prompt | Stuck at ~13-14 |
| 3 | Added `processing_class=processor` | Stuck at ~13-14 |
| 4 | Custom collator, separate images column | Stuck at ~13-14 |
| 5 | **Prompt masking in labels** | **Dropping to ~2.0** |

## Training Results

Final training: **2000 steps, average loss 5.86** over 10 epochs (~83 minutes on vast.ai with 32GB VRAM).

## Inference Results

Tested on 5 samples from the held-out test set:

### Sample 10 — Simple receipt (100% match)

**Ground Truth:**
```json
{
  "menu": {"nm": "CINNAMON SUGAR", "unitprice": "17,000", "cnt": "1 x", "price": "17,000"},
  "sub_total": {"subtotal_price": "17,000"},
  "total": {"total_price": "17,000", "cashprice": "20,000", "changeprice": "3,000"}
}
```

**Model Prediction:** Identical — perfect match.

### Sample 20 — Complex receipt with 5 items (100% match)

**Ground Truth:**
```json
{
  "menu": [
    {"nm": "cashew nuts chkn", "cnt": "1", "price": "64,500"},
    {"nm": "garlic pepper beef", "cnt": "1", "price": "79,500"},
    {"nm": "red curry beef", "cnt": "1", "price": "69,500"},
    {"nm": "phad thai", "cnt": "1", "price": "64,500"},
    {"nm": "steamed rice", "cnt": "4", "price": "47,600"}
  ],
  "sub_total": {"subtotal_price": "325,600", "service_price": "17,908", "tax_price": "34,351"},
  "total": {"total_price": "377,859"}
}
```

**Model Prediction:** Identical — all 5 menu items, subtotals, service charge, tax, and total perfectly extracted.

### Sample 15 — Multi-item receipt (~99% match)

**Ground Truth:**
```json
{
  "menu": [
    {"nm": "Lemon Tea (L)", "cnt": "1", "price": "25.000"},
    {"nm": "Caramel Small", "cnt": "1", "price": "38.000"}
  ],
  "total": {"total_price": "63.000", "cashprice": "70.000", "changeprice": "7.000"}
}
```

**Model Prediction:** Near-perfect — only difference was `7,000` vs `7.000` (comma vs dot in change price).

### Sample 5 — Simple receipt (~95% match)

Prediction matched all values. Only missing field: `menuqty_cnt` in total.

### Sample 0 — Receipt with discounts (~90% match)

Minor structural difference in menu (split into 2 items instead of 1) and `creditcardprice` predicted as `emoneyprice`. All numerical values correct.

### Summary

| Sample | Complexity | Accuracy |
|--------|-----------|----------|
| #0 | Moderate (discount) | ~90% |
| #5 | Simple | ~95% |
| #10 | Simple | 100% |
| #15 | Multi-item | ~99% |
| #20 | Complex (5 items + service) | 100% |

The model learned structured JSON receipt parsing from just 800 training samples in 10 epochs. Complex receipts with multiple items, taxes, and service charges are handled correctly.

## Lessons Learned

1. **Label masking is essential for VLM fine-tuning** — without it, loss is dominated by unpredictable prompt/image tokens and the model appears not to learn.
2. **Custom `trust_remote_code` models need manual integration** — SFTTrainer's auto-detection and built-in collators don't handle them well.
3. **Always test what the base model outputs first** — this reveals the expected input/output format and helps choose compatible training targets.
4. **Set environment variables before imports** — HF libraries cache their config paths at import time.
5. **Transformers version matters** — model configs evolve between major versions; match what the model was published with.
