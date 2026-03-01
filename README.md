# HuggingFace SFT & Fine-Tuning

Scripts and datasets for supervised fine-tuning (SFT) of large language models using the HuggingFace ecosystem, including support for text and vision-language models, LoRA, and quantization.

## Project Structure

```
├── SFTexample.py          # Production-ready SFT training pipeline
├── hf_ft_odel.py          # Vision-language model fine-tuning
├── hf_sft_example.py      # Quick-start inference example
├── data_preprocess.py     # UltraFeedback dataset preprocessing
├── prompt_data.csv        # UltraFeedback dataset (158K rows)
└── prompt_style.csv       # Preference dataset (16K rows)
```

## Requirements

- Python 3.10+
- CUDA-capable GPU

Key dependencies:

```
torch
transformers>=5.0.0
datasets>=4.0.0
accelerate>=1.10.1
trl
peft
bitsandbytes>=0.47.0
pillow
```

## Usage

### SFT Training (Full)

```bash
python SFTexample.py \
  --model_name_or_path Qwen/Qwen2-0.5B \
  --dataset_name trl-lib/Capybara \
  --learning_rate 2.0e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --output_dir Qwen2-0.5B-SFT \
  --push_to_hub
```

### SFT Training (LoRA)

```bash
python SFTexample.py \
  --model_name_or_path Qwen/Qwen2-0.5B \
  --dataset_name trl-lib/Capybara \
  --learning_rate 2.0e-4 \
  --num_train_epochs 1 \
  --use_peft \
  --lora_r 32 \
  --lora_alpha 16 \
  --output_dir Qwen2-0.5B-SFT
```

### Vision-Language Fine-Tuning

```bash
python hf_ft_odel.py \
  --model_name_or_path <vision-language-model> \
  --dataset_name FanqingM/MMIU-Benchmark
```

### Dataset Preprocessing

Process the UltraFeedback dataset into unpaired preference format:

```bash
python data_preprocess.py \
  --model_name "gpt-3.5-turbo" \
  --aspect "helpfulness" \
  --push_to_hub False
```

Supported aspects: `helpfulness`, `honesty`, `instruction-following`, `truthfulness`.

### Quick Inference Test

```bash
python hf_sft_example.py
```

Runs a basic chat completion with Qwen2.5-7B-Instruct.

## Scripts

### SFTexample.py

Full SFT training pipeline with:
- Auto-detection of vision vs. text models
- 4-bit / 8-bit quantization via BitsAndBytes
- LoRA (PEFT) for parameter-efficient fine-tuning
- Gradient checkpointing
- Dataset mixture support
- Push to HuggingFace Hub

### hf_ft_odel.py

Vision-language model fine-tuning using `AutoModelForImageTextToText`. Handles multi-image inputs, ZIP extraction, and batch processing with multiprocessing.

### data_preprocess.py

Converts the [OpenBMB UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset into an unpaired preference format with binary quality labels (rating >= 5).

## Datasets

| File | Rows | Description |
|------|------|-------------|
| `prompt_data.csv` | 158,690 | UltraFeedback multi-model responses with ratings across four quality dimensions |
| `prompt_style.csv` | 16,562 | Pre-processed prompt/completion pairs with binary preference labels |

## Models Tested

- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2-0.5B
- teknium/OpenHermes-2.5-Mistral-7B

## Environment

```bash
export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="~/.cache/huggingface"
```
