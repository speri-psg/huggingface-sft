import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

OCR_PROMPT = "Extract and parse the document content from this image."


def format_gt_parse(gt_parse: dict) -> str:
    """Convert the gt_parse dict into a readable string representation."""
    lines = []
    for key, value in gt_parse.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    parts = [f"  {k}: {v}" for k, v in item.items() if v]
                    lines.append("\n".join(parts))
                else:
                    lines.append(f"  {item}")
        elif isinstance(value, dict):
            parts = [f"  {k}: {v}" for k, v in value.items() if v]
            if parts:
                lines.append(f"{key}:")
                lines.append("\n".join(parts))
        elif value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def format_cord_v2(samples: dict[str, any]) -> dict[str, list]:
    """Format cord-v2 samples into chat messages for SFTTrainer."""
    formatted_samples = {"messages": []}
    for i in range(len(samples["image"])):
        image = samples["image"][i]
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Parse the ground_truth JSON string
        gt = json.loads(samples["ground_truth"][i])
        gt_parse = gt.get("gt_parse", {})
        target_text = format_gt_parse(gt_parse)

        formatted_samples["messages"].append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_text}],
                },
            ]
        )
    return formatted_samples


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.max_length = None

    ################
    # Model, Tokenizer & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = dataset.map(format_cord_v2, batched=True, batch_size=4, num_proc=1)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()
