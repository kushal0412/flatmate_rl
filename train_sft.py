"""Train a local Hugging Face causal LM on Flatmate RL synthetic SFT data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def require_training_deps():
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "SFT training requires optional dependencies. Install torch, transformers, and datasets first."
        ) from exc
    return torch, load_dataset, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def render_messages(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)
    if add_generation_prompt:
        rendered += "\n\nASSISTANT:\n"
    return rendered


def build_tokenizer_mapper(tokenizer: Any, max_length: int):
    def tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        prompt_messages = messages[:-1]
        full_text = render_messages(tokenizer, messages, add_generation_prompt=False)
        prompt_text = render_messages(tokenizer, prompt_messages, add_generation_prompt=True)

        full = tokenizer(full_text, truncation=True, max_length=max_length)
        prompt = tokenizer(prompt_text, truncation=True, max_length=max_length)
        labels = list(full["input_ids"])
        prompt_len = min(len(prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        if all(label == -100 for label in labels) and labels:
            # If truncation removed the assistant span, keep the last token trainable
            # so Trainer does not receive an all-ignored sample.
            labels[-1] = full["input_ids"][-1]

        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    return tokenize_row


def build_collator(tokenizer: Any):
    pad_id = tokenizer.pad_token_id

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            batch["labels"].append(item["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    return collate


def training_arguments(TrainingArguments: Any, args: argparse.Namespace):
    kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "report_to": "none",
        "bf16": args.bf16,
        "fp16": args.fp16,
    }
    try:
        return TrainingArguments(eval_strategy="steps", eval_steps=args.eval_steps, **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="steps", eval_steps=args.eval_steps, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT on generated Flatmate RL JSONL data.")
    parser.add_argument("--model-name", required=True, help="Base Hugging Face model name or local path.")
    parser.add_argument("--train-file", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/sft/eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/flatmate-sft"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    torch, load_dataset, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = require_training_deps()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "eval": str(args.eval_file)},
    )
    tokenized = dataset.map(
        build_tokenizer_mapper(tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments(TrainingArguments, args),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=build_collator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
