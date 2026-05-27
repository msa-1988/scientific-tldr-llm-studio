#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import ProjectConfig
from src.dataset_utils import load_prepared_jsonl
from src.modeling import (
    SupervisedDataCollator,
    attach_lora_adapter,
    count_parameters,
    format_training_example,
    load_base_model,
    load_tokenizer,
)
from src.paths import ARTIFACTS_DIR, PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--train-path", type=Path, default=PROCESSED_DIR / "train.jsonl")
    parser.add_argument("--eval-path", type=Path, default=PROCESSED_DIR / "validation.jsonl")
    return parser.parse_args()


def to_hf_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    features = [
        format_training_example(
            tokenizer=tokenizer,
            abstract=row["abstract"],
            summary=row["summary"],
            max_length=max_length,
        )
        for row in rows
    ]
    return Dataset.from_list(features)


def main() -> None:
    args = parse_args()
    config = ProjectConfig(max_steps=args.max_steps)
    tokenizer = load_tokenizer(config.base_model)
    model = load_base_model(config.base_model)
    model.config.use_cache = False
    model, _ = attach_lora_adapter(model, config)

    train_rows = load_prepared_jsonl(args.train_path)
    eval_rows = load_prepared_jsonl(args.eval_path)
    train_dataset = to_hf_dataset(train_rows, tokenizer, config.max_seq_length)
    eval_dataset = to_hf_dataset(eval_rows, tokenizer, config.max_seq_length)

    run_dir = ARTIFACTS_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_steps=15,
        save_steps=30,
        save_total_limit=1,
        eval_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer.train()

    adapter_dir = ARTIFACTS_DIR / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    parameter_counts = count_parameters(trainer.model)
    summary = {
        "base_model": config.base_model,
        "dataset": config.dataset_name,
        "dataset_config": config.dataset_config,
        "max_steps": config.max_steps,
        "train_rows": len(train_rows),
        "validation_rows": len(eval_rows),
        "parameter_counts": parameter_counts,
        "trainable_ratio_percent": round(parameter_counts["trainable"] * 100 / parameter_counts["total"], 4),
        "quantization": "4-bit NF4",
        "log_history": trainer.state.log_history,
    }
    with (ARTIFACTS_DIR / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
