#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import ProjectConfig
from src.dataset_utils import load_prepared_jsonl
from src.evaluation import compute_best_rouge_l, summarize_generation_metrics
from src.modeling import generate_summary, load_adapter_model, load_base_model, load_tokenizer
from src.paths import ARTIFACTS_DIR, PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--test-path", type=Path, default=PROCESSED_DIR / "test.jsonl")
    return parser.parse_args()


def evaluate_rows(model, tokenizer, rows: list[dict]) -> list[dict]:
    evaluated = []
    for row in tqdm(rows, desc="Evaluating"):
        generation = generate_summary(model, tokenizer, row["abstract"])
        rouge_l, best_reference = compute_best_rouge_l(row.get("references") or [row["summary"]], generation.text)
        evaluated.append(
            {
                "paper_id": row["paper_id"],
                "reference": best_reference or row["summary"],
                "references": row.get("references") or [row["summary"]],
                "prediction": generation.text,
                "rougeL": rouge_l,
                "latency_seconds": generation.latency_seconds,
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens,
                "abstract_preview": row["abstract"][:400],
            }
        )
    return evaluated


def main() -> None:
    args = parse_args()
    config = ProjectConfig(eval_sample_count=args.sample_count)
    rows = load_prepared_jsonl(args.test_path)[: args.sample_count]

    base_tokenizer = load_tokenizer(config.base_model)
    base_model = load_base_model(config.base_model)
    base_model.eval()
    base_rows = evaluate_rows(base_model, base_tokenizer, rows)

    adapter_dir = ARTIFACTS_DIR / "adapter"
    tuned_model, tuned_tokenizer = load_adapter_model(config.base_model, adapter_dir)
    tuned_rows = evaluate_rows(tuned_model, tuned_tokenizer, rows)

    base_summary = summarize_generation_metrics(base_rows)
    tuned_summary = summarize_generation_metrics(tuned_rows)
    metrics = {
        "base": base_summary,
        "tuned": tuned_summary,
        "evaluation_protocol": {
            "references": "best-of-available multi-reference ROUGE-L",
            "sample_count": len(rows),
        },
        "delta": {
            "rougeL_mean": round(tuned_summary["rougeL_mean"] - base_summary["rougeL_mean"], 4),
            "latency_mean_seconds": round(tuned_summary["latency_mean_seconds"] - base_summary["latency_mean_seconds"], 4),
            "generated_tokens_mean": round(tuned_summary["generated_tokens_mean"] - base_summary["generated_tokens_mean"], 2),
        },
    }

    qualitative = []
    paired_rows = []
    for base_row, tuned_row in zip(base_rows, tuned_rows):
        paired_rows.append((tuned_row["rougeL"] - base_row["rougeL"], base_row, tuned_row))
    paired_rows.sort(key=lambda item: item[0], reverse=True)
    for _, base_row, tuned_row in paired_rows[:6]:
        qualitative.append(
            {
                "paper_id": base_row["paper_id"],
                "reference": base_row["reference"],
                "references": base_row["references"],
                "base_prediction": base_row["prediction"],
                "tuned_prediction": tuned_row["prediction"],
                "base_rougeL": base_row["rougeL"],
                "tuned_rougeL": tuned_row["rougeL"],
            }
        )

    with (ARTIFACTS_DIR / "base_eval_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(base_rows, handle, indent=2, ensure_ascii=True)
    with (ARTIFACTS_DIR / "tuned_eval_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(tuned_rows, handle, indent=2, ensure_ascii=True)
    with (ARTIFACTS_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
    with (ARTIFACTS_DIR / "qualitative_examples.json").open("w", encoding="utf-8") as handle:
        json.dump(qualitative, handle, indent=2, ensure_ascii=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
