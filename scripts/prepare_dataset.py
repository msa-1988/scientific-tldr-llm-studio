#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import ProjectConfig
from src.dataset_utils import create_dataset_splits, export_jsonl, summarize_dataset
from src.paths import ARTIFACTS_DIR, PROCESSED_DIR


def main() -> None:
    config = ProjectConfig()
    rows_by_split = create_dataset_splits(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        train_size=config.train_size,
        validation_size=config.validation_size,
        test_size=config.test_size,
        max_input_words=config.max_input_words,
    )
    for split, rows in rows_by_split.items():
        export_jsonl(rows, PROCESSED_DIR / f"{split}.jsonl")
    profile = summarize_dataset(rows_by_split)
    profile["config"] = config.to_dict()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS_DIR / "dataset_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, ensure_ascii=True)
    print(f"Prepared dataset into {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

