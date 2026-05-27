import json
import random
import statistics
from pathlib import Path

import requests

from src.config import SYSTEM_PROMPT


SCITLDR_URLS = {
    "Abstract": "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-A/",
    "AIC": "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-AIC/",
    "FullText": "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-FullText/",
}
SCITLDR_FILES = {
    "train": "train.jsonl",
    "validation": "dev.jsonl",
    "test": "test.jsonl",
}


def compact_whitespace(text: str) -> str:
    return " ".join(text.split())


def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def normalize_scitldr_row(row: dict, max_input_words: int) -> dict:
    abstract_sentences = row.get("source") or []
    targets = row.get("target") or []
    abstract = compact_whitespace(" ".join(sentence.strip() for sentence in abstract_sentences if sentence))
    summary = compact_whitespace(next((item for item in targets if item and item.strip()), ""))
    return {
        "paper_id": row.get("paper_id", "unknown"),
        "abstract": limit_words(abstract, max_input_words),
        "summary": summary,
        "abstract_word_count": len(abstract.split()),
        "summary_word_count": len(summary.split()),
    }


def build_user_prompt(abstract: str) -> str:
    return (
        "Summarize the following scientific abstract in one technical TL;DR sentence.\n\n"
        f"Abstract:\n{abstract}\n\n"
        "Return only the TL;DR sentence."
    )


def build_chat_messages(abstract: str, summary: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(abstract)},
    ]
    if summary is not None:
        messages.append({"role": "assistant", "content": summary})
    return messages


def export_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_prepared_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def create_dataset_splits(
    dataset_name: str,
    dataset_config: str,
    train_size: int,
    validation_size: int,
    test_size: int,
    max_input_words: int,
    seed: int = 7,
) -> dict[str, list[dict]]:
    if dataset_name != "allenai/scitldr":
        raise ValueError(f"Unsupported dataset for this project: {dataset_name}")
    if dataset_config not in SCITLDR_URLS:
        raise ValueError(f"Unsupported SciTLDR config: {dataset_config}")

    base_url = SCITLDR_URLS[dataset_config]
    normalized = {}
    for split, filename in SCITLDR_FILES.items():
        response = requests.get(base_url + filename, timeout=60)
        response.raise_for_status()
        rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        normalized[split] = [
            normalize_scitldr_row(row, max_input_words=max_input_words)
            for row in rows
        ]
    random.seed(seed)
    for rows in normalized.values():
        random.shuffle(rows)
    return {
        "train": normalized["train"][:train_size],
        "validation": normalized["validation"][:validation_size],
        "test": normalized["test"][:test_size],
    }


def summarize_dataset(rows_by_split: dict[str, list[dict]]) -> dict:
    abstract_lengths = [row["abstract_word_count"] for rows in rows_by_split.values() for row in rows]
    summary_lengths = [row["summary_word_count"] for rows in rows_by_split.values() for row in rows]
    return {
        "split_sizes": {split: len(rows) for split, rows in rows_by_split.items()},
        "abstract_word_count": {
            "mean": round(statistics.mean(abstract_lengths), 2),
            "median": round(statistics.median(abstract_lengths), 2),
            "max": max(abstract_lengths),
        },
        "summary_word_count": {
            "mean": round(statistics.mean(summary_lengths), 2),
            "median": round(statistics.median(summary_lengths), 2),
            "max": max(summary_lengths),
        },
        "sample_prompt": build_user_prompt(rows_by_split["train"][0]["abstract"]),
    }
