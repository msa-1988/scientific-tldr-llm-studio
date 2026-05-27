#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import ProjectConfig
from src.dataset_utils import build_chat_messages, build_user_prompt


def main() -> None:
    config = ProjectConfig()
    abstract = (
        "We propose a compact transformer adaptation strategy for summarizing scientific abstracts "
        "with low-rank adapters and 4-bit quantization on a consumer GPU."
    )
    prompt = build_user_prompt(abstract)
    messages = build_chat_messages(abstract, "A compact QLoRA pipeline adapts a small model for scientific TL;DR generation.")
    assert "Abstract:" in prompt
    assert messages[-1]["role"] == "assistant"
    assert config.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
