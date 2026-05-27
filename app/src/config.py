from dataclasses import dataclass, asdict
from typing import Any


SYSTEM_PROMPT = (
    "You are a technical research assistant. Summarize the scientific abstract "
    "into one precise TL;DR sentence. Keep the answer factual, concise, and "
    "grounded only in the abstract."
)


@dataclass
class ProjectConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "allenai/scitldr"
    dataset_config: str = "Abstract"
    train_size: int = 1992
    validation_size: int = 300
    test_size: int = 300
    max_input_words: int = 240
    max_seq_length: int = 512
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 1.5e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    eval_batch_size: int = 4
    max_steps: int = 300
    eval_sample_count: int = 120

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
