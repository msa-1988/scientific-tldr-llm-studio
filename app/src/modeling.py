import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.config import ProjectConfig
from src.dataset_utils import build_chat_messages


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=get_compute_dtype(),
    )


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_quantization_config(),
        device_map="auto",
        trust_remote_code=False,
    )


def attach_lora_adapter(model, config: ProjectConfig):
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model, peft_config


def count_parameters(model) -> dict[str, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return {"trainable": trainable, "total": total}


def format_training_example(tokenizer, abstract: str, summary: str, max_length: int) -> dict[str, list[int]]:
    prompt_messages = build_chat_messages(abstract)
    full_messages = build_chat_messages(abstract, summary)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    ) + tokenizer.eos_token
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = input_ids.copy()
    prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
    labels[:prompt_length] = [-100] * prompt_length
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SupervisedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch_input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)
            batch_labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


@dataclass
class GenerationResult:
    text: str
    latency_seconds: float
    prompt_tokens: int
    generated_tokens: int


def generate_summary(model, tokenizer, abstract: str, max_new_tokens: int = 48) -> GenerationResult:
    messages = build_chat_messages(abstract)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    if hasattr(model, "generation_config"):
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        model.generation_config.temperature = None
        model.generation_config.repetition_penalty = 1.05
    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=min(max_new_tokens, 40),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.05,
    )
    latency_seconds = time.perf_counter() - start
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return GenerationResult(
        text=text,
        latency_seconds=latency_seconds,
        prompt_tokens=int(inputs["input_ids"].shape[-1]),
        generated_tokens=int(generated_tokens.shape[-1]),
    )


def load_adapter_model(base_model_name: str, adapter_dir: Path):
    tokenizer = load_tokenizer(base_model_name)
    base_model = load_base_model(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
