#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.paths import ARTIFACTS_DIR, SCREENSHOTS_DIR


SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def draw_box(ax, xy, width, height, title, body, facecolor):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#0F766E",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.72, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(xy[0] + width / 2, xy[1] + height * 0.38, body, ha="center", va="center", fontsize=10, wrap=True)


def render_pipeline_overview():
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, "Public SciTLDR", "Download and normalize\nscientific abstracts and TL;DR targets", "#EFFCF6"),
        (0.27, "Instruction Dataset", "Prompt + abstract ->\none-sentence technical summary", "#F0F9FF"),
        (0.51, "QLoRA Fine-Tuning", "4-bit quantized base model\n+ trainable LoRA adapters", "#FFFBEB"),
        (0.75, "Eval + Demo", "ROUGE-L, latency, saved comparisons,\ninteractive Streamlit app", "#FDF2F8"),
    ]
    for x, title, body, color in boxes:
        draw_box(ax, (x, 0.25), 0.2, 0.5, title, body, color)
    for start_x in (0.23, 0.47, 0.71):
        ax.annotate("", xy=(start_x + 0.03, 0.5), xytext=(start_x, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.03, 0.9, "Scientific TLDR LLM Studio", fontsize=20, weight="bold")
    ax.text(0.03, 0.84, "End-to-end compact domain adaptation for scientific abstract summarization", fontsize=11)
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "pipeline-overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_dataset_and_config():
    profile = read_json(ARTIFACTS_DIR / "dataset_profile.json")
    split_sizes = profile["split_sizes"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].bar(split_sizes.keys(), split_sizes.values(), color=["#0F766E", "#14B8A6", "#67E8F9"])
    axes[0].set_title("Prepared dataset split sizes")
    axes[0].set_ylabel("Rows")
    config = profile["config"]
    summary_lines = [
        f"Base model: {config['base_model']}",
        f"Dataset: {config['dataset_name']} / {config['dataset_config']}",
        f"Max seq length: {config['max_seq_length']}",
        f"LoRA rank / alpha: {config['lora_rank']} / {config['lora_alpha']}",
        f"Gradient accumulation: {config['gradient_accumulation_steps']}",
        f"Mean abstract words: {profile['abstract_word_count']['mean']}",
        f"Mean summary words: {profile['summary_word_count']['mean']}",
    ]
    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.98,
        "Training configuration\n\n" + "\n".join(summary_lines),
        va="top",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "dataset-and-config.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_training_curve():
    summary = read_json(ARTIFACTS_DIR / "training_summary.json")
    log_history = summary["log_history"]
    loss_points = [(row["step"], row["loss"]) for row in log_history if "loss" in row]
    eval_points = [(row["step"], row["eval_loss"]) for row in log_history if "eval_loss" in row]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    if loss_points:
        ax.plot([x for x, _ in loss_points], [y for _, y in loss_points], marker="o", label="Train loss")
    if eval_points:
        ax.plot([x for x, _ in eval_points], [y for _, y in eval_points], marker="s", label="Eval loss")
    ax.set_title("QLoRA training trajectory")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "training-curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_base_vs_tuned():
    metrics = read_json(ARTIFACTS_DIR / "evaluation_metrics.json")
    qualitative = read_json(ARTIFACTS_DIR / "qualitative_examples.json")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    rouge_values = [metrics["base"]["rougeL_mean"], metrics["tuned"]["rougeL_mean"]]
    axes[0].bar(
        ["Base", "Tuned"],
        rouge_values,
        color=["#94A3B8", "#0F766E"],
    )
    axes[0].set_title("Held-out ROUGE-L")
    axes[0].set_ylim(0, max(0.4, metrics["tuned"]["rougeL_mean"] * 1.35))
    for idx, value in enumerate(rouge_values):
        axes[0].text(idx, value + 0.01, f"{value:.3f}", ha="center", fontsize=11, weight="bold")
    latency = [metrics["base"]["latency_mean_seconds"], metrics["tuned"]["latency_mean_seconds"]]
    latency_axis = axes[0].twinx()
    latency_axis.plot(["Base", "Tuned"], latency, color="#F97316", marker="o", label="Latency (s)")
    latency_axis.set_ylabel("Latency (s)")
    latency_axis.set_ylim(0, max(latency) * 1.4)
    axes[0].legend([latency_axis.lines[0]], ["Latency (s)"], loc="upper left")
    example = max(qualitative, key=lambda row: row["tuned_rougeL"] - row["base_rougeL"])
    axes[1].axis("off")
    text = (
        "Sample comparison\n\n"
        f"Reference:\n{example['reference']}\n\n"
        f"Base:\n{example['base_prediction']}\n\n"
        f"Tuned:\n{example['tuned_prediction']}\n\n"
        f"ROUGE-L delta: {round(example['tuned_rougeL'] - example['base_rougeL'], 4)}"
    )
    axes[1].text(
        0.02,
        0.98,
        text,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "base-vs-tuned.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    render_pipeline_overview()
    render_dataset_and_config()
    render_training_curve()
    render_base_vs_tuned()
    print(f"Saved visuals to {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
