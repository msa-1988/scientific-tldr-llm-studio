from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import ProjectConfig
from src.modeling import generate_summary, load_adapter_model, load_base_model, load_tokenizer
from src.paths import ARTIFACTS_DIR


st.set_page_config(
    page_title="Scientific TLDR LLM Studio",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_base_runtime(model_name: str):
    return load_base_model(model_name), load_tokenizer(model_name)


@st.cache_resource
def load_tuned_runtime(model_name: str, adapter_dir: str):
    return load_adapter_model(model_name, Path(adapter_dir))


def render_metric_cards(metrics: dict) -> None:
    left, middle, right = st.columns(3)
    left.metric("Base ROUGE-L", metrics["base"]["rougeL_mean"])
    middle.metric("Tuned ROUGE-L", metrics["tuned"]["rougeL_mean"])
    right.metric("Delta", metrics["delta"]["rougeL_mean"])


def load_artifact_or_notice(filename: str):
    path = ARTIFACTS_DIR / filename
    if path.exists():
        return load_json(path)
    st.info(f"Run the pipeline to generate `{filename}`.")
    return None


config = ProjectConfig()
st.title("Scientific TLDR LLM Studio")
st.caption(
    "Fine-tune a compact instruct model for scientific abstract summarization with QLoRA, "
    "then compare the base model and the domain adapter side by side."
)

metrics = load_artifact_or_notice("evaluation_metrics.json")
dataset_profile = load_artifact_or_notice("dataset_profile.json")
training_summary = load_artifact_or_notice("training_summary.json")
qualitative_examples = load_artifact_or_notice("qualitative_examples.json")

if metrics:
    render_metric_cards(metrics)

tab_compare, tab_eval, tab_training = st.tabs(["Compare Outputs", "Evaluation", "Training Setup"])

with tab_compare:
    st.subheader("Live comparison")
    abstract = st.text_area(
        "Scientific abstract",
        value=(
            "We introduce a compact graph attention architecture for scientific summarization. "
            "The model is optimized with low-rank adapters and evaluated on a public technical-summary dataset."
        ),
        height=220,
    )
    adapter_dir = ARTIFACTS_DIR / "adapter"
    live_ready = adapter_dir.exists()
    if live_ready:
        if st.button("Run base vs tuned comparison", type="primary"):
            base_model, base_tokenizer = load_base_runtime(config.base_model)
            tuned_model, tuned_tokenizer = load_tuned_runtime(config.base_model, str(adapter_dir))
            with st.spinner("Generating base summary..."):
                base_output = generate_summary(base_model, base_tokenizer, abstract)
            with st.spinner("Generating tuned summary..."):
                tuned_output = generate_summary(tuned_model, tuned_tokenizer, abstract)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Base model**")
                st.write(base_output.text)
                st.caption(f"{base_output.latency_seconds:.2f}s, {base_output.generated_tokens} new tokens")
            with col2:
                st.markdown("**Fine-tuned adapter**")
                st.write(tuned_output.text)
                st.caption(f"{tuned_output.latency_seconds:.2f}s, {tuned_output.generated_tokens} new tokens")
    else:
        st.warning("No local adapter found yet. Run the training and evaluation steps first.")

    if qualitative_examples:
        st.subheader("Saved evaluation examples")
        df = pd.DataFrame(qualitative_examples)
        st.dataframe(df[["paper_id", "reference", "base_prediction", "tuned_prediction"]], width="stretch")

with tab_eval:
    st.subheader("Held-out evaluation")
    if metrics:
        metrics_df = pd.DataFrame(
            [
                {"Model": "Base", **metrics["base"]},
                {"Model": "Tuned", **metrics["tuned"]},
            ]
        )
        st.dataframe(metrics_df, width="stretch")
    if qualitative_examples:
        st.markdown("**Representative comparisons**")
        for row in qualitative_examples[:4]:
            st.markdown(f"**Paper ID:** `{row['paper_id']}`")
            st.write(f"Reference: {row['reference']}")
            st.write(f"Base: {row['base_prediction']}")
            st.write(f"Tuned: {row['tuned_prediction']}")
            st.divider()

with tab_training:
    st.subheader("Dataset and optimization choices")
    left, right = st.columns(2)
    with left:
        if dataset_profile:
            st.json(dataset_profile)
    with right:
        if training_summary:
            st.json(training_summary)

