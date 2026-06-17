# Scientific TLDR LLM Studio

Fine-tune and optimize a small domain LLM for scientific abstract summarization using `QLoRA`, `PEFT`, `Transformers`, and `Streamlit`.

This project adapts `Qwen/Qwen2.5-0.5B-Instruct` on the public `allenai/scitldr` dataset so the model can generate short, technical `TL;DR` summaries for scientific abstracts. It includes dataset preparation, 4-bit fine-tuning, baseline-vs-adapter evaluation, multi-reference scoring, qualitative comparisons, and a local demo app.

Repository: `https://github.com/msa-1988/scientific-tldr-llm-studio`

## Interface Preview

### Pipeline overview

![Pipeline overview](screenshots/pipeline-overview.png)

### Dataset and training configuration

![Dataset and training configuration](screenshots/dataset-and-config.png)

### Training curve

![Training curve](screenshots/training-curve.png)

### Base vs fine-tuned comparison

![Base vs fine-tuned comparison](screenshots/base-vs-tuned.png)

## Pipeline Diagram

```mermaid
flowchart TD
    A[Public SciTLDR dataset] --> B[Download train / validation / test JSONL]
    B --> C[Normalize abstract + references]
    C --> D[Limit abstract length + build instruction format]
    D --> E[Export processed JSONL splits]
    E --> F[Save dataset profile]

    E --> G[Load tokenizer]
    G --> H[Load 4-bit base model<br/>Qwen2.5-0.5B-Instruct]
    H --> I[Attach LoRA adapter]
    I --> J[Format supervised chat examples]
    J --> K[Trainer fine-tunes adapter]
    K --> L[Save adapter + tokenizer + training summary]

    E --> M[Test rows for evaluation]
    H --> N[Base model generation]
    L --> O[Load tuned adapter model]
    M --> N
    M --> P[Tuned model generation]
    O --> P

    N --> Q[Compute ROUGE-L + latency + token stats]
    P --> Q
    Q --> R[Base vs tuned comparison]
    R --> S[Save metrics + predictions + qualitative examples]
    S --> T[Streamlit comparison app]
```

## What This Project Shows

- public-dataset curation for instruction tuning
- 4-bit `QLoRA` fine-tuning on a consumer GPU
- `PEFT` adapter training with low VRAM overhead
- baseline vs fine-tuned evaluation on held-out samples
- latency and quality comparison for the optimized runtime
- a local demo surface for qualitative comparison and interview walkthroughs

## Validation Snapshot

Current validated run:

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Fine-tuning method: `LoRA` on a `4-bit NF4` quantized base model
- Dataset view: `SciTLDR Abstract`
- Train / validation / test rows: `1992 / 300 / 300`
- Trainable parameter ratio: `5.2887%`
- Training run: `300` optimizer steps, about `1.2` epochs, on a consumer GPU
- Final validation loss: `1.9106`

Held-out multi-reference evaluation on `120` test examples:

| Model | ROUGE-L | Mean latency (s) | Mean generated tokens |
| --- | ---: | ---: | ---: |
| Base | `0.2963` | `0.6384` | `35.87` |
| Fine-tuned adapter | `0.3432` | `0.7543` | `25.27` |

Key deltas:

- `+0.0469` ROUGE-L over the base model
- `-10.6` generated tokens on average, producing tighter summaries
- a latency increase of `+0.1159s` per generation

## Current MVP Setup

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Task: scientific abstract to one-sentence `TL;DR`
- Dataset: `allenai/scitldr`
- Dataset view: `Abstract`
- Fine-tuning method: `LoRA` on a 4-bit quantized base model
- Demo surface: `Streamlit`

## Workflow

1. Download and normalize `SciTLDR`
2. Turn each abstract into an instruction-tuning sample
3. Fine-tune a small instruct model with `QLoRA`
4. Evaluate `base` vs `fine-tuned` generations with multi-reference ROUGE-L
5. Save metrics, training logs, and qualitative comparisons
6. Explore the results in the local comparison app

## Local Run

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the dataset

```bash
python scripts/prepare_dataset.py
```

### 3. Fine-tune the adapter

```bash
python scripts/train_model.py --max-steps 300
```

### 4. Evaluate base vs adapter

```bash
python scripts/evaluate_model.py --sample-count 120
```

### 5. Generate README visuals

```bash
python scripts/generate_readme_visuals.py
```

### 6. Launch the demo app

```bash
./scripts/run_local.sh
```

Open `http://localhost:8503`

## Validation

Fast sanity check:

```bash
python scripts/smoke_test.py
```

## Expected Artifacts

The training and evaluation pipeline writes:

- `artifacts/dataset_profile.json`
- `artifacts/training_summary.json`
- `artifacts/evaluation_metrics.json`
- `artifacts/qualitative_examples.json`
- `artifacts/base_eval_predictions.json`
- `artifacts/tuned_eval_predictions.json`
- `artifacts/adapter/`

## Project Layout

```text
.
├── app/
│   ├── streamlit_app.py
│   └── src/
├── artifacts/
├── data/
│   ├── processed/
│   └── README.md
├── screenshots/
├── scripts/
├── .streamlit/
├── README.md
└── requirements.txt
```

## Notes

- This project is tuned for a compact public demo and a realistic local GPU workflow, not for leaderboard chasing.
- The reported evaluation uses `best-of-available multi-reference ROUGE-L`, which fits `SciTLDR` better than single-reference scoring.
- The fine-tuned adapter improves domain alignment; it does not replace larger instruction-tuned models on broad open-domain tasks.
- The public repo contains only code, visuals, and generated metrics. The local learning notes stay outside Git history.
