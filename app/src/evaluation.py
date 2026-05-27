from statistics import mean

from rouge_score import rouge_scorer


def compute_rouge_l(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def summarize_generation_metrics(rows: list[dict]) -> dict:
    return {
        "rougeL_mean": round(mean(row["rougeL"] for row in rows), 4),
        "latency_mean_seconds": round(mean(row["latency_seconds"] for row in rows), 4),
        "generated_tokens_mean": round(mean(row["generated_tokens"] for row in rows), 2),
        "prompt_tokens_mean": round(mean(row["prompt_tokens"] for row in rows), 2),
    }

