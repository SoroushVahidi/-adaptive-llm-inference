"""Evaluation metrics."""

from __future__ import annotations

from src.baselines.base import BaselineResult


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Case-insensitive exact-match comparison after stripping whitespace."""
    return predicted.strip().lower() == ground_truth.strip().lower()


def compute_accuracy(results: list[BaselineResult]) -> dict[str, float]:
    """Compute aggregate accuracy and compute statistics.

    Returns:
        Dictionary with ``accuracy``, ``total_samples``, ``total_queries``,
        and ``avg_samples_per_query``.
    """
    if not results:
        return {
            "accuracy": 0.0,
            "total_samples": 0,
            "total_queries": 0,
            "avg_samples_per_query": 0.0,
        }

    correct = sum(1 for r in results if r.correct)
    total_samples = sum(r.samples_used for r in results)
    n = len(results)
    return {
        "accuracy": correct / n,
        "total_samples": total_samples,
        "total_queries": n,
        "avg_samples_per_query": total_samples / n,
    }
