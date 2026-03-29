"""Learned revise-helpful classifier helpers.

This module provides lightweight, offline utilities for building a feature
matrix and evaluating binary revise-worthiness predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any


@dataclass(frozen=True)
class BinaryMetrics:
    """Standard binary classification metrics for revise_helpful predictions."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class SklearnSupport:
    """Captures sklearn availability status."""

    available: bool
    reason: str


def detect_sklearn_support() -> SklearnSupport:
    """Detect whether sklearn is available in the environment."""
    present = importlib.util.find_spec("sklearn") is not None
    if present:
        return SklearnSupport(available=True, reason="sklearn available")
    return SklearnSupport(
        available=False,
        reason=(
            "scikit-learn is not installed in this environment; "
            "tree/bagging/boosting training is blocked"
        ),
    )


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> BinaryMetrics:
    """Compute core binary metrics without external dependencies."""
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred, strict=True):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1

    total = max(1, len(y_true))
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def metrics_to_dict(metrics: BinaryMetrics) -> dict[str, Any]:
    """Serialize :class:`BinaryMetrics` to a JSON-friendly dict."""
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "false_positive_rate": metrics.false_positive_rate,
        "confusion_matrix": {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "tn": metrics.tn,
            "fn": metrics.fn,
        },
    }
