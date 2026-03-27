"""Synthetic test-time compute instances for allocation experiments.

This module provides a small generator for explicit utility tables:
``utility_table[i][k]`` is the expected utility for query ``i`` at compute level ``k``.

The synthetic mode is intentionally simple and is meant for:
1) testing allocation algorithms directly on explicit per-query utility curves, and
2) validating optimization logic before connecting to estimated curves from real LLM runs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np


def _validate_costs(costs: Sequence[int], n_levels: int) -> list[int]:
    if len(costs) != n_levels:
        raise ValueError(f"costs length {len(costs)} must equal n_levels {n_levels}")
    normalized = [int(c) for c in costs]
    if any(c < 0 for c in normalized):
        raise ValueError("costs must be non-negative integers")
    if any(normalized[i] > normalized[i + 1] for i in range(n_levels - 1)):
        raise ValueError("costs must be non-decreasing across levels")
    return normalized


def _generate_monotone_curve(rng: np.random.Generator, n_levels: int) -> list[float]:
    start = float(rng.uniform(0.0, 0.2))
    increments = rng.uniform(0.03, 0.25, size=n_levels - 1)
    curve = [start]
    for inc in increments:
        curve.append(min(1.0, curve[-1] + float(inc)))
    return [float(v) for v in curve]


def _generate_concave_curve(rng: np.random.Generator, n_levels: int) -> list[float]:
    start = float(rng.uniform(0.0, 0.2))
    raw = sorted(rng.uniform(0.01, 0.25, size=n_levels - 1), reverse=True)
    curve = [start]
    for inc in raw:
        curve.append(min(1.0, curve[-1] + float(inc)))
    return [float(v) for v in curve]


def _difficulty_weights(difficulty: str, n_steps: int) -> np.ndarray:
    if difficulty == "easy":
        weights = np.exp(-np.linspace(0.0, 2.0, num=n_steps))
    elif difficulty == "medium":
        weights = np.ones(n_steps)
    elif difficulty == "hard":
        weights = np.linspace(0.4, 1.6, num=n_steps)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return weights / np.sum(weights)


def _generate_mixed_difficulty_curve(
    rng: np.random.Generator, n_levels: int
) -> tuple[list[float], str]:
    difficulty = str(rng.choice(["easy", "medium", "hard"], p=[0.35, 0.4, 0.25]))

    if difficulty == "easy":
        start = float(rng.uniform(0.45, 0.7))
        total_gain = float(rng.uniform(0.15, 0.4))
    elif difficulty == "medium":
        start = float(rng.uniform(0.2, 0.45))
        total_gain = float(rng.uniform(0.2, 0.45))
    else:  # hard
        start = float(rng.uniform(0.02, 0.22))
        total_gain = float(rng.uniform(0.35, 0.65))

    weights = _difficulty_weights(difficulty, n_levels - 1)
    noise = rng.uniform(0.9, 1.1, size=n_levels - 1)
    increments = total_gain * weights * noise
    increments = np.clip(increments, 0.0, None)

    curve = [start]
    for inc in increments:
        curve.append(min(1.0, curve[-1] + float(inc)))
    curve = np.maximum.accumulate(np.array(curve, dtype=float))
    return [float(v) for v in curve], difficulty


def generate_synthetic_ttc_instance(
    n_queries: int,
    n_levels: int,
    curve_family: str = "mixed_difficulty",
    costs: Sequence[int] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic utility-table allocation instance.

    Args:
        n_queries: Number of independent queries.
        n_levels: Number of compute levels per query.
        curve_family: One of:
            - "monotone"
            - "concave" (monotone with diminishing returns)
            - "mixed_difficulty" (easy / medium / hard query behavior)
        costs: Optional integer cost per level. Defaults to ``[0, 1, ..., n_levels-1]``.
        seed: Optional random seed for reproducibility.

    Returns:
        A dict containing:
            - query_ids
            - utility_table
            - costs
            - metadata
    """
    if n_queries <= 0:
        raise ValueError("n_queries must be > 0")
    if n_levels <= 1:
        raise ValueError("n_levels must be > 1")

    rng = np.random.default_rng(seed)
    resolved_costs = _validate_costs(
        list(range(n_levels)) if costs is None else costs,
        n_levels,
    )

    query_ids = [f"q_{idx:04d}" for idx in range(n_queries)]
    utility_table: list[list[float]] = []
    difficulties: list[str] = []

    for _ in range(n_queries):
        if curve_family == "monotone":
            curve = _generate_monotone_curve(rng, n_levels)
            difficulties.append("n/a")
        elif curve_family == "concave":
            curve = _generate_concave_curve(rng, n_levels)
            difficulties.append("n/a")
        elif curve_family == "mixed_difficulty":
            curve, difficulty = _generate_mixed_difficulty_curve(rng, n_levels)
            difficulties.append(difficulty)
        else:
            raise ValueError(
                "curve_family must be one of: 'monotone', 'concave', 'mixed_difficulty'"
            )
        utility_table.append(curve)

    metadata: dict[str, Any] = {
        "generator": "synthetic_ttc",
        "curve_family": curve_family,
        "n_queries": n_queries,
        "n_levels": n_levels,
        "seed": seed,
        "difficulty_distribution": dict(Counter(difficulties)),
    }

    return {
        "query_ids": query_ids,
        "utility_table": utility_table,
        "costs": resolved_costs,
        "metadata": metadata,
    }

