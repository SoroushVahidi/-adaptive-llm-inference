"""Hard GSM8K slice: multiple reproducible selection strategies.

1. **Length-based (default)** — sort GSM8K questions by character length and take
   the longest first. Matches the historical ``main`` behavior used by
   ``run_strong_baselines`` when called with ``max_samples`` / ``fraction``.

2. **Feature-score ranking** — when ``k`` is passed, rank by a composite offline
   difficulty score from ``extract_query_features`` and return the top ``k``
   queries (used by recent-baselines experiments).
"""

from __future__ import annotations

import math

from src.datasets.gsm8k import Query, load_gsm8k
from src.features.precompute_features import extract_query_features


def _hardness_score(feats: dict) -> float:
    return (
        0.01 * float(feats["question_length_chars"])
        + 2.0 * float(feats["num_numeric_mentions"])
        + 5.0 * float(feats["has_multi_step_cue"])
        + 3.0 * float(feats["has_equation_like_pattern"])
        + 2.0 * float(feats["numeric_range_approx"])
    )


def load_hard_gsm8k(
    split: str = "test",
    max_samples: int | None = 500,
    cache_dir: str = "data",
    data_file: str | None = None,
    fraction: float | None = None,
    k: int | None = None,
    pool_max_samples: int | None = None,
) -> list[Query]:
    """Load GSM8K and return a "hard" subset.

    If ``k`` is set, ignore ``fraction`` / default length sorting and instead
    rank *all* loaded examples (optionally capped by ``pool_max_samples``) by
    offline feature hardness, then return the top ``k``.

    Otherwise (``k`` is None): load the full split, sort by descending question
    length, optionally keep the top ``ceil(fraction * N)``, then cap with
    ``max_samples``.
    """
    if k is not None:
        base = load_gsm8k(
            split=split,
            max_samples=pool_max_samples,
            cache_dir=cache_dir,
            data_file=data_file,
        )
        scored: list[tuple[float, Query]] = []
        for q in base:
            feats = extract_query_features(q.question)
            scored.append((_hardness_score(feats), q))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [q for _, q in scored[:k]]

    queries = load_gsm8k(split=split, max_samples=None, cache_dir=cache_dir, data_file=data_file)
    queries.sort(key=lambda q: len(q.question), reverse=True)
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1].")
        k_frac = max(1, math.ceil(fraction * len(queries)))
        queries = queries[:k_frac]
    if max_samples is not None:
        queries = queries[:max_samples]
    return queries
