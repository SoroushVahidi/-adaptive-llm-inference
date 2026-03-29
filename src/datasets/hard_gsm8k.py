"""Hard GSM8K slice: longest problems from the GSM8K test split (by character length).

This is a simple, reproducible proxy for "hard" grade-school math without an
external label file. Use ``max_samples`` to cap size for quick runs.
"""

from __future__ import annotations

import math

from src.datasets.gsm8k import Query, load_gsm8k


def load_hard_gsm8k(
    split: str = "test",
    max_samples: int | None = 500,
    cache_dir: str = "data",
    data_file: str | None = None,
    fraction: float | None = None,
) -> list[Query]:
    """Load GSM8K and return the longest questions first.

    Args:
        split: Passed through to ``load_gsm8k``.
        max_samples: Cap after sorting (default 500). None = all test examples.
        cache_dir: HuggingFace cache directory.
        data_file: Optional local JSON (same format as ``load_gsm8k``).
        fraction: If set (0, 1], take the top ``ceil(fraction * N)`` longest
            after loading full split (before ``max_samples`` cap). Ignored if None.
    """
    queries = load_gsm8k(split=split, max_samples=None, cache_dir=cache_dir, data_file=data_file)
    queries.sort(key=lambda q: len(q.question), reverse=True)
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1].")
        k = max(1, math.ceil(fraction * len(queries)))
        queries = queries[:k]
    if max_samples is not None:
        queries = queries[:max_samples]
    return queries
