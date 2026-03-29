"""Hard GSM8K subset: deterministic top-k hardest problems by query features.

Used for cross-regime comparison against the standard test slice.  "Hard"
here means higher composite difficulty score from lightweight string features
(longer text, more numbers, multi-step cues), not model-based labeling.
"""

from __future__ import annotations

from typing import Optional

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
    k: int = 100,
    pool_max_samples: Optional[int] = None,
    cache_dir: str = "data",
    data_file: Optional[str] = None,
) -> list[Query]:
    """Load GSM8K *split*, rank by offline hardness score, return top *k* queries.

    *pool_max_samples* optionally caps how many examples are loaded before
    ranking (``None`` = full split — recommended so the hard slice is mined
    from the complete test set).
    """
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
