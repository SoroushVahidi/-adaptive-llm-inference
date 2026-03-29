"""AIME 2024-style problems from HuggingFace ``HuggingFaceH4/aime_2024``."""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset
from src.datasets.gsm8k import Query
from src.utils.answer_extraction import normalize_math_answer


def load_aime2024_hf(
    max_samples: Optional[int] = None,
    cache_dir: str = "data",
    dataset_id: str = "HuggingFaceH4/aime_2024",
    split: str = "train",
) -> list[Query]:
    """Load AIME rows; answers normalized like MATH500."""
    ds = load_dataset(dataset_id, split=split, cache_dir=cache_dir)
    out: list[Query] = []
    for i, ex in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        qid = str(ex.get("id", f"aime_{i}"))
        problem = str(ex.get("problem", "")).strip()
        ans = str(ex.get("answer", "")).strip()
        if not problem or not ans:
            continue
        out.append(
            Query(
                id=qid,
                question=problem,
                answer=normalize_math_answer(ans),
            )
        )
    return out
