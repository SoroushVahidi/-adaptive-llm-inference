"""GSM8K dataset loader.

Downloads and standardizes GSM8K into a uniform format for evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset


@dataclass
class Query:
    """A single evaluation query."""

    id: str
    question: str
    answer: str  # ground-truth numeric answer


def _extract_answer(solution: str) -> str:
    """Extract the final numeric answer from a GSM8K solution string.

    GSM8K answers end with ``#### <number>``.
    """
    match = re.search(r"####\s*(.+)", solution)
    if match:
        return match.group(1).strip().replace(",", "")
    return solution.strip()


def load_gsm8k(
    split: str = "test",
    max_samples: Optional[int] = None,
    cache_dir: str = "data",
) -> list[Query]:
    """Load GSM8K and return a list of standardized Query objects.

    Args:
        split: Which split to load (``"train"`` or ``"test"``).
        max_samples: Cap the number of returned queries (useful for debugging).
        cache_dir: Local directory for the HuggingFace cache.

    Returns:
        List of ``Query`` objects.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)

    queries: list[Query] = []
    for idx, example in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        queries.append(
            Query(
                id=f"gsm8k_{split}_{idx}",
                question=example["question"],
                answer=_extract_answer(example["answer"]),
            )
        )
    return queries
