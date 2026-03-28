"""GSM8K dataset loader.

Downloads and standardizes GSM8K into a uniform format for evaluation.
Also supports loading from a local JSON file when HuggingFace is unavailable
(e.g. offline CI environments).  The local file format is a list of objects
with ``"question"`` and ``"answer"`` string fields, where the answer is either
a plain numeric string or a full GSM8K solution string ending with
``#### <number>``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
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


def _load_from_file(
    data_file: str | Path,
    split: str,
    max_samples: Optional[int],
) -> list[Query]:
    """Load queries from a local JSON file.

    The file must contain a JSON array.  Each element must have at least
    ``"question"`` and ``"answer"`` fields.  An optional ``"id"`` field is
    used when present; otherwise IDs are generated from the split and index.
    """
    records = json.loads(Path(data_file).read_text())
    if not isinstance(records, list):
        raise ValueError(f"Local data file {data_file} must contain a JSON array.")
    queries: list[Query] = []
    for idx, rec in enumerate(records):
        if max_samples is not None and idx >= max_samples:
            break
        qid = str(rec.get("id", f"gsm8k_{split}_{idx}"))
        queries.append(
            Query(
                id=qid,
                question=str(rec["question"]),
                answer=_extract_answer(str(rec["answer"])),
            )
        )
    return queries


def load_gsm8k(
    split: str = "test",
    max_samples: Optional[int] = None,
    cache_dir: str = "data",
    data_file: Optional[str | Path] = None,
) -> list[Query]:
    """Load GSM8K and return a list of standardized Query objects.

    Args:
        split: Which split to load (``"train"`` or ``"test"``).
        max_samples: Cap the number of returned queries (useful for debugging).
        cache_dir: Local directory for the HuggingFace cache.
        data_file: Optional path to a local JSON file to use instead of
            downloading from HuggingFace.  Useful for offline environments.

    Returns:
        List of ``Query`` objects.
    """
    if data_file is not None:
        return _load_from_file(data_file, split, max_samples)

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
