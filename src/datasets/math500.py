"""MATH500 dataset loader.

Loads the public ``HuggingFaceH4/MATH-500`` split into the same lightweight
``Query`` format used elsewhere in the repository.  Gold answers are normalized
with a minimal LaTeX-aware exact-match routine so the downstream diagnostic can
compare symbolic answers such as fractions, tuples, and boxed expressions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from src.datasets.gsm8k import Query
from src.utils.answer_extraction import normalize_math_answer

DEFAULT_DATASET_SOURCE = "HuggingFaceH4/MATH-500"


def _record_to_query(record: dict, split: str, index: int) -> Query:
    question_id = str(
        record.get("question_id")
        or record.get("unique_id")
        or f"math500_{split}_{index}"
    )
    question = str(record.get("question") or record.get("problem") or "")
    answer = str(record.get("gold_answer") or record.get("answer") or "")
    if not question:
        raise ValueError(f"MATH500 record {question_id} is missing question/problem text")
    if not answer:
        raise ValueError(f"MATH500 record {question_id} is missing answer/gold_answer")
    return Query(
        id=question_id,
        question=question,
        answer=normalize_math_answer(answer),
    )


def _load_from_file(
    data_file: str | Path,
    split: str,
    max_samples: Optional[int],
) -> list[Query]:
    if str(data_file).endswith(".jsonl"):
        records = []
        with Path(data_file).open() as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    else:
        payload = Path(data_file).read_text()
        records = json.loads(payload)
        if not isinstance(records, list):
            raise ValueError(f"Local data file {data_file} must contain a JSON array.")

    queries: list[Query] = []
    for idx, record in enumerate(records):
        if max_samples is not None and idx >= max_samples:
            break
        if not isinstance(record, dict):
            raise ValueError(f"Local MATH500 record {idx} must be an object.")
        queries.append(_record_to_query(record, split=split, index=idx))
    return queries


def load_math500(
    split: str = "test",
    max_samples: Optional[int] = None,
    cache_dir: str = "data",
    data_file: Optional[str | Path] = None,
    dataset_source: str = DEFAULT_DATASET_SOURCE,
) -> list[Query]:
    """Load MATH500 and return standardized ``Query`` objects.

    Args:
        split: Dataset split to load.  The public benchmark currently exposes
            ``"test"``.
        max_samples: Optional cap on the number of returned queries.
        cache_dir: HuggingFace cache directory.
        data_file: Optional local JSON file for manual/offline loading.
        dataset_source: HuggingFace dataset identifier.
    """
    if data_file is not None:
        return _load_from_file(data_file=data_file, split=split, max_samples=max_samples)

    dataset = load_dataset(dataset_source, split=split, cache_dir=cache_dir)
    queries: list[Query] = []
    for idx, example in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        queries.append(_record_to_query(dict(example), split=split, index=idx))
    return queries
