"""AIME-style problems (e.g. math-ai/aime24 on HuggingFace) in ``Query`` format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from src.datasets.gsm8k import Query
from src.utils.answer_extraction import extract_math_answer, normalize_math_answer

DEFAULT_SOURCE = "math-ai/aime24"
DEFAULT_SPLIT = "test"


def _answer_from_solution(solution: str) -> str:
    if not solution or not str(solution).strip():
        return ""
    return normalize_math_answer(extract_math_answer(str(solution)))


def _record_to_query(record: dict, index: int) -> Query:
    qid = str(record.get("id") or f"aime_{index}")
    problem = str(record.get("problem") or "")
    sol = record.get("solution") or ""
    gold = _answer_from_solution(str(sol))
    if not problem:
        raise ValueError(f"AIME record {qid} missing problem text")
    if not gold:
        raise ValueError(f"AIME record {qid} missing extractable gold from solution")
    return Query(id=qid, question=problem, answer=gold)


def _load_from_file(data_file: str | Path, max_samples: Optional[int]) -> list[Query]:
    path = Path(data_file)
    if str(data_file).endswith(".jsonl"):
        records = []
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    else:
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"{data_file} must be a JSON array")
    out: list[Query] = []
    for idx, rec in enumerate(records):
        if max_samples is not None and idx >= max_samples:
            break
        if not isinstance(rec, dict):
            raise ValueError(f"Record {idx} must be an object")
        out.append(_record_to_query(rec, idx))
    return out


def load_aime2024(
    split: str = DEFAULT_SPLIT,
    max_samples: Optional[int] = None,
    cache_dir: str = "data",
    data_file: Optional[str | Path] = None,
    dataset_source: str = DEFAULT_SOURCE,
) -> list[Query]:
    """Load AIME-style benchmark rows as ``Query`` objects."""
    if data_file is not None:
        return _load_from_file(data_file, max_samples)

    ds = load_dataset(dataset_source, split=split, cache_dir=cache_dir)
    queries: list[Query] = []
    for idx, row in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        queries.append(_record_to_query(dict(row), idx))
    return queries
