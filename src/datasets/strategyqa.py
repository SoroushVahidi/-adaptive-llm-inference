"""StrategyQA loader (local normalized file first, optional HF fallback)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from src.datasets.gsm8k import Query
from src.utils.answer_extraction import normalize_boolean_answer

DEFAULT_NORMALIZED_PATH = Path("data/strategyqa_normalized.jsonl")
DEFAULT_SAMPLE_PATH = Path("data/strategyqa_sample.jsonl")
HF_DATASET_ID = "ChilleD/StrategyQA"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _subset_indices(n: int, max_samples: Optional[int], seed: Optional[int]) -> list[int]:
    idxs = list(range(n))
    if max_samples is None or max_samples >= n:
        return idxs
    if seed is None:
        return idxs[:max_samples]
    rng = random.Random(seed)
    return sorted(rng.sample(idxs, k=max_samples))


def load_strategyqa_records(
    *,
    normalized_path: str | Path = DEFAULT_NORMALIZED_PATH,
    sample_path: str | Path = DEFAULT_SAMPLE_PATH,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    allow_external: bool = True,
    cache_dir: str = "data",
) -> list[dict[str, Any]]:
    p = Path(normalized_path)
    if p.is_file():
        rows = _load_jsonl(p)
    else:
        sp = Path(sample_path)
        if sp.is_file():
            rows = _load_jsonl(sp)
        elif allow_external:
            rows = []
            for split in ["train", "test"]:
                ds = load_dataset(HF_DATASET_ID, split=split, cache_dir=cache_dir)
                for i, ex in enumerate(ds):
                    ans = normalize_boolean_answer(str(ex.get("answer", "")))
                    rows.append(
                        {
                            "dataset": "strategyqa",
                            "question_id": str(ex.get("qid", f"strategyqa_{split}_{i}")),
                            "question": str(ex.get("question", "")),
                            "options": None,
                            "answer": ans,
                            "answer_format": "boolean",
                            "category": "strategyqa",
                            "source_split": split,
                            "metadata": {
                                "term": ex.get("term", ""),
                                "description": ex.get("description", ""),
                                "facts": ex.get("facts", ""),
                            },
                        }
                    )
        else:
            raise FileNotFoundError(
                f"No local StrategyQA file found at {p} or {sp}; set allow_external=True to fetch."
            )
    idxs = _subset_indices(len(rows), max_samples, seed)
    return [rows[i] for i in idxs]


def load_strategyqa(
    *,
    normalized_path: str | Path = DEFAULT_NORMALIZED_PATH,
    sample_path: str | Path = DEFAULT_SAMPLE_PATH,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    allow_external: bool = True,
    cache_dir: str = "data",
) -> list[Query]:
    rows = load_strategyqa_records(
        normalized_path=normalized_path,
        sample_path=sample_path,
        max_samples=max_samples,
        seed=seed,
        allow_external=allow_external,
        cache_dir=cache_dir,
    )
    return [
        Query(
            id=str(r["question_id"]),
            question=str(r["question"]),
            answer=normalize_boolean_answer(str(r.get("answer", ""))),
            choices=None,
        )
        for r in rows
    ]
