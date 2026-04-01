"""BIG-Bench Hard (BBH) loader (local normalized first, optional HF fallback)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from datasets import get_dataset_config_names, load_dataset
from src.datasets.gsm8k import Query
from src.utils.answer_extraction import normalize_text_answer

DEFAULT_NORMALIZED_PATH = Path("data/bbh_normalized.jsonl")
DEFAULT_SAMPLE_PATH = Path("data/bbh_sample.jsonl")
HF_DATASET_ID = "lukaemon/bbh"


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


def load_bbh_records(
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
            cfgs = get_dataset_config_names(HF_DATASET_ID)
            for task in sorted(cfgs):
                ds = load_dataset(HF_DATASET_ID, task, split="test", cache_dir=cache_dir)
                for i, ex in enumerate(ds):
                    rows.append(
                        {
                            "dataset": "bbh",
                            "question_id": f"bbh_{task}_{i}",
                            "question": str(ex.get("input", "")),
                            "options": None,
                            "answer": normalize_text_answer(str(ex.get("target", ""))),
                            "answer_format": "text",
                            "category": task,
                            "task": task,
                            "source_split": "test",
                            "metadata": {"task": task},
                        }
                    )
        else:
            raise FileNotFoundError(
                f"No local BBH file found at {p} or {sp}; set allow_external=True to fetch."
            )
    idxs = _subset_indices(len(rows), max_samples, seed)
    return [rows[i] for i in idxs]


def load_bbh(
    *,
    normalized_path: str | Path = DEFAULT_NORMALIZED_PATH,
    sample_path: str | Path = DEFAULT_SAMPLE_PATH,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    allow_external: bool = True,
    cache_dir: str = "data",
) -> list[Query]:
    rows = load_bbh_records(
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
            answer=normalize_text_answer(str(r.get("answer", ""))),
            choices=None,
        )
        for r in rows
    ]
