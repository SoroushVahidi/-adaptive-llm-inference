"""MMLU-Pro loader (local normalized file first, optional HF fallback)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from src.datasets.gsm8k import Query

DEFAULT_NORMALIZED_PATH = Path("data/mmlu_pro_normalized.jsonl")
DEFAULT_SAMPLE_PATH = Path("data/mmlu_pro_sample.jsonl")
HF_DATASET_ID = "TIGER-Lab/MMLU-Pro"


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
    chosen = sorted(rng.sample(idxs, k=max_samples))
    return chosen


def load_mmlu_pro_records(
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
            ds = load_dataset(HF_DATASET_ID, split="test", cache_dir=cache_dir)
            rows = []
            for i, ex in enumerate(ds):
                opts = [str(x) for x in ex["options"]]
                rows.append(
                    {
                        "dataset": "mmlu_pro",
                        "question_id": str(ex.get("question_id", f"mmlu_pro_test_{i}")),
                        "question": str(ex["question"]),
                        "options": opts,
                        "answer": str(ex.get("answer", "")).strip().upper(),
                        "answer_format": "multiple_choice",
                        "category": str(ex.get("category", "")),
                        "source_split": "test",
                        "metadata": {
                            "answer_index": int(ex.get("answer_index", -1)),
                            "src": ex.get("src", ""),
                        },
                    }
                )
        else:
            raise FileNotFoundError(
                f"No local MMLU-Pro file found at {p} or {sp}; set allow_external=True to fetch."
            )
    idxs = _subset_indices(len(rows), max_samples, seed)
    return [rows[i] for i in idxs]


def load_mmlu_pro(
    *,
    normalized_path: str | Path = DEFAULT_NORMALIZED_PATH,
    sample_path: str | Path = DEFAULT_SAMPLE_PATH,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    allow_external: bool = True,
    cache_dir: str = "data",
) -> list[Query]:
    rows = load_mmlu_pro_records(
        normalized_path=normalized_path,
        sample_path=sample_path,
        max_samples=max_samples,
        seed=seed,
        allow_external=allow_external,
        cache_dir=cache_dir,
    )
    out: list[Query] = []
    for r in rows:
        out.append(
            Query(
                id=str(r["question_id"]),
                question=str(r["question"]),
                answer=str(r["answer"]).upper(),
                choices=tuple(str(x) for x in (r.get("options") or [])),
            )
        )
    return out
