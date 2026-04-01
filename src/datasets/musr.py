"""MuSR loader (local normalized file first, optional HF fallback)."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from src.datasets.gsm8k import Query

DEFAULT_NORMALIZED_PATH = Path("data/musr_normalized.jsonl")
DEFAULT_SAMPLE_PATH = Path("data/musr_sample.jsonl")
HF_DATASET_ID = "TAUR-Lab/MuSR"


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


def _letter(i: int) -> str:
    return chr(ord("A") + i)


def _parse_choices(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    txt = str(raw).strip()
    try:
        parsed = ast.literal_eval(txt)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []


def load_musr_records(
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
            dsd = load_dataset(HF_DATASET_ID, cache_dir=cache_dir)
            rows = []
            for split_name in sorted(dsd.keys()):
                split_ds = dsd[split_name]
                for i, ex in enumerate(split_ds):
                    opts = _parse_choices(ex.get("choices"))
                    ans_idx = int(ex.get("answer_index", -1))
                    q = f"{str(ex.get('narrative', '')).strip()}\n\nQuestion: {str(ex.get('question', '')).strip()}"
                    rows.append(
                        {
                            "dataset": "musr",
                            "question_id": f"musr_{split_name}_{i}",
                            "question": q,
                            "options": opts,
                            "answer": _letter(ans_idx) if 0 <= ans_idx < len(opts) else "",
                            "answer_format": "multiple_choice",
                            "category": split_name,
                            "source_split": split_name,
                            "metadata": {
                                "subtask": split_name,
                                "answer_choice": ex.get("answer_choice", ""),
                            },
                        }
                    )
        else:
            raise FileNotFoundError(
                f"No local MuSR file found at {p} or {sp}; set allow_external=True to fetch."
            )
    idxs = _subset_indices(len(rows), max_samples, seed)
    return [rows[i] for i in idxs]


def load_musr(
    *,
    normalized_path: str | Path = DEFAULT_NORMALIZED_PATH,
    sample_path: str | Path = DEFAULT_SAMPLE_PATH,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    allow_external: bool = True,
    cache_dir: str = "data",
) -> list[Query]:
    rows = load_musr_records(
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
            answer=str(r["answer"]).upper(),
            choices=tuple(str(x) for x in (r.get("options") or [])),
        )
        for r in rows
    ]
