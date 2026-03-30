"""AIME 2024 loaders (HuggingFace + optional normalized JSONL cache).

Primary HF source: ``HuggingFaceH4/aime_2024``.  Routing scripts on main use
``load_aime2024_hf``; the strong-baselines runner uses ``load_aime2024`` (JSONL
export + optional local files).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from src.datasets.gsm8k import Query
from src.utils.answer_extraction import normalize_math_answer

DEFAULT_HF_SOURCE = "HuggingFaceH4/aime_2024"
DEFAULT_JSONL_REL = Path("data/aime_2024_normalized.jsonl")


def _row_to_record(ex: dict) -> dict[str, str]:
    """Map HF row to normalized {question, answer} (numeric / math string)."""
    q = str(ex.get("problem") or ex.get("Problem") or "").strip()
    ans = ex.get("answer")
    if ans is None:
        ans = ex.get("Answer")
    a = str(ans).strip() if ans is not None else ""
    return {"question": q, "answer": a}


def write_aime2024_normalized_jsonl(
    records: list[dict[str, str]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_aime2024(
    max_samples: int | None = None,
    cache_dir: str = "data",
    hf_source: str = DEFAULT_HF_SOURCE,
    split: str = "train",
    jsonl_path: str | Path | None = DEFAULT_JSONL_REL,
    data_file: str | Path | None = None,
) -> list[Query]:
    """Load AIME 2024 as ``Query`` objects; gold ``answer`` is math-normalized.

    If ``data_file`` is set, load JSON array or JSONL with ``question``/``answer``.
    Otherwise load from HuggingFace and optionally write ``jsonl_path`` (pass
    ``jsonl_path=None`` to skip writing).
    """
    if data_file is not None:
        p = Path(data_file)
        records: list[dict[str, str]] = []
        if str(data_file).endswith(".jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            payload = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"{data_file} must be a JSON array or .jsonl")
            records = payload
        out: list[Query] = []
        for idx, rec in enumerate(records):
            if max_samples is not None and idx >= max_samples:
                break
            qid = str(rec.get("id") or rec.get("question_id") or f"aime2024_{idx}")
            qtext = str(rec["question"]).strip()
            raw_ans = str(rec["answer"]).strip()
            if not qtext or not raw_ans:
                continue
            out.append(
                Query(
                    id=qid,
                    question=qtext,
                    answer=normalize_math_answer(raw_ans),
                    choices=None,
                )
            )
        return out

    ds = load_dataset(hf_source, split=split, cache_dir=cache_dir)
    norm_rows: list[dict[str, str]] = []
    queries: list[Query] = []
    for idx, ex in enumerate(ds):
        rec = _row_to_record(dict(ex))
        norm_rows.append(rec)
        if max_samples is not None and idx >= max_samples:
            continue
        if not rec["question"] or not rec["answer"]:
            continue
        qid = str(dict(ex).get("id") or dict(ex).get("ID") or f"aime2024_{split}_{idx}")
        queries.append(
            Query(
                id=qid,
                question=rec["question"],
                answer=normalize_math_answer(rec["answer"]),
                choices=None,
            )
        )

    if jsonl_path is not None:
        write_aime2024_normalized_jsonl(norm_rows, jsonl_path)

    return queries


def load_aime2024_hf(
    max_samples: Optional[int] = None,
    cache_dir: str = "data",
    dataset_id: str = "HuggingFaceH4/aime_2024",
    split: str = "train",
) -> list[Query]:
    """Load AIME rows from HF; same normalization as ``load_aime2024`` (no JSONL write)."""
    return load_aime2024(
        max_samples=max_samples,
        cache_dir=cache_dir,
        hf_source=dataset_id,
        split=split,
        jsonl_path=None,
        data_file=None,
    )
