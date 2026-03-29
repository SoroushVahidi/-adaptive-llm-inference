"""AIME 2024 loader (HuggingFace canonical + optional normalized JSONL cache)."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from src.datasets.gsm8k import Query

# Primary public source (30 problems, 2024)
DEFAULT_HF_SOURCE = "HuggingFaceH4/aime_2024"
DEFAULT_JSONL_REL = Path("data/aime_2024_normalized.jsonl")


def _row_to_record(ex: dict) -> dict[str, str]:
    """Map HF row to normalized {question, answer} (numeric string)."""
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
    """Load AIME 2024 as ``Query`` objects (numeric gold in ``answer``).

    If ``data_file`` is set, load JSON array or JSONL with ``question``/``answer``.
    Otherwise load from HuggingFace and optionally write ``jsonl_path``.
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
            out.append(
                Query(
                    id=qid,
                    question=str(rec["question"]),
                    answer=str(rec["answer"]).strip(),
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
        qid = str(dict(ex).get("id") or dict(ex).get("ID") or f"aime2024_{split}_{idx}")
        queries.append(
            Query(
                id=qid,
                question=rec["question"],
                answer=rec["answer"],
                choices=None,
            )
        )

    if jsonl_path is not None:
        write_aime2024_normalized_jsonl(norm_rows, jsonl_path)

    return queries
