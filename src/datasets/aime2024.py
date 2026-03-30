"""AIME 2024 loaders (Hugging Face mirrors)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from src.datasets.gsm8k import Query


def _record_to_query(rec: dict[str, Any], idx: int) -> Query:
    if "problem" in rec:
        qtext = str(rec["problem"])
        ans = str(rec.get("answer", "")).strip()
        qid = str(rec.get("id", f"aime2024_{idx}"))
    elif "Problem" in rec:
        qtext = str(rec["Problem"])
        raw_ans = rec.get("Answer", "")
        ans = str(raw_ans).strip() if raw_ans is not None else ""
        qid = str(rec.get("ID", f"aime2024_{idx}"))
    else:
        raise ValueError(f"AIME record missing problem text: keys={list(rec.keys())}")
    return Query(id=qid, question=qtext, answer=ans)


def try_load_aime2024_hf(
    max_samples: Optional[int] = None,
    cache_dir: str = "data",
) -> tuple[list[Query], str, list[dict[str, str]]]:
    """Load AIME 2024 from Hugging Face. Tries sources in order.

    Returns:
        queries, winning_source, blocker_log (empty on success).
    """
    sources = ["HuggingFaceH4/aime_2024", "Maxwell-Jia/AIME_2024"]
    errors: list[dict[str, str]] = []
    for name in sources:
        try:
            ds = load_dataset(name, split="train", cache_dir=cache_dir)
            records = [dict(ds[i]) for i in range(len(ds))]
            if max_samples is not None:
                records = records[:max_samples]
            queries = [_record_to_query(r, i) for i, r in enumerate(records)]
            return queries, name, []
        except Exception as exc:  # noqa: BLE001
            errors.append({"source": name, "error_type": type(exc).__name__, "error": str(exc)})
    return [], "", errors


def write_aime2024_jsonl(queries: list[Query], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for q in queries:
            fh.write(
                json.dumps(
                    {"question_id": q.id, "question": q.question, "answer": q.answer},
                    ensure_ascii=False,
                )
                + "\n"
            )
