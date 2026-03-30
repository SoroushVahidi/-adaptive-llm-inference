"""GPQA Diamond-style multiple choice (public HF mirrors)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from src.datasets.gsm8k import Query


@dataclass
class GPQAQuery(Query):
    """Query with gold as a single letter A–D (``answer`` and ``correct_option`` match)."""

    correct_option: str = ""


def try_load_gpqa_diamond_hf(
    max_samples: Optional[int] = None,
    cache_dir: str = "data",
) -> tuple[list[GPQAQuery], str, list[dict[str, str]]]:
    """Try public GPQA Diamond uploads on the Hub (not the gated idavidrein/gpqa)."""
    candidates = [
        "aradhye/gpqa_diamond",
        "nichenshun/gpqa_diamond",
    ]
    errors: list[dict[str, str]] = []
    for name in candidates:
        try:
            ds = load_dataset(name, split="train", cache_dir=cache_dir)
            queries: list[GPQAQuery] = []
            n = len(ds) if max_samples is None else min(len(ds), max_samples)
            for i in range(n):
                row = dict(ds[i])
                problem = str(row.get("problem") or row.get("question") or "")
                ans = str(row.get("answer") or row.get("correct") or "").strip()
                qid = str(row.get("id") or row.get("question_id") or f"gpqa_{i}")
                if not problem:
                    errors.append(
                        {
                            "source": name,
                            "error_type": "schema",
                            "error": f"Row {i} missing problem field; keys={list(row.keys())}",
                        }
                    )
                    continue
                letter = ans.strip().upper()[:1]
                queries.append(
                    GPQAQuery(id=qid, question=problem, answer=letter, correct_option=letter)
                )
            if queries:
                return queries, name, []
        except Exception as exc:  # noqa: BLE001
            errors.append({"source": name, "error_type": type(exc).__name__, "error": str(exc)})
    return [], "", errors


def write_gpqa_normalized_jsonl(queries: list[GPQAQuery], path: str | Path) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for q in queries:
            fh.write(
                json.dumps(
                    {
                        "question_id": q.id,
                        "question": q.question,
                        "correct_option": q.correct_option,
                        "options": "embedded_in_question",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
