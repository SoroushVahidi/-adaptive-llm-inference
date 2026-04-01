#!/usr/bin/env python3
"""Build normalized MuSR JSONL and a small smoke-test sample."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_choices(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    txt = str(raw).strip()
    parsed = ast.literal_eval(txt)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list-like choices, got {type(parsed).__name__}")
    return [str(x) for x in parsed]


def _letter(i: int) -> str:
    return chr(ord("A") + i)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/musr_normalized.jsonl")
    p.add_argument("--sample-output", default="data/musr_sample.jsonl")
    p.add_argument("--sample-size", type=int, default=64)
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args()

    dsd = load_dataset("TAUR-Lab/MuSR", cache_dir=args.cache_dir)
    rows: list[dict] = []
    for split_name in sorted(dsd.keys()):
        split_ds = dsd[split_name]
        for i, ex in enumerate(split_ds):
            choices = _parse_choices(ex.get("choices"))
            ans_idx = int(ex.get("answer_index", -1))
            if not (0 <= ans_idx < len(choices)):
                raise ValueError(f"Schema mismatch at {split_name}:{i} answer_index out of range")
            q = f"{str(ex.get('narrative', '')).strip()}\n\nQuestion: {str(ex.get('question', '')).strip()}"
            rows.append(
                {
                    "dataset": "musr",
                    "question_id": f"musr_{split_name}_{i}",
                    "question": q,
                    "options": choices,
                    "answer": _letter(ans_idx),
                    "answer_format": "multiple_choice",
                    "category": split_name,
                    "source_split": split_name,
                    "metadata": {
                        "subtask": split_name,
                        "answer_choice": ex.get("answer_choice", ""),
                    },
                }
            )

    out = Path(args.output)
    sample_out = Path(args.sample_output)
    _write_jsonl(out, rows)
    _write_jsonl(sample_out, rows[: args.sample_size])
    print(
        json.dumps(
            {
                "dataset": "musr",
                "rows": len(rows),
                "output": str(out),
                "output_bytes": out.stat().st_size,
                "sample_rows": min(args.sample_size, len(rows)),
                "sample_output": str(sample_out),
                "sample_output_bytes": sample_out.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
