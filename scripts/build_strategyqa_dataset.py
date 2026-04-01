#!/usr/bin/env python3
"""Build normalized StrategyQA JSONL and a small smoke-test sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.answer_extraction import normalize_boolean_answer  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/strategyqa_normalized.jsonl")
    p.add_argument("--sample-output", default="data/strategyqa_sample.jsonl")
    p.add_argument("--sample-size", type=int, default=64)
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args()

    rows: list[dict] = []
    for split in ["train", "test"]:
        ds = load_dataset("ChilleD/StrategyQA", split=split, cache_dir=args.cache_dir)
        for i, ex in enumerate(ds):
            answer = normalize_boolean_answer(str(ex.get("answer", "")))
            if answer not in {"true", "false"}:
                raise ValueError(f"Schema mismatch at {split}:{i}: invalid boolean answer")
            rows.append(
                {
                    "dataset": "strategyqa",
                    "question_id": str(ex.get("qid", f"strategyqa_{split}_{i}")),
                    "question": str(ex.get("question", "")),
                    "options": None,
                    "answer": answer,
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

    out = Path(args.output)
    sample_out = Path(args.sample_output)
    _write_jsonl(out, rows)
    _write_jsonl(sample_out, rows[: args.sample_size])
    print(
        json.dumps(
            {
                "dataset": "strategyqa",
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
