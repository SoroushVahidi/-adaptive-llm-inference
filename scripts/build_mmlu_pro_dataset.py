#!/usr/bin/env python3
"""Build normalized MMLU-Pro JSONL and a small smoke-test sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/mmlu_pro_normalized.jsonl")
    p.add_argument("--sample-output", default="data/mmlu_pro_sample.jsonl")
    p.add_argument("--sample-size", type=int, default=64)
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args()

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", cache_dir=args.cache_dir)
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        options = ex.get("options")
        if not isinstance(options, list):
            raise ValueError(f"Schema mismatch at row {i}: options must be list")
        rows.append(
            {
                "dataset": "mmlu_pro",
                "question_id": str(ex.get("question_id", f"mmlu_pro_test_{i}")),
                "question": str(ex.get("question", "")),
                "options": [str(x) for x in options],
                "answer": str(ex.get("answer", "")).strip().upper(),
                "answer_format": "multiple_choice",
                "category": str(ex.get("category", "")),
                "source_split": "test",
                "metadata": {
                    "answer_index": int(ex.get("answer_index", -1)),
                    "src": ex.get("src", ""),
                    "cot_content_present": bool(str(ex.get("cot_content", "")).strip()),
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
                "dataset": "mmlu_pro",
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
