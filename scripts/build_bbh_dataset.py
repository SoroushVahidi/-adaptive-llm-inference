#!/usr/bin/env python3
"""Build normalized BBH JSONL and a small smoke-test sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.answer_extraction import normalize_text_answer  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/bbh_normalized.jsonl")
    p.add_argument("--sample-output", default="data/bbh_sample.jsonl")
    p.add_argument("--sample-size", type=int, default=64)
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args()

    rows: list[dict] = []
    for task in sorted(get_dataset_config_names("lukaemon/bbh")):
        ds = load_dataset("lukaemon/bbh", task, split="test", cache_dir=args.cache_dir)
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

    out = Path(args.output)
    sample_out = Path(args.sample_output)
    _write_jsonl(out, rows)
    _write_jsonl(sample_out, rows[: args.sample_size])
    print(
        json.dumps(
            {
                "dataset": "bbh",
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
