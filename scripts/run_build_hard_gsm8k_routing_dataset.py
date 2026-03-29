#!/usr/bin/env python3
"""Build routing dataset for hard GSM8K subset (after run_select_hard_gsm8k)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_real_routing_dataset import (  # noqa: E402
    BuildConfig,
    build_real_routing_dataset,
)
from src.datasets.gsm8k import Query  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--selection-csv",
        default="outputs/hard_regime_selection/hard_gsm8k_selection.csv",
    )
    p.add_argument("--output-dir", default="outputs/real_hard_gsm8k_routing")
    p.add_argument("--output-dataset-csv", default="data/real_hard_gsm8k_routing_dataset.csv")
    p.add_argument("--model-name", default="gpt-4o-mini")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=90)
    args = p.parse_args()

    sel = Path(args.selection_csv)
    if not sel.exists():
        print(f"BLOCKED: missing {sel}")
        sys.exit(1)

    queries: list[Query] = []
    with sel.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            queries.append(
                Query(
                    id=str(row["question_id"]),
                    question=str(row["question"]),
                    answer=str(row["gold_answer"]),
                )
            )

    result = build_real_routing_dataset(
        BuildConfig(
            dataset="custom",
            queries_override=queries,
            subset_size=len(queries),
            output_dir=args.output_dir,
            output_dataset_csv=args.output_dataset_csv,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            answer_match_mode="numeric",
            summary_filename="hard_gsm8k_run_summary.json",
            per_query_csv_filename="per_query_outputs.csv",
            regime_label="hard_gsm8k",
        )
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
