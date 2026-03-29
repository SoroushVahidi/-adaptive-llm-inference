#!/usr/bin/env python3
"""Build real MATH500 routing dataset (reasoning_greedy + direct_plus_revise)."""

from __future__ import annotations

import argparse
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset-size", type=int, default=100)
    p.add_argument("--math500-data-file", default="data/math500_uploaded_normalized.jsonl")
    p.add_argument("--output-dir", default="outputs/real_math500_routing")
    p.add_argument("--output-dataset-csv", default="data/real_math500_routing_dataset.csv")
    p.add_argument("--model-name", default="gpt-4o-mini")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()
    mfile = args.math500_data_file
    if not Path(mfile).exists():
        mfile = None

    result = build_real_routing_dataset(
        BuildConfig(
            dataset="math500",
            math500_data_file=mfile,
            subset_size=args.subset_size,
            output_dir=args.output_dir,
            output_dataset_csv=args.output_dataset_csv,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            summary_filename="math500_run_summary.json",
            per_query_csv_filename="per_query_outputs.csv",
            regime_label="math500",
        )
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
