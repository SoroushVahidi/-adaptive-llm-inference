#!/usr/bin/env python3
"""Build first real GSM8K routing dataset from live strategy outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_real_routing_dataset import BuildConfig, build_real_routing_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k-data-file", default="data/gsm8k_uploaded_normalized.jsonl")
    parser.add_argument("--subset-size", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/real_routing_dataset")
    parser.add_argument("--output-dataset-csv", default="data/real_gsm8k_routing_dataset.csv")
    parser.add_argument("--model-name", default="gpt-4o-mini")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_real_routing_dataset(
        BuildConfig(
            gsm8k_data_file=args.gsm8k_data_file,
            subset_size=args.subset_size,
            output_dir=args.output_dir,
            output_dataset_csv=args.output_dataset_csv,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"summary_json={result['summary_path']}")
    print(f"per_query_csv={result['per_query_csv']}")
    print(f"dataset_csv={result['dataset_csv']}")


if __name__ == "__main__":
    main()
