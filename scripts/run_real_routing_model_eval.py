#!/usr/bin/env python3
"""Run first learned routing model eval on real GSM8K routing rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.real_routing_model_eval import run_real_routing_model_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", default="data/real_gsm8k_routing_dataset.csv")
    parser.add_argument("--output-dir", default="outputs/real_routing_model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_real_routing_model_eval(dataset_csv=args.dataset_csv, output_dir=args.output_dir)
    print(json.dumps(result["summary"], indent=2))
    print(f"summary_json={result['summary_path']}")


if __name__ == "__main__":
    main()
