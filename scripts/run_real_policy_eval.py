#!/usr/bin/env python3
"""Evaluate v5/v6/v7 (and fixed baselines) on data/real_gsm8k_routing_dataset.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.real_policy_eval import run_real_policy_eval  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-csv", default="data/real_gsm8k_routing_dataset.csv")
    p.add_argument("--output-dir", default="outputs/real_policy_eval")
    p.add_argument(
        "--conf-target-cost",
        type=float,
        default=1.2,
        help="Target avg cost for confidence-threshold operating point (enriched CSVs only)",
    )
    args = p.parse_args()
    r = run_real_policy_eval(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        conf_target_cost=args.conf_target_cost,
    )
    print(json.dumps(r["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
