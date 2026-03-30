#!/usr/bin/env python3
"""Run AIME-2024 policy evaluation (small experiment pass).

Reads the committed ``data/real_aime2024_routing_dataset.csv`` and evaluates:
- cheap baseline (reasoning_greedy)
- always-revise baseline (direct_plus_revise)
- adaptive policies v5 / v6 / v7
- confidence-threshold router baseline
- oracle upper bound

No API calls required — all data is already committed.

Usage::

    python scripts/run_small_pass_aime_eval.py
    python scripts/run_small_pass_aime_eval.py --dataset-csv data/real_aime2024_routing_dataset.csv
    python scripts/run_small_pass_aime_eval.py --output-dir outputs/small_pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.small_pass_aime_eval import run_small_pass_aime_eval  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-csv",
        default="data/real_aime2024_routing_dataset.csv",
        help="Path to AIME-2024 routing dataset CSV (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/small_pass",
        help="Directory for output files (default: %(default)s)",
    )
    p.add_argument(
        "--conf-target-cost",
        type=float,
        default=1.2,
        help="Target avg-cost for confidence-router operating-point (default: %(default)s)",
    )
    args = p.parse_args()

    result = run_small_pass_aime_eval(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        conf_target_cost=args.conf_target_cost,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"].get("run_status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
