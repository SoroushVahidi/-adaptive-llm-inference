#!/usr/bin/env python3
"""Run offline role-signal calibration analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.role_calibration import run_role_calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run role calibration analysis")
    parser.add_argument("--benchmark", default="data/consistency_benchmark.json")
    parser.add_argument("--output-dir", default="outputs/role_calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run_role_calibration(args.benchmark, args.output_dir)
    evaluation = out["evaluation"]
    printable = {
        "num_questions": evaluation["num_questions"],
        "num_candidates": evaluation["num_candidates"],
        "variant_metrics": evaluation["variant_metrics"],
        "false_positive_by_signal": out["false_positive_analysis"]["false_positives_by_signal"],
        "output_paths": {
            k: v for k, v in out.items() if k.endswith("_csv") or k.endswith("_json")
        },
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
