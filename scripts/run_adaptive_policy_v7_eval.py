#!/usr/bin/env python3
"""Run offline v7 evaluation: false-positive fixtures, recall proxies, probe snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.adaptive_policy_v7_eval import run_adaptive_policy_v7_offline_eval  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        default="outputs/adaptive_policy_v7",
        help="Directory for CSV/JSON outputs (relative to repo root)",
    )
    args = p.parse_args()
    result = run_adaptive_policy_v7_offline_eval(output_dir=args.output_dir)
    print("--- adaptive_policy_v7 offline eval ---")
    for k, v in result.items():
        if k.endswith("_rows"):
            continue
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
