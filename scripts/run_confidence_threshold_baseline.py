"""Run the confidence-threshold routing baseline on all four manuscript regimes.

Usage
-----
    python scripts/run_confidence_threshold_baseline.py [--output-dir PATH] [--target-cost FLOAT]

Outputs (written to ``outputs/baselines/confidence_threshold/`` by default):
- ``confidence_threshold_sweep.csv``  — full threshold sweep per regime
- ``confidence_threshold_summary.csv`` — chosen operating point per regime
- ``confidence_threshold_summary.json`` — same as JSON
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.confidence_threshold_router import (
    REGIME_FILES,
    sweep_and_summarise,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confidence-threshold routing baseline for manuscript regimes."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baselines/confidence_threshold",
        help="Directory to write output files (default: outputs/baselines/confidence_threshold)",
    )
    parser.add_argument(
        "--target-cost",
        type=float,
        default=1.2,
        help="Target average cost for operating-point selection (default: 1.2)",
    )
    args = parser.parse_args()

    print("Running confidence-threshold routing baseline …")
    print(f"  Regimes : {', '.join(REGIME_FILES.keys())}")
    print(f"  Target cost for operating point: {args.target_cost}")

    results = sweep_and_summarise(
        output_dir=args.output_dir,
        target_cost=args.target_cost,
    )

    print(f"\nResults written to: {args.output_dir}")
    print("\nPer-regime summary at operating point:")
    print(f"  {'Regime':<25} {'Threshold':>10} {'Accuracy':>10} {'AvgCost':>10} {'RevRate':>10}")
    print("  " + "-" * 65)
    for r in results:
        print(
            f"  {r.regime:<25} {r.operating_threshold:>10.2f} "
            f"{r.accuracy:>10.3f} {r.avg_cost:>10.3f} {r.revise_rate:>10.3f}"
        )


if __name__ == "__main__":
    main()
