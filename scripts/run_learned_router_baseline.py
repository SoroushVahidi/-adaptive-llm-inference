"""Run the lightweight learned-router baseline on all four manuscript regimes.

Usage
-----
    python scripts/run_learned_router_baseline.py [--output-dir PATH] [--cv-folds INT]

Outputs (written to ``outputs/baselines/learned_router/`` by default):
- ``learned_router_summary.csv``  — per-regime, per-model metrics
- ``learned_router_summary.json`` — same as JSON
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.learned_router_baseline import (
    CV_FOLDS,
    REGIME_FILES,
    run_all_regimes,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learned-router baseline (logistic regression + decision tree) "
        "on the four main manuscript regimes."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baselines/learned_router",
        help="Directory to write output files (default: outputs/baselines/learned_router)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS,
        help=f"Number of cross-validation folds (default: {CV_FOLDS})",
    )
    args = parser.parse_args()

    print("Running learned-router baseline …")
    print(f"  Regimes  : {', '.join(REGIME_FILES.keys())}")
    print(f"  CV folds : {args.cv_folds}")

    results = run_all_regimes(
        output_dir=args.output_dir,
        cv_folds=args.cv_folds,
    )

    print(f"\nResults written to: {args.output_dir}")
    print(
        f"\n  {'Regime':<25} {'Model':<22} {'Accuracy':>10} {'AvgCost':>10} "
        f"{'RevRate':>10} {'Degen.':>8} {'Note'}"
    )
    print("  " + "-" * 100)
    for r in results:
        print(
            f"  {r.regime:<25} {r.model_name:<22} {r.accuracy:>10.3f} "
            f"{r.avg_cost:>10.3f} {r.revise_rate:>10.3f} {str(r.degenerate):>8}  {r.note}"
        )


if __name__ == "__main__":
    main()
