"""Run bootstrap uncertainty analysis for the main manuscript comparisons.

Usage
-----
    python scripts/run_uncertainty_analysis.py [--output-dir PATH] [--n-bootstrap INT]

Outputs (written to ``outputs/manuscript_support/`` by default):
- ``uncertainty_analysis.json``         — full bootstrap results (all comparisons)
- ``uncertainty_analysis_summary.csv``  — flat CSV for manuscript tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.uncertainty_analysis import (
    N_BOOTSTRAP_DEFAULT,
    REGIME_FILES,
    run_uncertainty_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for key manuscript comparisons."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/manuscript_support",
        help="Directory to write output files (default: outputs/manuscript_support)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP_DEFAULT,
        help=f"Number of bootstrap replicates (default: {N_BOOTSTRAP_DEFAULT})",
    )
    args = parser.parse_args()

    print("Running bootstrap uncertainty analysis …")
    print(f"  Regimes     : {', '.join(REGIME_FILES.keys())}")
    print(f"  Replicates  : {args.n_bootstrap:,}")

    results = run_uncertainty_analysis(
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
    )

    print(f"\nResults written to: {args.output_dir}")
    print(
        f"\n  {'Regime':<25} {'Comparison':<45} "
        f"{'Delta':>8} {'CI lower':>10} {'CI upper':>10} {'Sig.':>6}"
    )
    print("  " + "-" * 110)
    for r in results:
        for c in r.comparisons:
            print(
                f"  {c.regime:<25} {c.comparison:<45} "
                f"{c.observed_delta:>8.4f} {c.ci_lower:>10.4f} {c.ci_upper:>10.4f} "
                f"{'YES' if c.significant_at_95pct else 'no':>6}"
            )


if __name__ == "__main__":
    main()
