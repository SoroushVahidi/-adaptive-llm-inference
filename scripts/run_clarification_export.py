"""Export the manuscript clarification table reconciling routing strategies.

Usage
-----
    python scripts/run_clarification_export.py [--output-dir PATH]

Outputs (written to ``outputs/manuscript_support/`` by default):
- ``clarification_table.csv``   — tidy format (strategy × regime)
- ``clarification_wide.csv``    — wide format (one row per regime)
- ``clarification_table.tex``   — LaTeX-ready booktabs table
- ``clarification_table.json``  — machine-readable JSON
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.clarification_export import (
    BUDGET_CURVES_CSV,
    MAIN_RESULTS_CSV,
    ORACLE_CSV,
    REGIME_FILES,
    run_clarification_export,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export manuscript clarification table (always-reasoning vs "
        "best-adaptive vs oracle vs budget frontier)."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/manuscript_support",
        help="Directory to write output files (default: outputs/manuscript_support)",
    )
    args = parser.parse_args()

    print("Building manuscript clarification table …")
    print(f"  Regimes          : {', '.join(REGIME_FILES.keys())}")
    print(f"  Main results CSV : {MAIN_RESULTS_CSV}")
    print(f"  Oracle CSV       : {ORACLE_CSV}")
    print(f"  Budget curves    : {BUDGET_CURVES_CSV}")

    rows = run_clarification_export(output_dir=args.output_dir)

    print(f"\nResults written to: {args.output_dir}")
    print(
        f"\n  {'Regime':<25} {'Always-Reas':>12} {'BestAdapt':>12} {'Cost':>7} "
        f"{'Oracle':>10} {'BF@1.1':>8} {'BF@1.2':>8}"
    )
    print("  " + "-" * 90)
    for r in rows:
        bf11 = f"{r.budget_frontier_1_1_acc:.3f}" if r.budget_frontier_1_1_acc is not None else "—"
        bf12 = f"{r.budget_frontier_1_2_acc:.3f}" if r.budget_frontier_1_2_acc is not None else "—"
        print(
            f"  {r.regime:<25} {r.always_reasoning_acc:>12.3f} "
            f"{r.best_adaptive_acc:>12.3f} {r.best_adaptive_cost:>7.2f} "
            f"{r.oracle_acc:>10.3f} {bf11:>8} {bf12:>8}"
        )


if __name__ == "__main__":
    main()
