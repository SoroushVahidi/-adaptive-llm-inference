#!/usr/bin/env python3
"""Run the feature gap analysis for revise-helps cases.

Usage:
    python3 scripts/run_feature_gap_analysis.py

Reads from:
    outputs/oracle_subset_eval/per_query_matrix.csv
    outputs/oracle_subset_eval/oracle_assignments.csv
    outputs/revise_case_analysis/case_table.csv          (optional)
    outputs/revise_case_analysis/category_summary.csv    (optional)
    outputs/adaptive_policy_v3/per_query_results.csv     (optional)
    outputs/adaptive_policy_v4/per_query_results.csv     (optional)

Writes to:
    outputs/feature_gap_analysis/group_feature_summary.csv
    outputs/feature_gap_analysis/missed_revise_cases.csv
    outputs/feature_gap_analysis/pattern_notes.json

All input files are optional; missing files are handled gracefully.
The script always produces output (with empty group sizes when no data
is available) so that downstream tests can run without live oracle data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.feature_gap_analysis import run_feature_gap_analysis

# ---------------------------------------------------------------------------
# Default input / output paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

_ORACLE_ASSIGNMENTS = _REPO_ROOT / "outputs/oracle_subset_eval/oracle_assignments.csv"
_PER_QUERY_MATRIX = _REPO_ROOT / "outputs/oracle_subset_eval/per_query_matrix.csv"
_CASE_TABLE = _REPO_ROOT / "outputs/revise_case_analysis/case_table.csv"
_CATEGORY_SUMMARY = _REPO_ROOT / "outputs/revise_case_analysis/category_summary.csv"
_V3_RESULTS = _REPO_ROOT / "outputs/adaptive_policy_v3/per_query_results.csv"
_V4_RESULTS = _REPO_ROOT / "outputs/adaptive_policy_v4/per_query_results.csv"
_OUTPUT_DIR = _REPO_ROOT / "outputs/feature_gap_analysis"


def main() -> None:
    print("=== Feature Gap Analysis: Revise-Helps Cases ===")
    print()

    # Report which input files are present
    inputs = {
        "oracle_assignments": _ORACLE_ASSIGNMENTS,
        "per_query_matrix": _PER_QUERY_MATRIX,
        "case_table": _CASE_TABLE,
        "category_summary": _CATEGORY_SUMMARY,
        "v3_results": _V3_RESULTS,
        "v4_results": _V4_RESULTS,
    }
    for name, path in inputs.items():
        status = "✓ found" if path.exists() else "✗ missing (skipped)"
        print(f"  [{status:16s}]  {path.relative_to(_REPO_ROOT)}")
    print()

    result = run_feature_gap_analysis(
        oracle_assignments_path=_ORACLE_ASSIGNMENTS,
        per_query_matrix_path=_PER_QUERY_MATRIX,
        case_table_path=_CASE_TABLE,
        category_summary_path=_CATEGORY_SUMMARY,
        v3_results_path=_V3_RESULTS,
        v4_results_path=_V4_RESULTS,
        output_dir=_OUTPUT_DIR,
    )

    # ---- Print results summary -------------------------------------------
    print("Group sizes:")
    for group, n in sorted(result["group_sizes"].items()):
        print(f"  {group:25s}: {n}")
    print()

    n_missed = result["n_missed_revise_cases"]
    n_revise = result["group_sizes"].get("revise_helps", 0)
    print(f"Missed revise cases (revise_helps not caught by v3/v4): {n_missed}/{n_revise}")
    print()

    # ---- Print top wording-trap gaps -------------------------------------
    wt_gaps = result["pattern_notes"].get("wording_trap_feature_gaps", [])
    if wt_gaps:
        print("Top wording-trap feature gaps (revise_helps vs reasoning_enough):")
        for gap in wt_gaps[:5]:
            print(
                f"  {gap['feature']:40s}  "
                f"revise_helps={gap['revise_helps_rate']:.3f}  "
                f"reasoning_enough={gap['reasoning_enough_rate']:.3f}  "
                f"gap={gap['gap']:+.3f}"
            )
        print()

    # ---- Print current feature failures ----------------------------------
    failures = result["pattern_notes"].get("current_feature_failures", [])
    if failures:
        print("Current feature failures:")
        for f in failures:
            print(f"  • {f}")
        print()

    # ---- Print suggested direction ----------------------------------------
    direction = result["pattern_notes"].get("suggested_direction", "")
    if direction:
        print(f"Suggested direction:\n  {direction}")
        print()

    # ---- Print output paths ----------------------------------------------
    print("Output files written:")
    for key, path in result["output_paths"].items():
        print(f"  {key}: {path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
