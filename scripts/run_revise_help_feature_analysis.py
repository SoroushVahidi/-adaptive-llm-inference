#!/usr/bin/env python3
"""Run the revise-help feature analysis using target-quantity / wording-trap features.

Usage:
    python3 scripts/run_revise_help_feature_analysis.py

Reads from (all optional — missing files are handled gracefully):
    outputs/oracle_subset_eval/per_query_matrix.csv
    outputs/oracle_subset_eval/oracle_assignments.csv
    outputs/revise_case_analysis/case_table.csv
    src/datasets/bundled/gsm8k_test_sample.json     (fallback for question text)

Writes to:
    outputs/revise_help_feature_analysis/group_feature_rates.csv
    outputs/revise_help_feature_analysis/feature_differences.csv
    outputs/revise_help_feature_analysis/query_feature_table.csv
    outputs/revise_help_feature_analysis/example_cases.json

See docs/REVISE_HELP_FEATURE_ANALYSIS.md for full design rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.revise_help_feature_analysis import (
    GROUP_DIRECT_ALREADY_ENOUGH,
    GROUP_REASONING_ENOUGH,
    GROUP_REVISE_HELPS,
    TARGET_QUANTITY_FEATURES,
    run_revise_help_feature_analysis,
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

_ORACLE_ASSIGNMENTS = _REPO_ROOT / "outputs/oracle_subset_eval/oracle_assignments.csv"
_PER_QUERY_MATRIX = _REPO_ROOT / "outputs/oracle_subset_eval/per_query_matrix.csv"
_CASE_TABLE = _REPO_ROOT / "outputs/revise_case_analysis/case_table.csv"
_BUNDLED_GSM8K = _REPO_ROOT / "src/datasets/bundled/gsm8k_test_sample.json"
_OUTPUT_DIR = _REPO_ROOT / "outputs/revise_help_feature_analysis"

_SECTION = "=" * 70


def main() -> None:
    print(_SECTION)
    print("  Revise-Help Feature Analysis — Target-Quantity Integration")
    print(_SECTION)
    print()

    # ---- Report input file availability ------------------------------------
    inputs = {
        "oracle_assignments":  _ORACLE_ASSIGNMENTS,
        "per_query_matrix":    _PER_QUERY_MATRIX,
        "case_table":          _CASE_TABLE,
        "bundled_gsm8k":       _BUNDLED_GSM8K,
    }
    for name, path in inputs.items():
        status = "✓ found" if path.exists() else "✗ missing (skipped)"
        try:
            rel = path.relative_to(_REPO_ROOT)
        except ValueError:
            rel = path
        print(f"  [{status:16s}]  {rel}")
    print()

    # ---- Run analysis -------------------------------------------------------
    result = run_revise_help_feature_analysis(
        oracle_assignments_path=_ORACLE_ASSIGNMENTS,
        per_query_matrix_path=_PER_QUERY_MATRIX,
        case_table_path=_CASE_TABLE,
        bundled_gsm8k_path=_BUNDLED_GSM8K,
        output_dir=_OUTPUT_DIR,
    )

    # ---- Group sizes --------------------------------------------------------
    print("Group sizes:")
    for grp in [
        GROUP_REVISE_HELPS,
        GROUP_DIRECT_ALREADY_ENOUGH,
        "unique_other_strategy_case",
        "revise_not_enough",
        "reasoning_enough",
    ]:
        n = result["group_sizes"].get(grp, 0)
        print(f"  {grp:35s}: {n}")
    print()

    # ---- Top separating features -------------------------------------------
    top = result["top_separating_features"]
    if top:
        print("Top separating features (revise_helps vs direct_already_enough):")
        print(f"  {'Feature':<40}  {'revise_helps':>12}  {'direct_enough':>13}  {'diff':>8}")
        print(f"  {'-'*40}  {'-'*12}  {'-'*13}  {'-'*8}")
        for r in top:
            print(
                f"  {r['feature']:<40}  {r['focus_rate']:>12.4f}  "
                f"{r['baseline_rate']:>13.4f}  {r['difference']:>+8.4f}"
            )
        print()
    else:
        print("(No feature difference data — no oracle data present.)")
        print()

    # ---- Feature rates by group for target-quantity features ----------------
    rates = result["feature_rates"]
    rate_by_group = {r["group"]: r for r in rates}
    groups_to_show = [
        GROUP_REVISE_HELPS,
        GROUP_DIRECT_ALREADY_ENOUGH,
        GROUP_REASONING_ENOUGH,
    ]
    active_groups = [g for g in groups_to_show if rate_by_group.get(g, {}).get("n", 0) > 0]

    if active_groups:
        print("Target-quantity feature rates by group:")
        header_parts = ["  " + f"{'Feature':<40}"]
        for g in active_groups:
            n = rate_by_group[g]["n"]
            label = f"{g[:14]}(n={n})"
            header_parts.append(f"{label:>20}")
        print("".join(header_parts))
        print("  " + "-" * (40 + 20 * len(active_groups)))
        for feat in TARGET_QUANTITY_FEATURES:
            key = f"{feat}_rate"
            parts = [f"  {feat:<40}"]
            for g in active_groups:
                val = rate_by_group.get(g, {}).get(key, 0.0)
                parts.append(f"{val:>20.4f}")
            print("".join(parts))
        print()

    # ---- Example cases -------------------------------------------------------
    examples = result["example_cases"]
    if examples:
        print(f"Example queries from '{GROUP_REVISE_HELPS}' group:")
        for i, ex in enumerate(examples, 1):
            q = ex["question_text"]
            fired = ex["target_quantity_features_fired"]
            print(f"  [{i}] {q[:90]}{'...' if len(q) > 90 else ''}")
            print(f"       Features fired: {', '.join(fired) if fired else '(none)'}")
        print()

    # ---- Recommendation -------------------------------------------------------
    rec = result["recommendation"]
    print(f"Recommendation: {rec['recommendation'].upper()}")
    print(f"  {rec['justification']}")
    if rec["strongest_separating_signals"]:
        print(f"  Strongest signals: {', '.join(rec['strongest_separating_signals'])}")
    print()

    # ---- Output paths ---------------------------------------------------------
    print("Output files written:")
    for key, path in result["output_paths"].items():
        try:
            rel = Path(path).relative_to(_REPO_ROOT)
        except ValueError:
            rel = Path(path)
        print(f"  {rel}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
