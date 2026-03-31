#!/usr/bin/env python3
"""Generate enhanced manuscript-quality tables, figures, and graphic abstract.

Usage
-----
    python3 scripts/generate_enhanced_artifacts.py [--repo-root .] [--only <comma-sep>]

Available modules (--only values):
    tables          All four enhanced tables
    figures         All six enhanced figures
    graphic_abstract  One-page graphic abstract

Examples
--------
    python3 scripts/generate_enhanced_artifacts.py
    python3 scripts/generate_enhanced_artifacts.py --only tables
    python3 scripts/generate_enhanced_artifacts.py --only figures,graphic_abstract
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup_pythonpath(repo_root: Path) -> None:
    """Ensure repo root is on sys.path so src/ is importable."""
    repo_str = str(repo_root.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _run_tables(root: Path) -> None:
    from src.paper_artifacts.exports_enhanced import (
        export_oracle_gap_table,
        export_cost_efficiency_table,
        export_policy_ranking_table,
        export_aime_supplementary_table,
    )

    tables_dir = root / "outputs" / "paper_tables_enhanced"
    print("\n[TABLES] Exporting to:", tables_dir)

    # Oracle-gap table
    p = export_oracle_gap_table(root, tables_dir)
    print(f"  ✅  oracle_gap_table:         {p.relative_to(root)}")

    # Cost-efficiency table
    p = export_cost_efficiency_table(root, tables_dir)
    print(f"  ✅  cost_efficiency_gain:      {p.relative_to(root)}")

    # Policy ranking table
    p = export_policy_ranking_table(root, tables_dir)
    print(f"  ✅  policy_ranking_table:      {p.relative_to(root)}")

    # AIME supplementary table
    p = export_aime_supplementary_table(root, tables_dir)
    if p is not None:
        print(f"  ✅  aime_supplementary_table: {p.relative_to(root)}")
    else:
        print("  ⚠️   aime_supplementary_table: SKIPPED (source data not found)")


def _run_figures(root: Path) -> None:
    from src.paper_artifacts.exports_enhanced import (
        figure_oracle_gap_bar,
        figure_revise_helpful_vs_gain_scatter,
        figure_cost_accuracy_pareto,
        figure_policy_revise_rate,
        figure_confidence_baseline_comparison,
        figure_aime_limit_case,
    )

    figs_dir = root / "outputs" / "paper_figures_enhanced"
    print("\n[FIGURES] Exporting to:", figs_dir)

    p = figure_oracle_gap_bar(root, figs_dir)
    print(f"  ✅  oracle_gap_bar_chart:            {p.relative_to(root)}")

    p = figure_revise_helpful_vs_gain_scatter(root, figs_dir)
    print(f"  ✅  revise_helpful_vs_gain_scatter:  {p.relative_to(root)}")

    p = figure_cost_accuracy_pareto(root, figs_dir)
    print(f"  ✅  cost_accuracy_pareto:             {p.relative_to(root)}")

    p = figure_policy_revise_rate(root, figs_dir)
    print(f"  ✅  policy_revise_rate_comparison:   {p.relative_to(root)}")

    p = figure_confidence_baseline_comparison(root, figs_dir)
    print(f"  ✅  confidence_baseline_comparison:  {p.relative_to(root)}")

    p = figure_aime_limit_case(root, figs_dir)
    if p is not None:
        print(f"  ✅  aime_limit_case:                 {p.relative_to(root)}")
    else:
        print("  ⚠️   aime_limit_case: SKIPPED (source data not found)")


def _run_graphic_abstract(root: Path) -> None:
    from src.paper_artifacts.exports_enhanced import (
        figure_graphic_abstract,
        write_graphic_abstract_notes,
    )

    ga_dir = root / "outputs" / "graphic_abstract"
    print("\n[GRAPHIC ABSTRACT] Exporting to:", ga_dir)

    p = figure_graphic_abstract(root, ga_dir)
    print(f"  ✅  graphic_abstract:       {p.relative_to(root)}")
    pdf = p.with_suffix(".pdf")
    if pdf.is_file():
        print(f"  ✅  graphic_abstract (PDF): {pdf.relative_to(root)}")

    p = write_graphic_abstract_notes(ga_dir)
    print(f"  ✅  graphic_abstract_notes: {p.relative_to(root)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--only", default="",
                        help="Comma-separated subset: tables,figures,graphic_abstract")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    _setup_pythonpath(root)

    modules = {m.strip() for m in args.only.split(",") if m.strip()} if args.only else {
        "tables", "figures", "graphic_abstract"
    }

    print(f"Repo root : {root}")
    print(f"Modules   : {sorted(modules)}")

    if "tables" in modules:
        _run_tables(root)

    if "figures" in modules:
        _run_figures(root)

    if "graphic_abstract" in modules:
        _run_graphic_abstract(root)

    print("\nDone. All enhanced artifacts written.")


if __name__ == "__main__":
    main()
