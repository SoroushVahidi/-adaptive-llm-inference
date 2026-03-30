"""Manuscript clarification export: reconcile always-reasoning, best adaptive
policy, oracle routing, and budget-frontier values.

This module reads *only* from committed artifact files in the repository and
produces a clean comparison table that helps the manuscript explain the
relationship between:

1. ``always_reasoning``   — baseline: apply first-pass reasoning to all queries.
2. ``best_adaptive``      — best deployable policy (v6 or v7) on each regime.
3. ``oracle``             — oracle routing (knows revise_helpful in advance).
4. ``budget_frontier_1.1`` — oracle budget-frontier accuracy at avg_cost ≤ 1.1.
5. ``budget_frontier_1.2`` — oracle budget-frontier accuracy at avg_cost ≤ 1.2.

All values are drawn directly from the committed experiment outputs (no
re-computation of LLM calls) and from the routing datasets (for per-query
correct indicators).

Output files
------------
``<output_dir>/clarification_table.csv``
    Tidy CSV with one row per regime × strategy.
``<output_dir>/clarification_wide.csv``
    Wide-format CSV (one row per regime, columns per strategy) — easy to paste
    into a manuscript table.
``<output_dir>/clarification_table.tex``
    LaTeX-ready booktabs table (generated if ``tabulate`` is available, or via
    a minimal built-in formatter).

Public API
----------
- ``REGIME_FILES``      — routing dataset CSV paths.
- ``MAIN_RESULTS_CSV``  — path to ``outputs/paper_tables_cleaned/main_results_summary.csv``.
- ``ORACLE_CSV``        — path to ``outputs/paper_tables_cleaned/oracle_routing_eval.csv``.
- ``BUDGET_CURVES_CSV`` — path to ``outputs/paper_tables_cleaned/budget_curves_all_datasets.csv``.
- ``ClarificationRow``  — dataclass for one regime's full comparison.
- ``build_clarification_table(…)`` → list[ClarificationRow]
- ``run_clarification_export(output_dir)``
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

REGIME_FILES: dict[str, str] = {
    "gsm8k_random_100": "data/real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": "data/real_math500_routing_dataset_enriched.csv",
}

MAIN_RESULTS_CSV = Path("outputs/paper_tables_cleaned/main_results_summary.csv")
CROSS_REGIME_CSV = Path("outputs/paper_tables_cleaned/final_cross_regime_summary_fixed.csv")
ORACLE_CSV = Path("outputs/paper_tables_cleaned/oracle_routing_eval.csv")
BUDGET_CURVES_CSV = Path("outputs/paper_tables_cleaned/budget_curves_all_datasets.csv")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClarificationRow:
    """One regime's reconciled comparison across strategies."""

    regime: str
    n: int
    # Always-reasoning
    always_reasoning_acc: float
    always_reasoning_cost: float
    # Best adaptive policy
    best_adaptive_acc: float
    best_adaptive_cost: float
    best_adaptive_name: str
    # Oracle routing
    oracle_acc: float
    oracle_cost: float
    # Budget frontier at 1.1 and 1.2
    budget_frontier_1_1_acc: Optional[float]
    budget_frontier_1_2_acc: Optional[float]

    def to_tidy_rows(self) -> list[dict]:
        """Return a list of tidy (regime, strategy, accuracy, cost) rows."""
        rows = []
        for strategy, acc, cost in [
            ("always_reasoning", self.always_reasoning_acc, self.always_reasoning_cost),
            ("best_adaptive", self.best_adaptive_acc, self.best_adaptive_cost),
            ("oracle", self.oracle_acc, self.oracle_cost),
        ]:
            rows.append(
                {
                    "regime": self.regime,
                    "strategy": strategy,
                    "accuracy": acc,
                    "avg_cost": cost,
                    "n": self.n,
                }
            )
        for budget_label, acc, budget_cost in [
            ("budget_frontier_1.1", self.budget_frontier_1_1_acc, 1.1),
            ("budget_frontier_1.2", self.budget_frontier_1_2_acc, 1.2),
        ]:
            if acc is not None:
                rows.append(
                    {
                        "regime": self.regime,
                        "strategy": budget_label,
                        "accuracy": acc,
                        "avg_cost": budget_cost,
                        "n": self.n,
                    }
                )
        return rows

    def to_wide_dict(self) -> dict:
        return {
            "regime": self.regime,
            "n": self.n,
            "always_reasoning_acc": self.always_reasoning_acc,
            "always_reasoning_cost": self.always_reasoning_cost,
            "best_adaptive_acc": self.best_adaptive_acc,
            "best_adaptive_cost": self.best_adaptive_cost,
            "best_adaptive_name": self.best_adaptive_name,
            "oracle_acc": self.oracle_acc,
            "oracle_cost": self.oracle_cost,
            "budget_frontier_1.1_acc": self.budget_frontier_1_1_acc,
            "budget_frontier_1.2_acc": self.budget_frontier_1_2_acc,
        }


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------


def _load_main_results(path: Path) -> dict[str, dict]:
    """Load main_results_summary.csv keyed by dataset name."""
    df = pd.read_csv(path)
    return {row["dataset"]: row.to_dict() for _, row in df.iterrows()}


def _load_oracle(path: Path) -> dict[str, dict]:
    """Load oracle_routing_eval.csv keyed by dataset name."""
    df = pd.read_csv(path)
    return {row["dataset"]: row.to_dict() for _, row in df.iterrows()}


def _load_budget_curves(path: Path) -> dict[str, pd.DataFrame]:
    """Load budget_curves_all_datasets.csv as per-dataset DataFrames."""
    df = pd.read_csv(path)
    return {name: grp for name, grp in df.groupby("dataset")}


def _budget_frontier_acc(curves_df: pd.DataFrame, target_cost: float) -> Optional[float]:
    """Return accuracy from budget curves at the given target_avg_cost.

    The budget curves are oracle-sorted allocations; we return the accuracy
    at the row whose ``target_avg_cost`` or ``avg_cost`` most closely matches
    *target_cost* (within 0.01).
    """
    # Try exact match on target_avg_cost column first
    for col in ("target_avg_cost", "avg_cost"):
        if col not in curves_df.columns:
            continue
        mask = abs(curves_df[col] - target_cost) < 0.015
        subset = curves_df[mask]
        if not subset.empty:
            return float(subset.iloc[0]["accuracy"])
    return None


# ---------------------------------------------------------------------------
# Build the clarification table
# ---------------------------------------------------------------------------


def build_clarification_table(
    regime_files: dict[str, str] | None = None,
    main_results_csv: Path = MAIN_RESULTS_CSV,
    oracle_csv: Path = ORACLE_CSV,
    budget_curves_csv: Path = BUDGET_CURVES_CSV,
    cross_regime_csv: Path = CROSS_REGIME_CSV,
) -> list[ClarificationRow]:
    """Build one ClarificationRow per regime from committed artifact files."""
    files = regime_files if regime_files is not None else REGIME_FILES

    main_results = _load_main_results(main_results_csv)
    oracle_data = _load_oracle(oracle_csv)
    budget_curves = _load_budget_curves(budget_curves_csv)

    # Load best-policy names from cross-regime summary (has best_policy_name column)
    cross_results: dict[str, dict] = {}
    if cross_regime_csv.exists():
        df_cross = pd.read_csv(cross_regime_csv)
        cross_results = {row["dataset"]: row.to_dict() for _, row in df_cross.iterrows()}

    rows: list[ClarificationRow] = []
    for regime in files:
        mr = main_results.get(regime, {})
        cr = cross_results.get(regime, {})
        oc = oracle_data.get(regime, {})
        bc = budget_curves.get(regime, pd.DataFrame())

        # Always-reasoning: reasoning_acc from main results, cost = 1.0
        always_reasoning_acc = float(mr.get("reasoning_acc", float("nan")))

        # Best adaptive policy — prefer cross_regime_summary for the name
        best_adaptive_acc = float(mr.get("best_policy_acc", float("nan")))
        best_adaptive_cost = float(mr.get("best_policy_cost", float("nan")))
        best_adaptive_name = str(
            cr.get("best_policy_name", mr.get("best_policy_name", "adaptive_policy"))
        )
        if not best_adaptive_name or best_adaptive_name == "nan":
            best_adaptive_name = "adaptive_policy"

        # Oracle
        oracle_acc = float(oc.get("oracle_acc", float("nan")))
        oracle_cost = float(oc.get("oracle_avg_cost", float("nan")))

        # Budget frontier
        bf_1_1 = _budget_frontier_acc(bc, 1.1) if not bc.empty else None
        bf_1_2 = _budget_frontier_acc(bc, 1.2) if not bc.empty else None

        # n from routing dataset
        df_route = pd.read_csv(files[regime])
        n = len(df_route)

        rows.append(
            ClarificationRow(
                regime=regime,
                n=n,
                always_reasoning_acc=always_reasoning_acc,
                always_reasoning_cost=1.0,
                best_adaptive_acc=best_adaptive_acc,
                best_adaptive_cost=best_adaptive_cost,
                best_adaptive_name=best_adaptive_name,
                oracle_acc=oracle_acc,
                oracle_cost=oracle_cost,
                budget_frontier_1_1_acc=bf_1_1,
                budget_frontier_1_2_acc=bf_1_2,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# LaTeX formatter (no external deps)
# ---------------------------------------------------------------------------


def _format_pct(v: Optional[float], digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):  # nan check
        return "—"
    return f"{v * 100:.{digits}f}\\%"


def _build_latex(rows: list[ClarificationRow]) -> str:
    header = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Comparison of routing strategies across the four main regimes. "
        "Budget-frontier values are from the oracle-sorted budget sweep.}\n"
        "\\label{tab:clarification}\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        "Regime & Always-Reas. & Best Adaptive & Cost & Oracle & BF@1.1 & BF@1.2 \\\\\n"
        "\\midrule\n"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.regime} & "
            f"{_format_pct(r.always_reasoning_acc)} & "
            f"{_format_pct(r.best_adaptive_acc)} & "
            f"{r.best_adaptive_cost:.2f}\\times & "
            f"{_format_pct(r.oracle_acc)} & "
            f"{_format_pct(r.budget_frontier_1_1_acc)} & "
            f"{_format_pct(r.budget_frontier_1_2_acc)} \\\\\n"
        )
    lines.append(
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return "".join(lines)


# ---------------------------------------------------------------------------
# Full export runner
# ---------------------------------------------------------------------------


def run_clarification_export(
    regime_files: dict[str, str] | None = None,
    output_dir: str | Path = "outputs/manuscript_support",
    main_results_csv: Path = MAIN_RESULTS_CSV,
    oracle_csv: Path = ORACLE_CSV,
    budget_curves_csv: Path = BUDGET_CURVES_CSV,
    cross_regime_csv: Path = CROSS_REGIME_CSV,
) -> list[ClarificationRow]:
    """Build and save all clarification table outputs.

    Writes
    ------
    ``<output_dir>/clarification_table.csv``     — tidy format
    ``<output_dir>/clarification_wide.csv``       — wide format (one row per regime)
    ``<output_dir>/clarification_table.tex``      — LaTeX booktabs table
    ``<output_dir>/clarification_table.json``     — JSON (machine-readable)
    """
    rows = build_clarification_table(
        regime_files=regime_files,
        main_results_csv=main_results_csv,
        oracle_csv=oracle_csv,
        budget_curves_csv=budget_curves_csv,
        cross_regime_csv=cross_regime_csv,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tidy CSV
    tidy_rows: list[dict] = []
    for r in rows:
        tidy_rows.extend(r.to_tidy_rows())
    tidy_path = out_dir / "clarification_table.csv"
    if tidy_rows:
        with tidy_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(tidy_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tidy_rows)

    # Wide CSV
    wide_rows = [r.to_wide_dict() for r in rows]
    wide_path = out_dir / "clarification_wide.csv"
    if wide_rows:
        with wide_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(wide_rows[0].keys()))
            writer.writeheader()
            writer.writerows(wide_rows)

    # LaTeX
    latex_str = _build_latex(rows)
    (out_dir / "clarification_table.tex").write_text(latex_str)

    # JSON
    (out_dir / "clarification_table.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2)
    )

    return rows
