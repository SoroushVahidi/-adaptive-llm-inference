"""Small-pass combined evaluation: runs AIME eval + confidence baseline sweep,
produces manuscript-oriented comparison tables in outputs/paper_tables_small_pass/.

This module orchestrates:
1. AIME-2024 policy evaluation (fully offline, uses committed data).
2. Confidence-threshold baseline sweep across all four main manuscript regimes.
3. Combined paper-tables export.

GPQA-Diamond is **not** part of this orchestrator; run the dedicated pipeline
(``scripts/run_build_real_routing_dataset.py --paired-outcomes --dataset gpqa_diamond``)
and see ``docs/GPQA_EVALUATION_STATUS.md``.

Public API
----------
- ``run_small_pass(output_dir, tables_dir, conf_target_cost)`` → summary dict
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.baselines.confidence_threshold_router import (
    REGIME_FILES as MAIN_REGIME_FILES,
    sweep_and_summarise,
)
from src.evaluation.small_pass_aime_eval import (
    DEFAULT_AIME_CSV,
    run_small_pass_aime_eval,
)

DEFAULT_OUTPUT_DIR = "outputs/small_pass"
DEFAULT_TABLES_DIR = "outputs/paper_tables_small_pass"


def _load_main_results() -> list[dict[str, Any]]:
    """Load the four main-paper regime results for side-by-side comparison."""
    p = Path("outputs/paper_tables_cleaned/main_results_summary.csv")
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_small_pass(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    tables_dir: str | Path = DEFAULT_TABLES_DIR,
    conf_target_cost: float = 1.2,
) -> dict[str, Any]:
    """Run the complete small experiment pass.

    Steps
    -----
    1. AIME-2024 policy evaluation (offline).
    2. Confidence-threshold baseline sweep on the four main manuscript regimes.
    3. Produce combined comparison table in *tables_dir*.

    Parameters
    ----------
    output_dir:
        Raw outputs directory (per-query CSVs, sweeps, etc.).
    tables_dir:
        Publication-oriented tables directory.
    conf_target_cost:
        Target avg-cost budget for confidence-router operating-point selection.

    Returns
    -------
    dict with keys ``"aime"``, ``"confidence_baseline"``, ``"run_status"``.
    """
    out = Path(output_dir)
    tables = Path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. AIME-2024 policy evaluation
    # ------------------------------------------------------------------
    aime_result = run_small_pass_aime_eval(
        dataset_csv=DEFAULT_AIME_CSV,
        output_dir=out,
        conf_target_cost=conf_target_cost,
    )

    # ------------------------------------------------------------------
    # 2. Confidence-threshold sweep on main manuscript regimes
    # ------------------------------------------------------------------
    # Build absolute paths for the main regime files
    repo_root = Path(__file__).resolve().parent.parent.parent
    regime_files_abs = {k: str(repo_root / v) for k, v in MAIN_REGIME_FILES.items()}

    conf_results = sweep_and_summarise(
        regime_files=regime_files_abs,
        output_dir=out / "confidence_threshold",
        target_cost=conf_target_cost,
    )

    # ------------------------------------------------------------------
    # 3. Build combined comparison table (confidence baseline vs main paper)
    # ------------------------------------------------------------------
    main_rows = _load_main_results()

    # Confidence results keyed by regime
    conf_by_regime = {r.regime: r for r in conf_results}

    combined_rows: list[dict[str, Any]] = []
    for mr in main_rows:
        regime = mr["dataset"]
        cr = conf_by_regime.get(regime)
        combined_rows.append(
            {
                "regime": regime,
                "n": mr.get("n", 100),
                "reasoning_greedy_acc": mr.get("reasoning_acc", ""),
                "always_revise_acc": mr.get("revise_acc", ""),
                "best_adaptive_policy_acc": mr.get("best_policy_acc", ""),
                "best_adaptive_policy_cost": mr.get("best_policy_cost", ""),
                "oracle_acc": mr.get("oracle_acc", ""),
                "confidence_router_acc": round(cr.accuracy, 4) if cr else "N/A",
                "confidence_router_cost": round(cr.avg_cost, 4) if cr else "N/A",
                "confidence_router_threshold": cr.operating_threshold if cr else "N/A",
                "confidence_router_revise_rate": round(cr.revise_rate, 4) if cr else "N/A",
            }
        )

    # Write combined table
    combined_csv = tables / "small_pass_combined_comparison.csv"
    if combined_rows:
        with combined_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(combined_rows[0].keys()))
            w.writeheader()
            w.writerows(combined_rows)

    # Write AIME standalone result table
    aime_summary = aime_result.get("summary", {})
    aime_comparison = aime_summary.get("comparison", [])
    if aime_comparison:
        aime_table_csv = tables / "aime_policy_comparison.csv"
        with aime_table_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(aime_comparison[0].keys()))
            w.writeheader()
            w.writerows(aime_comparison)

    # Write confidence baseline summary table
    conf_summary_rows = [r.to_summary_dict() for r in conf_results]
    conf_table_csv = tables / "confidence_baseline_main_regimes.csv"
    if conf_summary_rows:
        with conf_table_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(conf_summary_rows[0].keys()))
            w.writeheader()
            w.writerows(conf_summary_rows)

    # ------------------------------------------------------------------
    # 4. Top-level summary JSON
    # ------------------------------------------------------------------
    run_summary: dict[str, Any] = {
        "run_status": "COMPLETED",
        "aime_status": aime_summary.get("run_status", "UNKNOWN"),
        "gpqa_status": "NOT_RUN_IN_SMALL_PASS",
        "gpqa_note": (
            "GPQA-Diamond manuscript eval is a separate entry point; see "
            "docs/GPQA_EVALUATION_STATUS.md (build + policy eval scripts)."
        ),
        "confidence_baseline_regimes": list(conf_by_regime.keys()),
        "outputs": {
            "aime_summary": str(out / "aime_summary.json"),
            "aime_policy_comparison": str(out / "aime_policy_comparison.csv"),
            "confidence_threshold_sweep": str(out / "confidence_threshold" / "confidence_threshold_sweep.csv"),
            "combined_comparison_table": str(combined_csv),
            "aime_table": str(tables / "aime_policy_comparison.csv"),
            "confidence_table": str(conf_table_csv),
        },
    }

    (out / "small_pass_run_summary.json").write_text(json.dumps(run_summary, indent=2))
    (tables / "small_pass_run_summary.json").write_text(json.dumps(run_summary, indent=2))

    return {
        "aime": aime_result,
        "confidence_baseline": conf_summary_rows,
        "run_status": run_summary,
    }
