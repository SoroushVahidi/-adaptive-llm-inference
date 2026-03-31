"""Confidence-threshold routing baseline for manuscript comparison.

Uses ``unified_confidence_score`` (already present in all four enriched routing
datasets) as a scalar routing signal: if a query's confidence is *below* a
chosen threshold, the policy routes to ``direct_plus_revise``; otherwise it
stays with ``reasoning_greedy``.

This is a simple, fully offline, parameter-free-except-threshold baseline that
is directly comparable to the main adaptive policy results (v6/v7), because it
operates on the same binary action space and the same four manuscript regimes.

Public API
----------
- ``REGIME_FILES`` — mapping from regime id to enriched CSV path.
- ``evaluate_threshold(df, threshold)`` → ``ThresholdResult``
- ``sweep_thresholds(df, thresholds)`` → list[ThresholdResult]
- ``choose_operating_point(results, target_cost)`` → ThresholdResult
- ``evaluate_all_regimes(threshold, regime_files)`` → list[RegimeResult]
- ``sweep_and_summarise(regime_files, thresholds, output_dir)``
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

_LOG = logging.getLogger(__name__)

import pandas as pd

# ---------------------------------------------------------------------------
# Regime registry
# ---------------------------------------------------------------------------

#: Maps manuscript regime id → enriched routing dataset CSV path.
REGIME_FILES: dict[str, str] = {
    "gsm8k_random_100": "data/real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": "data/real_math500_routing_dataset_enriched.csv",
    "gpqa_diamond_198": "data/real_gpqa_diamond_routing_dataset_enriched.csv",
}

#: Column used as the routing confidence signal.
CONFIDENCE_COL = "unified_confidence_score"

#: Default threshold sweep points.
DEFAULT_THRESHOLDS = [round(t * 0.05, 2) for t in range(0, 21)]  # 0.00 … 1.00

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ThresholdResult:
    """Per-threshold evaluation result for a single regime."""

    threshold: float
    accuracy: float
    avg_cost: float
    revise_rate: float
    n: int


@dataclass
class RegimeResult:
    """Best-operating-point result and full sweep for one manuscript regime."""

    regime: str
    operating_threshold: float
    accuracy: float
    avg_cost: float
    revise_rate: float
    n: int
    sweep: list[ThresholdResult]

    def to_summary_dict(self) -> dict:
        return {
            "regime": self.regime,
            "baseline": "confidence_threshold",
            "threshold": self.operating_threshold,
            "accuracy": self.accuracy,
            "avg_cost": self.avg_cost,
            "revise_rate": self.revise_rate,
            "n": self.n,
        }


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def evaluate_threshold(df: pd.DataFrame, threshold: float) -> ThresholdResult:
    """Evaluate one threshold value on a routing dataset DataFrame.

    Routing rule:
        ``confidence < threshold``  →  ``direct_plus_revise``  (revise=1)
        ``confidence >= threshold`` →  ``reasoning_greedy``    (revise=0)

    Accuracy is computed per query:
        - routed to revise  → correct iff ``revise_correct == 1``
        - routed to greedy  → correct iff ``reasoning_correct == 1``

    Cost model (matching the rest of the manuscript):
        - ``reasoning_greedy`` costs 1.0
        - ``direct_plus_revise`` costs 2.0
        → ``avg_cost = 1.0 + revise_rate``
    """
    n = len(df)
    revise_mask = df[CONFIDENCE_COL] < threshold

    correct = revise_mask * df["revise_correct"] + (~revise_mask) * df["reasoning_correct"]
    accuracy = float(correct.mean())
    revise_rate = float(revise_mask.mean())
    avg_cost = 1.0 + revise_rate

    return ThresholdResult(
        threshold=threshold,
        accuracy=accuracy,
        avg_cost=avg_cost,
        revise_rate=revise_rate,
        n=n,
    )


def sweep_thresholds(
    df: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
) -> list[ThresholdResult]:
    """Evaluate all threshold values; return results sorted by threshold."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    return [evaluate_threshold(df, t) for t in sorted(set(thresholds))]


def choose_operating_point(
    results: list[ThresholdResult],
    target_cost: float = 1.2,
) -> ThresholdResult:
    """Choose the operating point with the highest accuracy at or below *target_cost*.

    Falls back to the result with avg_cost closest to *target_cost* when no
    result meets the constraint exactly.
    """
    feasible = [r for r in results if r.avg_cost <= target_cost + 1e-9]
    if feasible:
        return max(feasible, key=lambda r: (r.accuracy, -r.avg_cost))
    # Fallback: closest to target cost
    return min(results, key=lambda r: abs(r.avg_cost - target_cost))


# ---------------------------------------------------------------------------
# Multi-regime helpers
# ---------------------------------------------------------------------------


def load_regime_df(regime: str, regime_files: dict[str, str] | None = None) -> pd.DataFrame:
    """Load the enriched CSV for *regime* and validate required columns."""
    files = regime_files if regime_files is not None else REGIME_FILES
    path = Path(files[regime])
    df = pd.read_csv(path)
    required = {CONFIDENCE_COL, "reasoning_correct", "revise_correct", "revise_helpful"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Regime '{regime}' CSV missing columns: {missing}")
    return df


def evaluate_all_regimes(
    threshold: float,
    regime_files: dict[str, str] | None = None,
) -> list[ThresholdResult]:
    """Evaluate a fixed threshold across all four manuscript regimes."""
    files = regime_files if regime_files is not None else REGIME_FILES
    results = []
    for regime in files:
        df = load_regime_df(regime, files)
        result = evaluate_threshold(df, threshold)
        results.append(result)
    return results


def sweep_and_summarise(
    regime_files: dict[str, str] | None = None,
    thresholds: Iterable[float] | None = None,
    output_dir: str | Path = "outputs/baselines/confidence_threshold",
    target_cost: float = 1.2,
) -> list[RegimeResult]:
    """Sweep thresholds for all regimes, pick operating points, write outputs.

    Writes
    ------
    ``<output_dir>/confidence_threshold_sweep.csv``
        Full threshold sweep (regime, threshold, accuracy, avg_cost, revise_rate).
    ``<output_dir>/confidence_threshold_summary.csv``
        One row per regime at the chosen operating point — matches the format of
        ``outputs/paper_tables_cleaned/main_results_summary.csv`` so it can be
        slotted directly into manuscript comparison tables.
    ``<output_dir>/confidence_threshold_summary.json``
        Same data as the CSV, serialised as a JSON list.
    """
    files = regime_files if regime_files is not None else REGIME_FILES
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_regime_results: list[RegimeResult] = []
    sweep_rows: list[dict] = []

    for regime in files:
        csv_path = Path(files[regime])
        if not csv_path.is_file():
            _LOG.warning(
                "Skipping regime %s: enriched routing CSV not found (%s)",
                regime,
                csv_path,
            )
            continue
        df = load_regime_df(regime, files)
        sweep = sweep_thresholds(df, thresholds)
        op = choose_operating_point(sweep, target_cost=target_cost)

        rr = RegimeResult(
            regime=regime,
            operating_threshold=op.threshold,
            accuracy=op.accuracy,
            avg_cost=op.avg_cost,
            revise_rate=op.revise_rate,
            n=op.n,
            sweep=sweep,
        )
        all_regime_results.append(rr)

        for tr in sweep:
            sweep_rows.append(
                {
                    "regime": regime,
                    "threshold": tr.threshold,
                    "accuracy": tr.accuracy,
                    "avg_cost": tr.avg_cost,
                    "revise_rate": tr.revise_rate,
                    "n": tr.n,
                }
            )

    # Write sweep CSV
    sweep_path = out_dir / "confidence_threshold_sweep.csv"
    _write_csv(sweep_path, sweep_rows)

    # Write summary CSV
    summary_rows = [r.to_summary_dict() for r in all_regime_results]
    summary_path = out_dir / "confidence_threshold_summary.csv"
    _write_csv(summary_path, summary_rows)

    # Write summary JSON
    json_path = out_dir / "confidence_threshold_summary.json"
    json_path.write_text(
        json.dumps([asdict(r) | {"sweep": None} for r in all_regime_results], indent=2)
    )

    return all_regime_results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
