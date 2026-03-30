"""AIME-2024 policy evaluation for the small experiment pass.

Reads the committed ``data/real_aime2024_routing_dataset.csv`` (30 queries,
all routing features pre-computed) and runs the same policy-comparison
pipeline used for GSM8K / MATH500, extended with:

- **oracle** routing (revise only when ``revise_helpful == 1``)
- **confidence-threshold** routing baseline (sweep over ``unified_confidence_score``)

No API calls are required — the dataset is fully committed.

Public API
----------
- ``run_small_pass_aime_eval(dataset_csv, output_dir, conf_target_cost)``
  → writes results to *output_dir* and returns a summary dict.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.baselines.confidence_threshold_router import (
    choose_operating_point,
    evaluate_threshold,
    sweep_thresholds,
)
from src.policies.adaptive_policy_v5 import (
    AdaptivePolicyV5Config,
    extract_question_features_v5,
)
from src.policies.adaptive_policy_v5 import choose_strategy as _c5
from src.policies.adaptive_policy_v6 import (
    AdaptivePolicyV6Config,
    extract_question_features_v6,
)
from src.policies.adaptive_policy_v6 import choose_strategy as _c6
from src.policies.adaptive_policy_v7 import AdaptivePolicyV7Config
from src.policies.adaptive_policy_v7 import choose_strategy as _c7

#: Default path to the committed AIME routing dataset.
DEFAULT_AIME_CSV = "data/real_aime2024_routing_dataset.csv"

#: Default output directory for small-pass AIME results.
DEFAULT_OUTPUT_DIR = "outputs/small_pass"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _int(x: Any) -> int:
    try:
        return int(float(str(x).strip()))
    except (ValueError, TypeError):
        return 0


def _float(x: Any) -> float:
    try:
        return float(str(x).strip())
    except (ValueError, TypeError):
        return 0.0


def _blocked(out: Path, reason: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_status": "BLOCKED",
        "evidence_status": "blocked",
        "reason": reason,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "aime_summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_small_pass_aime_eval(
    dataset_csv: str | Path = DEFAULT_AIME_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    conf_target_cost: float = 1.2,
) -> dict[str, Any]:
    """Evaluate routing policies on the AIME-2024 routing dataset.

    Parameters
    ----------
    dataset_csv:
        Path to ``real_aime2024_routing_dataset.csv`` (or a compatible file).
    output_dir:
        Directory for all output files.
    conf_target_cost:
        Target avg-cost budget used to choose the confidence-router operating
        point (same convention as the main confidence-threshold module).

    Returns
    -------
    dict with key ``"summary"`` containing the run summary.

    Output files written
    --------------------
    ``aime_summary.json``
        Top-level run summary (status, n, all comparison rows).
    ``aime_policy_comparison.csv``
        One row per routing strategy with accuracy / avg_cost / revise_rate.
    ``aime_per_query_decisions.csv``
        Per-query policy decisions and correctness outcomes.
    ``aime_confidence_sweep.csv``
        Full threshold sweep results for the confidence router.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = Path(dataset_csv)

    if not p.exists():
        return _blocked(out, f"dataset file not found: {p}")

    rows = _read_rows(p)
    if not rows:
        return _blocked(out, "dataset CSV is empty")

    # Check required columns
    required_cols = {
        "question",
        "reasoning_raw",
        "reasoning_correct",
        "revise_correct",
        "revise_helpful",
        "unified_confidence_score",
    }
    first_row_keys = set(rows[0].keys())
    missing = required_cols - first_row_keys
    if missing:
        return _blocked(out, f"missing required columns: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Per-query policy decisions
    # ------------------------------------------------------------------
    v5c = AdaptivePolicyV5Config()
    v6c = AdaptivePolicyV6Config()
    v7c = AdaptivePolicyV7Config()

    per_rows: list[dict[str, Any]] = []
    for r in rows:
        q = str(r.get("question", ""))
        first_pass = str(r.get("reasoning_raw", ""))
        if not q or not first_pass:
            continue

        f5 = extract_question_features_v5(q, v5c)
        f6 = extract_question_features_v6(q, v6c)
        f7 = extract_question_features_v6(q, v7c)

        ch5 = _c5(q, f5, first_pass, v5c)
        ch6 = _c6(q, f6, first_pass, v6c)
        ch7 = _c7(q, f7, first_pass, v7c)

        rc = _int(r.get("reasoning_correct"))
        vc = _int(r.get("revise_correct"))
        rh = _int(r.get("revise_helpful"))

        # Oracle: revise only when revision actually helps
        oracle_decision = "direct_plus_revise" if rh == 1 else "reasoning_greedy"

        per_rows.append(
            {
                "question_id": r.get("question_id", ""),
                "reasoning_correct": rc,
                "revise_correct": vc,
                "revise_helpful": rh,
                "unified_confidence_score": _float(r.get("unified_confidence_score", 0.0)),
                "policy_v5": ch5,
                "policy_v6": ch6,
                "policy_v7": ch7,
                "oracle_decision": oracle_decision,
                "correct_if_v5": vc if ch5 == "direct_plus_revise" else rc,
                "correct_if_v6": vc if ch6 == "direct_plus_revise" else rc,
                "correct_if_v7": vc if ch7 == "direct_plus_revise" else rc,
                "correct_if_oracle": vc if oracle_decision == "direct_plus_revise" else rc,
                "cost_v5": 2 if ch5 == "direct_plus_revise" else 1,
                "cost_v6": 2 if ch6 == "direct_plus_revise" else 1,
                "cost_v7": 2 if ch7 == "direct_plus_revise" else 1,
                "cost_oracle": 2 if oracle_decision == "direct_plus_revise" else 1,
            }
        )

    n = len(per_rows)
    if n == 0:
        return _blocked(out, "no scorable rows found in dataset")

    # ------------------------------------------------------------------
    # Confidence-threshold router sweep on the AIME data
    # ------------------------------------------------------------------
    import pandas as pd  # imported here to match existing code style

    aime_df = pd.DataFrame(
        [
            {
                "unified_confidence_score": r["unified_confidence_score"],
                "reasoning_correct": r["reasoning_correct"],
                "revise_correct": r["revise_correct"],
                "revise_helpful": r["revise_helpful"],
            }
            for r in per_rows
        ]
    )
    conf_sweep = sweep_thresholds(aime_df)
    conf_op = choose_operating_point(conf_sweep, target_cost=conf_target_cost)

    # ------------------------------------------------------------------
    # Build comparison table
    # ------------------------------------------------------------------
    comparison: list[dict[str, Any]] = [
        {
            "route": "reasoning_greedy",
            "accuracy": sum(r["reasoning_correct"] for r in per_rows) / n,
            "avg_cost": 1.0,
            "revise_rate": 0.0,
            "notes": "cheap baseline (no revision)",
        },
        {
            "route": "direct_plus_revise",
            "accuracy": sum(r["revise_correct"] for r in per_rows) / n,
            "avg_cost": 2.0,
            "revise_rate": 1.0,
            "notes": "always-revise baseline",
        },
        {
            "route": "adaptive_policy_v5",
            "accuracy": sum(r["correct_if_v5"] for r in per_rows) / n,
            "avg_cost": sum(r["cost_v5"] for r in per_rows) / n,
            "revise_rate": sum(1 for r in per_rows if r["policy_v5"] == "direct_plus_revise") / n,
            "notes": "calibrated role + unified error policy",
        },
        {
            "route": "adaptive_policy_v6",
            "accuracy": sum(r["correct_if_v6"] for r in per_rows) / n,
            "avg_cost": sum(r["cost_v6"] for r in per_rows) / n,
            "revise_rate": sum(1 for r in per_rows if r["policy_v6"] == "direct_plus_revise") / n,
            "notes": "v5 + answer confidence filtering",
        },
        {
            "route": "adaptive_policy_v7",
            "accuracy": sum(r["correct_if_v7"] for r in per_rows) / n,
            "avg_cost": sum(r["cost_v7"] for r in per_rows) / n,
            "revise_rate": sum(1 for r in per_rows if r["policy_v7"] == "direct_plus_revise") / n,
            "notes": "v6 + extended false-negative fixes (main policy)",
        },
        {
            "route": "confidence_threshold",
            "accuracy": conf_op.accuracy,
            "avg_cost": conf_op.avg_cost,
            "revise_rate": conf_op.revise_rate,
            "notes": (
                f"confidence router (threshold={conf_op.threshold:.2f}, "
                f"target_cost≤{conf_target_cost})"
            ),
        },
        {
            "route": "oracle",
            "accuracy": sum(r["correct_if_oracle"] for r in per_rows) / n,
            "avg_cost": sum(r["cost_oracle"] for r in per_rows) / n,
            "revise_rate": sum(
                1 for r in per_rows if r["oracle_decision"] == "direct_plus_revise"
            ) / n,
            "notes": "oracle upper bound (revise only when revise_helpful=1)",
        },
    ]

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    # Per-query decisions
    with (out / "aime_per_query_decisions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)

    # Policy comparison table
    with (out / "aime_policy_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["route", "accuracy", "avg_cost", "revise_rate", "notes"])
        w.writeheader()
        w.writerows(comparison)

    # Confidence sweep CSV
    sweep_rows = [
        {
            "threshold": tr.threshold,
            "accuracy": tr.accuracy,
            "avg_cost": tr.avg_cost,
            "revise_rate": tr.revise_rate,
            "n": tr.n,
        }
        for tr in conf_sweep
    ]
    with (out / "aime_confidence_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    v7_acc = next(r["accuracy"] for r in comparison if r["route"] == "adaptive_policy_v7")
    greedy_acc = next(r["accuracy"] for r in comparison if r["route"] == "reasoning_greedy")
    oracle_acc = next(r["accuracy"] for r in comparison if r["route"] == "oracle")

    summary: dict[str, Any] = {
        "run_status": "COMPLETED",
        "evidence_status": "measured_now",
        "dataset": "aime2024",
        "num_rows": n,
        "revise_helpful_prevalence": sum(r["revise_helpful"] for r in per_rows) / n,
        "reasoning_greedy_accuracy": greedy_acc,
        "v7_accuracy": v7_acc,
        "oracle_accuracy": oracle_acc,
        "confidence_router_accuracy": conf_op.accuracy,
        "confidence_router_threshold": conf_op.threshold,
        "comparison": comparison,
    }
    (out / "aime_summary.json").write_text(json.dumps(summary, indent=2))

    return {"summary": summary}
