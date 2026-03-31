#!/usr/bin/env python3
"""Generate all lightweight paper-strengthening artifacts from existing stored outputs.

This script detects available artifacts, computes new tables and figures, and
skips gracefully when inputs are missing.  No new LLM inference is performed.
All computations are deterministic (fixed random seed where applicable).

Usage:
    python scripts/generate_lightweight_paper_artifacts.py

Outputs are written under:
    outputs/paper_tables/   (CSV tables)
    outputs/paper_figures/  (PNG figures)
    outputs/paper_tables/LIGHTWEIGHT_ARTIFACT_GENERATION_REPORT.md
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap configuration
# ---------------------------------------------------------------------------
_BOOTSTRAP_SEED = 42
_BOOTSTRAP_N_RESAMPLES = 10_000

# Cost model: cheap action = 1.0, expensive action = 2.0 (reasoning-greedy vs
# direct-plus-revise in this repository).
_COST_CHEAP = 1.0
_COST_EXPENSIVE = 2.0

# Alternative cost ratio denominators to test sensitivity (expensive / cheap).
_ALT_COST_RATIOS: list[tuple[str, float]] = [
    ("1:1.5", 1.5),
    ("1:2",   2.0),   # baseline
    ("1:3",   3.0),
]

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
_OUTPUTS = _REPO / "outputs"
_TABLES_OUT = _OUTPUTS / "paper_tables"
_FIGURES_OUT = _OUTPUTS / "paper_figures"

# Per-query decision files keyed by regime label
_PER_QUERY_FILES: dict[str, Path] = {
    "math500_100":      _OUTPUTS / "real_math500_policy_eval"      / "per_query_policy_decisions.csv",
    "hard_gsm8k_100":   _OUTPUTS / "real_hard_gsm8k_policy_eval"   / "per_query_policy_decisions.csv",
    "hard_gsm8k_b2":    _OUTPUTS / "real_hard_gsm8k_b2_policy_eval"/ "per_query_policy_decisions.csv",
    "gsm8k_random_100": _OUTPUTS / "real_policy_eval"              / "per_query_policy_decisions.csv",
}

# Oracle summary files keyed by regime label
_ORACLE_FILES: dict[str, Path] = {
    "math500_100":      _OUTPUTS / "oracle_routing_eval" / "math500_100_oracle_summary.json",
    "hard_gsm8k_100":   _OUTPUTS / "oracle_routing_eval" / "hard_gsm8k_100_oracle_summary.json",
    "hard_gsm8k_b2":    _OUTPUTS / "oracle_routing_eval" / "hard_gsm8k_b2_oracle_summary.json",
    "gsm8k_random_100": _OUTPUTS / "oracle_routing_eval" / "gsm8k_random100_oracle_summary.json",
}

# Policy comparison summary CSVs keyed by regime label
_POLICY_COMP_FILES: dict[str, Path] = {
    "math500_100":      _OUTPUTS / "real_math500_policy_eval"      / "policy_comparison.csv",
    "hard_gsm8k_100":   _OUTPUTS / "real_hard_gsm8k_policy_eval"   / "policy_comparison.csv",
    "hard_gsm8k_b2":    _OUTPUTS / "real_hard_gsm8k_b2_policy_eval"/ "policy_comparison.csv",
    "gsm8k_random_100": _OUTPUTS / "real_policy_eval"              / "policy_comparison.csv",
}

# Confidence threshold sweep CSV
_THRESHOLD_SWEEP_FILE = _OUTPUTS / "baselines" / "confidence_threshold" / "confidence_threshold_sweep.csv"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dicts (all values as strings)."""
    import csv
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_div(numerator: float, denominator: float, fallback: str = "N/A") -> Any:
    if denominator == 0 or denominator != denominator:  # 0 or NaN
        return fallback
    return numerator / denominator


def _fmt(v: Any, decimals: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


# ---------------------------------------------------------------------------
# Generation tracking
# ---------------------------------------------------------------------------
_generated: list[str] = []
_skipped: list[tuple[str, str]] = []


def _record_generated(path: Path) -> None:
    _generated.append(str(path.relative_to(_REPO)))


def _record_skipped(name: str, reason: str) -> None:
    _skipped.append((name, reason))


# ---------------------------------------------------------------------------
# 1.  paired_policy_comparison.csv
# ---------------------------------------------------------------------------

def _generate_paired_policy_comparison() -> None:
    """For each regime compare policies against Always-RG and Always-DPR."""
    rows: list[dict[str, Any]] = []

    for regime, comp_path in _POLICY_COMP_FILES.items():
        if not comp_path.exists():
            _record_skipped("paired_policy_comparison", f"missing {comp_path}")
            continue

        records = _read_csv(comp_path)

        # Index by route name
        by_route: dict[str, dict[str, str]] = {r["route"]: r for r in records}

        rg = by_route.get("reasoning_greedy")
        dpr = by_route.get("direct_plus_revise")
        if rg is None or dpr is None:
            _record_skipped("paired_policy_comparison", f"missing baseline routes in {regime}")
            continue

        rg_acc  = float(rg["accuracy"])
        rg_cost = float(rg["avg_cost"])
        dpr_acc  = float(dpr["accuracy"])
        dpr_cost = float(dpr["avg_cost"])

        # Oracle
        oracle_acc: float | None = None
        oracle_path = _ORACLE_FILES.get(regime)
        if oracle_path and oracle_path.exists():
            with oracle_path.open() as fh:
                oracle_data = json.load(fh)
            oracle_acc = oracle_data.get("accuracy")

        for rec in records:
            pol  = rec["route"]
            acc  = float(rec["accuracy"])
            cost = float(rec["avg_cost"])
            rev  = float(rec["revise_rate"])

            gain_vs_rg   = acc - rg_acc
            cost_delta   = cost - rg_cost
            cost_savings = dpr_cost - cost   # positive = cheaper than DPR

            oracle_gap = (oracle_acc - acc) if oracle_acc is not None else "N/A"

            rows.append({
                "regime":                    regime,
                "policy":                    pol,
                "accuracy":                  _fmt(acc),
                "average_cost":              _fmt(cost),
                "revise_rate":               _fmt(rev),
                "gain_vs_always_rg":         _fmt(gain_vs_rg),
                "cost_delta_vs_always_rg":   _fmt(cost_delta),
                "cost_savings_vs_always_dpr":_fmt(cost_savings),
                "oracle_gap":                _fmt(oracle_gap) if oracle_gap != "N/A" else "N/A",
            })

    if not rows:
        _record_skipped("paired_policy_comparison", "no data available")
        return

    out = _TABLES_OUT / "paired_policy_comparison.csv"
    _write_csv(out, rows, [
        "regime", "policy", "accuracy", "average_cost", "revise_rate",
        "gain_vs_always_rg", "cost_delta_vs_always_rg",
        "cost_savings_vs_always_dpr", "oracle_gap",
    ])
    _record_generated(out)


# ---------------------------------------------------------------------------
# 2.  routing_outcome_breakdown.csv
# ---------------------------------------------------------------------------

def _generate_routing_outcome_breakdown() -> None:
    """Count per-query routing outcomes (both correct, RG only, DPR only, both wrong)."""
    rows: list[dict[str, Any]] = []

    for regime, pq_path in _PER_QUERY_FILES.items():
        if not pq_path.exists():
            _record_skipped("routing_outcome_breakdown", f"missing {pq_path}")
            continue

        records = _read_csv(pq_path)
        n = len(records)
        if n == 0:
            continue

        both_correct = sum(
            1 for r in records
            if r.get("reasoning_correct", "0") == "1" and r.get("revise_correct", "0") == "1"
        )
        rg_only = sum(
            1 for r in records
            if r.get("reasoning_correct", "0") == "1" and r.get("revise_correct", "0") != "1"
        )
        dpr_only = sum(
            1 for r in records
            if r.get("reasoning_correct", "0") != "1" and r.get("revise_correct", "0") == "1"
        )
        both_wrong = sum(
            1 for r in records
            if r.get("reasoning_correct", "0") != "1" and r.get("revise_correct", "0") != "1"
        )

        rows.append({
            "regime":                 regime,
            "n":                      n,
            "both_correct":           both_correct,
            "rg_only_correct":        rg_only,
            "dpr_only_correct":       dpr_only,
            "both_wrong":             both_wrong,
            "frac_both_correct":      _fmt(both_correct / n),
            "frac_rg_only":           _fmt(rg_only / n),
            "frac_dpr_only":          _fmt(dpr_only / n),
            "frac_both_wrong":        _fmt(both_wrong / n),
            "routing_headroom":       _fmt(dpr_only / n),  # fraction where routing can help
        })

    if not rows:
        _record_skipped("routing_outcome_breakdown", "no per-query data available")
        return

    out = _TABLES_OUT / "routing_outcome_breakdown.csv"
    _write_csv(out, rows, [
        "regime", "n",
        "both_correct", "rg_only_correct", "dpr_only_correct", "both_wrong",
        "frac_both_correct", "frac_rg_only", "frac_dpr_only", "frac_both_wrong",
        "routing_headroom",
    ])
    _record_generated(out)


# ---------------------------------------------------------------------------
# 3.  policy_efficiency_table.csv
# ---------------------------------------------------------------------------

def _generate_policy_efficiency_table() -> None:
    """Accuracy gain per cost unit and DPR-recovery fractions for adaptive policies."""
    rows: list[dict[str, Any]] = []

    for regime, comp_path in _POLICY_COMP_FILES.items():
        if not comp_path.exists():
            continue

        records = _read_csv(comp_path)
        by_route: dict[str, dict[str, str]] = {r["route"]: r for r in records}

        rg = by_route.get("reasoning_greedy")
        dpr = by_route.get("direct_plus_revise")
        if rg is None or dpr is None:
            continue

        rg_acc   = float(rg["accuracy"])
        rg_cost  = float(rg["avg_cost"])
        dpr_acc  = float(dpr["accuracy"])
        dpr_cost = float(dpr["avg_cost"])

        # Denominators for normalisation
        acc_range  = dpr_acc - rg_acc   # total accuracy gain available
        cost_range = dpr_cost - rg_cost  # total extra cost

        for rec in records:
            pol  = rec["route"]
            if pol in ("reasoning_greedy", "direct_plus_revise"):
                continue  # skip baselines in this table

            acc  = float(rec["accuracy"])
            cost = float(rec["avg_cost"])

            acc_gain    = acc - rg_acc
            extra_cost  = cost - rg_cost
            gain_per_cost = _safe_div(acc_gain, extra_cost)
            frac_dpr_acc  = _safe_div(acc_gain,  acc_range)
            frac_dpr_cost_avoided = _safe_div(dpr_cost - cost, cost_range)

            rows.append({
                "regime":                         regime,
                "policy":                         pol,
                "accuracy":                       _fmt(acc),
                "average_cost":                   _fmt(cost),
                "accuracy_gain_vs_rg":            _fmt(acc_gain),
                "extra_cost_vs_rg":               _fmt(extra_cost),
                "gain_per_extra_cost_unit":       _fmt(gain_per_cost) if gain_per_cost != "N/A" else "N/A",
                "frac_dpr_accuracy_recovered":    _fmt(frac_dpr_acc) if frac_dpr_acc != "N/A" else "N/A",
                "frac_dpr_cost_avoided":          _fmt(frac_dpr_cost_avoided) if frac_dpr_cost_avoided != "N/A" else "N/A",
            })

    if not rows:
        _record_skipped("policy_efficiency_table", "no data available")
        return

    out = _TABLES_OUT / "policy_efficiency_table.csv"
    _write_csv(out, rows, [
        "regime", "policy", "accuracy", "average_cost",
        "accuracy_gain_vs_rg", "extra_cost_vs_rg",
        "gain_per_extra_cost_unit",
        "frac_dpr_accuracy_recovered",
        "frac_dpr_cost_avoided",
    ])
    _record_generated(out)


# ---------------------------------------------------------------------------
# 4.  oracle_headroom_table.csv
# ---------------------------------------------------------------------------

def _generate_oracle_headroom_table() -> None:
    """Compare RG, DPR, oracle, and best adaptive accuracy per regime."""
    rows: list[dict[str, Any]] = []

    for regime in _POLICY_COMP_FILES:
        comp_path = _POLICY_COMP_FILES[regime]
        oracle_path = _ORACLE_FILES.get(regime)

        if not comp_path.exists():
            _record_skipped("oracle_headroom_table", f"missing policy comp for {regime}")
            continue

        records = _read_csv(comp_path)
        by_route: dict[str, dict[str, str]] = {r["route"]: r for r in records}

        rg  = by_route.get("reasoning_greedy")
        dpr = by_route.get("direct_plus_revise")
        if rg is None or dpr is None:
            continue

        rg_acc  = float(rg["accuracy"])
        dpr_acc = float(dpr["accuracy"])

        oracle_acc: float | None = None
        revise_helpful_rate: str = "N/A"
        if oracle_path and oracle_path.exists():
            with oracle_path.open() as fh:
                od = json.load(fh)
            oracle_acc = od.get("accuracy")
            revise_helpful_rate = _fmt(od.get("revise_rate", "N/A"))

        # Best adaptive policy = max accuracy among non-baseline policies
        adaptive_records = [r for r in records if r["route"] not in ("reasoning_greedy", "direct_plus_revise")]
        best_adaptive_acc: float | None = None
        best_adaptive_policy = "N/A"
        if adaptive_records:
            best_rec = max(adaptive_records, key=lambda r: float(r["accuracy"]))
            best_adaptive_acc = float(best_rec["accuracy"])
            best_adaptive_policy = best_rec["route"]

        adaptive_to_oracle_gap = (
            _fmt(oracle_acc - best_adaptive_acc)
            if (oracle_acc is not None and best_adaptive_acc is not None)
            else "N/A"
        )
        rg_to_oracle_gap  = _fmt(oracle_acc - rg_acc)  if oracle_acc is not None else "N/A"
        dpr_to_oracle_gap = _fmt(oracle_acc - dpr_acc) if oracle_acc is not None else "N/A"

        rows.append({
            "regime":                  regime,
            "always_rg_accuracy":      _fmt(rg_acc),
            "always_dpr_accuracy":     _fmt(dpr_acc),
            "oracle_accuracy":         _fmt(oracle_acc) if oracle_acc is not None else "N/A",
            "revise_helpful_rate":     revise_helpful_rate,
            "best_adaptive_policy":    best_adaptive_policy,
            "best_adaptive_accuracy":  _fmt(best_adaptive_acc) if best_adaptive_acc is not None else "N/A",
            "adaptive_to_oracle_gap":  adaptive_to_oracle_gap,
            "rg_to_oracle_gap":        rg_to_oracle_gap,
            "dpr_to_oracle_gap":       dpr_to_oracle_gap,
        })

    if not rows:
        _record_skipped("oracle_headroom_table", "no data available")
        return

    out = _TABLES_OUT / "oracle_headroom_table.csv"
    _write_csv(out, rows, [
        "regime",
        "always_rg_accuracy", "always_dpr_accuracy", "oracle_accuracy",
        "revise_helpful_rate",
        "best_adaptive_policy", "best_adaptive_accuracy",
        "adaptive_to_oracle_gap", "rg_to_oracle_gap", "dpr_to_oracle_gap",
    ])
    _record_generated(out)


# ---------------------------------------------------------------------------
# 5.  Bootstrap CIs  &  paired difference tests
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    samples: list[int],
    n_resamples: int = _BOOTSTRAP_N_RESAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) from binary correctness array via bootstrap."""
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    resampled_means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_resamples)
    ])
    mean = arr.mean()
    ci_lo, ci_hi = float(np.percentile(resampled_means, 2.5)), float(np.percentile(resampled_means, 97.5))
    return float(mean), ci_lo, ci_hi


def _paired_diff_bootstrap(
    a: list[int], b: list[int],
    n_resamples: int = _BOOTSTRAP_N_RESAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float, float, float]:
    """Return (mean_diff, ci_lower, ci_upper, p_value) for a-b paired difference.

    p_value is the fraction of bootstrap samples with diff <= 0 (or >= 0 for
    the other tail), giving an approximate one-sided p-value under the null
    that E[a-b] <= 0.  Two-sided p = 2*min(p_one_sided, 1-p_one_sided).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    boot_means = np.array([
        rng.choice(diff, size=n, replace=True).mean()
        for _ in range(n_resamples)
    ])
    mean_d = float(diff.mean())
    ci_lo  = float(np.percentile(boot_means, 2.5))
    ci_hi  = float(np.percentile(boot_means, 97.5))
    # One-sided p: fraction of boot means <= 0 (i.e. no positive difference)
    p_one = float((boot_means <= 0).mean())
    p_two = float(2 * min(p_one, 1 - p_one))
    return mean_d, ci_lo, ci_hi, p_two


def _generate_bootstrap_artifacts() -> None:
    import numpy as np  # noqa: F401

    ci_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []

    for regime, pq_path in _PER_QUERY_FILES.items():
        if not pq_path.exists():
            _record_skipped("bootstrap_accuracy_ci", f"missing {pq_path}")
            continue

        records = _read_csv(pq_path)
        if not records:
            continue

        # Binary correctness vectors for each policy
        rg_correct  = [int(r.get("reasoning_correct", 0)) for r in records]
        dpr_correct = [int(r.get("revise_correct", 0))    for r in records]

        # Determine available policy columns
        policy_cols = [k for k in records[0].keys() if k.startswith("correct_if_")]

        # Compute CI for RG and DPR
        for label, corr_vec in [("reasoning_greedy", rg_correct), ("direct_plus_revise", dpr_correct)]:
            m, lo, hi = _bootstrap_ci(corr_vec)
            ci_rows.append({
                "regime": regime, "policy": label,
                "accuracy": _fmt(m), "ci_lower_95": _fmt(lo), "ci_upper_95": _fmt(hi),
                "n": len(corr_vec),
            })

        # Compute CI for each adaptive policy
        policy_correct: dict[str, list[int]] = {}
        for col in policy_cols:
            pol_name = col.replace("correct_if_", "")
            corr_vec = [int(r.get(col, 0)) for r in records]
            m, lo, hi = _bootstrap_ci(corr_vec)
            ci_rows.append({
                "regime": regime, "policy": pol_name,
                "accuracy": _fmt(m), "ci_lower_95": _fmt(lo), "ci_upper_95": _fmt(hi),
                "n": len(corr_vec),
            })
            policy_correct[pol_name] = corr_vec

        # Paired differences: best adaptive minus RG
        if policy_correct:
            # Best adaptive by mean accuracy
            best_pol = max(policy_correct, key=lambda k: sum(policy_correct[k]) / len(policy_correct[k]))
            best_vec = policy_correct[best_pol]

            md, lo, hi, p = _paired_diff_bootstrap(best_vec, rg_correct)
            diff_rows.append({
                "regime": regime, "comparison": f"{best_pol} minus reasoning_greedy",
                "mean_diff": _fmt(md), "ci_lower_95": _fmt(lo), "ci_upper_95": _fmt(hi),
                "p_value_two_sided": _fmt(p), "n": len(rg_correct),
                "method": "paired-bootstrap",
            })

            # DPR minus best adaptive
            md2, lo2, hi2, p2 = _paired_diff_bootstrap(dpr_correct, best_vec)
            diff_rows.append({
                "regime": regime, "comparison": f"direct_plus_revise minus {best_pol}",
                "mean_diff": _fmt(md2), "ci_lower_95": _fmt(lo2), "ci_upper_95": _fmt(hi2),
                "p_value_two_sided": _fmt(p2), "n": len(rg_correct),
                "method": "paired-bootstrap",
            })

        # Oracle gap if available
        oracle_path = _ORACLE_FILES.get(regime)
        if oracle_path and oracle_path.exists() and policy_correct:
            with oracle_path.open() as fh:
                od = json.load(fh)
            oracle_acc = od.get("accuracy")
            if oracle_acc is not None:
                # Compute oracle binary correctness from revise_helpful and rg_correct
                # oracle chooses the best action per query
                oracle_vec = [
                    max(int(r.get("reasoning_correct", 0)), int(r.get("revise_correct", 0)))
                    for r in records
                ]
                best_pol = max(policy_correct, key=lambda k: sum(policy_correct[k]) / len(policy_correct[k]))
                best_vec = policy_correct[best_pol]
                md3, lo3, hi3, p3 = _paired_diff_bootstrap(oracle_vec, best_vec)
                diff_rows.append({
                    "regime": regime, "comparison": f"oracle minus {best_pol}",
                    "mean_diff": _fmt(md3), "ci_lower_95": _fmt(lo3), "ci_upper_95": _fmt(hi3),
                    "p_value_two_sided": _fmt(p3), "n": len(rg_correct),
                    "method": "paired-bootstrap",
                })

                # CI for oracle
                mo, loo, hio = _bootstrap_ci(oracle_vec)
                ci_rows.append({
                    "regime": regime, "policy": "oracle",
                    "accuracy": _fmt(mo), "ci_lower_95": _fmt(loo), "ci_upper_95": _fmt(hio),
                    "n": len(oracle_vec),
                })

    if ci_rows:
        out_ci = _TABLES_OUT / "bootstrap_accuracy_ci.csv"
        _write_csv(out_ci, ci_rows, [
            "regime", "policy", "accuracy", "ci_lower_95", "ci_upper_95", "n",
        ])
        _record_generated(out_ci)
    else:
        _record_skipped("bootstrap_accuracy_ci", "no per-query data available")

    if diff_rows:
        out_diff = _TABLES_OUT / "paired_difference_tests.csv"
        _write_csv(out_diff, diff_rows, [
            "regime", "comparison", "mean_diff",
            "ci_lower_95", "ci_upper_95", "p_value_two_sided", "n", "method",
        ])
        _record_generated(out_diff)
    else:
        _record_skipped("paired_difference_tests", "no per-query data available")


# ---------------------------------------------------------------------------
# 6.  Threshold sweep summary
# ---------------------------------------------------------------------------

def _generate_threshold_sweep_summary() -> None:
    """Reformat the confidence threshold sweep into a clean paper-ready table."""
    if not _THRESHOLD_SWEEP_FILE.exists():
        _record_skipped("threshold_sweep_summary", f"missing {_THRESHOLD_SWEEP_FILE}")
        return

    rows = _read_csv(_THRESHOLD_SWEEP_FILE)
    out = _TABLES_OUT / "threshold_sweep_summary.csv"
    _write_csv(out, rows, ["regime", "threshold", "accuracy", "avg_cost", "revise_rate", "n"])
    _record_generated(out)


# ---------------------------------------------------------------------------
# 7.  Cost ratio sensitivity
# ---------------------------------------------------------------------------

def _generate_cost_ratio_sensitivity() -> None:
    """Recompute avg_cost and compare policies under alternative cheap:expensive ratios.

    We use the stored revise_rate (= fraction of queries routed to DPR) and
    recompute avg_cost = (1 - revise_rate)*cheap + revise_rate*expensive.
    """
    rows: list[dict[str, Any]] = []

    for regime, comp_path in _POLICY_COMP_FILES.items():
        if not comp_path.exists():
            continue

        records = _read_csv(comp_path)

        for label, expensive in _ALT_COST_RATIOS:
            cheap = 1.0
            for rec in records:
                pol  = rec["route"]
                acc  = float(rec["accuracy"])
                rev  = float(rec["revise_rate"])
                recomputed_cost = (1 - rev) * cheap + rev * expensive
                rows.append({
                    "regime":            regime,
                    "policy":            pol,
                    "cost_ratio":        label,
                    "cheap_cost":        _fmt(cheap),
                    "expensive_cost":    _fmt(expensive),
                    "revise_rate":       _fmt(rev),
                    "accuracy":          _fmt(acc),
                    "recomputed_avg_cost": _fmt(recomputed_cost),
                })

    if not rows:
        _record_skipped("cost_ratio_sensitivity", "no policy comparison data available")
        return

    out = _TABLES_OUT / "cost_ratio_sensitivity.csv"
    _write_csv(out, rows, [
        "regime", "policy", "cost_ratio",
        "cheap_cost", "expensive_cost", "revise_rate",
        "accuracy", "recomputed_avg_cost",
    ])
    _record_generated(out)


# ---------------------------------------------------------------------------
# 8.  Policy ranking stability
# ---------------------------------------------------------------------------

def _generate_policy_ranking_stability() -> None:
    """Show best adaptive policy per regime under each cost ratio."""
    rows: list[dict[str, Any]] = []

    for regime, comp_path in _POLICY_COMP_FILES.items():
        if not comp_path.exists():
            continue

        records = _read_csv(comp_path)
        adaptive = [r for r in records if r["route"] not in ("reasoning_greedy", "direct_plus_revise")]
        if not adaptive:
            continue

        rg  = next((r for r in records if r["route"] == "reasoning_greedy"), None)
        dpr = next((r for r in records if r["route"] == "direct_plus_revise"), None)
        rg_acc = float(rg["accuracy"]) if rg else 0.0

        for label, expensive in _ALT_COST_RATIOS:
            cheap = 1.0
            # Rank by accuracy first, then by cost (ascending) as tie-breaker
            def _score(rec: dict[str, str]) -> tuple[float, float]:
                rev = float(rec["revise_rate"])
                cost = (1 - rev) * cheap + rev * expensive
                return (float(rec["accuracy"]), -cost)

            best = max(adaptive, key=_score)
            rev  = float(best["revise_rate"])
            cost = (1 - rev) * cheap + rev * expensive
            gain = float(best["accuracy"]) - rg_acc

            rows.append({
                "regime":             regime,
                "cost_ratio":         label,
                "best_adaptive_policy": best["route"],
                "accuracy":           _fmt(float(best["accuracy"])),
                "recomputed_avg_cost": _fmt(cost),
                "gain_vs_rg":         _fmt(gain),
            })

    if not rows:
        _record_skipped("policy_ranking_stability", "no data available")
        return

    out = _TABLES_OUT / "policy_ranking_stability.csv"
    _write_csv(out, rows, [
        "regime", "cost_ratio", "best_adaptive_policy",
        "accuracy", "recomputed_avg_cost", "gain_vs_rg",
    ])
    _record_generated(out)


# ---------------------------------------------------------------------------
# 9.  Figures
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> Any:
    """Import matplotlib, configure non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    return plt


def _figure_headroom_barplot() -> None:
    """routing_headroom_barplot.png – RG / DPR / oracle / best adaptive by regime."""
    plt = _setup_matplotlib()

    regimes: list[str] = []
    rg_accs: list[float] = []
    dpr_accs: list[float] = []
    oracle_accs: list[float | None] = []
    best_adaptive_accs: list[float | None] = []

    for regime, comp_path in _POLICY_COMP_FILES.items():
        if not comp_path.exists():
            continue

        records = _read_csv(comp_path)
        by_route = {r["route"]: r for r in records}
        rg  = by_route.get("reasoning_greedy")
        dpr = by_route.get("direct_plus_revise")
        if rg is None or dpr is None:
            continue

        regimes.append(regime)
        rg_accs.append(float(rg["accuracy"]))
        dpr_accs.append(float(dpr["accuracy"]))

        oracle_acc = None
        op = _ORACLE_FILES.get(regime)
        if op and op.exists():
            with op.open() as fh:
                od = json.load(fh)
            oracle_acc = od.get("accuracy")
        oracle_accs.append(oracle_acc)

        adaptive = [r for r in records if r["route"] not in ("reasoning_greedy", "direct_plus_revise")]
        best_adaptive_accs.append(
            max(float(r["accuracy"]) for r in adaptive) if adaptive else None
        )

    if not regimes:
        _record_skipped("routing_headroom_barplot.png", "no policy data available")
        return

    import numpy as np
    n = len(regimes)
    x = np.arange(n)
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(6, n * 1.8), 4))
    ax.bar(x - 1.5 * width, rg_accs, width, label="Always-RG", color="#4C72B0")
    ax.bar(x - 0.5 * width, dpr_accs, width, label="Always-DPR", color="#DD8452")
    ax.bar(x + 0.5 * width,
           [v if v is not None else 0 for v in best_adaptive_accs],
           width, label="Best Adaptive", color="#55A868")
    ax.bar(x + 1.5 * width,
           [v if v is not None else 0 for v in oracle_accs],
           width, label="Oracle", color="#C44E52", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Routing Headroom: Accuracy by Policy and Regime")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = _FIGURES_OUT / "routing_headroom_barplot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _record_generated(out)


def _figure_efficiency_scatter() -> None:
    """adaptive_efficiency_scatter.png – accuracy vs avg_cost per policy/regime."""
    plt = _setup_matplotlib()
    import numpy as np  # noqa: F401

    all_points: list[tuple[str, str, float, float]] = []  # regime, policy, cost, acc

    for regime, comp_path in _POLICY_COMP_FILES.items():
        if not comp_path.exists():
            continue
        for rec in _read_csv(comp_path):
            all_points.append((regime, rec["route"], float(rec["avg_cost"]), float(rec["accuracy"])))

    if not all_points:
        _record_skipped("adaptive_efficiency_scatter.png", "no data available")
        return

    # Color by regime, marker by policy type
    regime_labels = sorted(set(p[0] for p in all_points))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    regime_color = {r: colors[i % len(colors)] for i, r in enumerate(regime_labels)}

    marker_map = {
        "reasoning_greedy":   "s",
        "direct_plus_revise": "D",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for regime, policy, cost, acc in all_points:
        marker = marker_map.get(policy, "o")
        is_adaptive = policy not in ("reasoning_greedy", "direct_plus_revise")
        ax.scatter(cost, acc,
                   color=regime_color[regime],
                   marker=marker,
                   s=80,
                   alpha=0.85 if is_adaptive else 0.55,
                   zorder=3 if is_adaptive else 2)

    # Legend for regimes
    from matplotlib.lines import Line2D  # noqa: PLC0415
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=regime_color[r],
               markersize=8, label=r)
        for r in regime_labels
    ]
    handles += [
        Line2D([0], [0], marker="s", color="grey", linestyle="None", markersize=7, label="Always-RG"),
        Line2D([0], [0], marker="D", color="grey", linestyle="None", markersize=7, label="Always-DPR"),
        Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=7, label="Adaptive"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_xlabel("Average Cost")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Average Cost per Policy/Regime")
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = _FIGURES_OUT / "adaptive_efficiency_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _record_generated(out)


def _figure_oracle_gap_barplot() -> None:
    """oracle_gap_barplot.png – gap from best adaptive to oracle by regime."""
    plt = _setup_matplotlib()

    regimes: list[str] = []
    gaps: list[float] = []
    rg_gaps: list[float] = []

    for regime, comp_path in _POLICY_COMP_FILES.items():
        oracle_path = _ORACLE_FILES.get(regime)
        if not comp_path.exists() or not (oracle_path and oracle_path.exists()):
            continue

        records = _read_csv(comp_path)
        adaptive = [r for r in records if r["route"] not in ("reasoning_greedy", "direct_plus_revise")]
        rg = next((r for r in records if r["route"] == "reasoning_greedy"), None)
        if not adaptive or rg is None:
            continue

        best_adaptive_acc = max(float(r["accuracy"]) for r in adaptive)
        rg_acc = float(rg["accuracy"])

        with oracle_path.open() as fh:
            od = json.load(fh)
        oracle_acc = od.get("accuracy")
        if oracle_acc is None:
            continue

        regimes.append(regime)
        gaps.append(oracle_acc - best_adaptive_acc)
        rg_gaps.append(oracle_acc - rg_acc)

    if not regimes:
        _record_skipped("oracle_gap_barplot.png", "no oracle data available")
        return

    import numpy as np
    x = np.arange(len(regimes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, len(regimes) * 1.4), 4))
    ax.bar(x - width / 2, rg_gaps, width, label="Oracle – Always-RG", color="#4C72B0", alpha=0.8)
    ax.bar(x + width / 2, gaps, width, label="Oracle – Best Adaptive", color="#55A868", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=15, ha="right")
    ax.set_ylabel("Accuracy Gap")
    ax.set_title("Oracle Gap by Regime")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    out = _FIGURES_OUT / "oracle_gap_barplot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _record_generated(out)


def _figure_routing_outcome_stacked_bar() -> None:
    """routing_outcome_stacked_bar.png – stacked routing outcomes per regime."""
    plt = _setup_matplotlib()

    breakdown_path = _TABLES_OUT / "routing_outcome_breakdown.csv"
    if not breakdown_path.exists():
        _record_skipped("routing_outcome_stacked_bar.png", "routing_outcome_breakdown.csv not yet generated")
        return

    rows = _read_csv(breakdown_path)
    if not rows:
        _record_skipped("routing_outcome_stacked_bar.png", "empty routing_outcome_breakdown.csv")
        return

    regimes    = [r["regime"] for r in rows]
    both_c     = [float(r["frac_both_correct"]) for r in rows]
    rg_only    = [float(r["frac_rg_only"])     for r in rows]
    dpr_only   = [float(r["frac_dpr_only"])    for r in rows]
    both_wrong = [float(r["frac_both_wrong"])  for r in rows]

    import numpy as np
    x = np.arange(len(regimes))

    fig, ax = plt.subplots(figsize=(max(5, len(regimes) * 1.6), 4))
    ax.bar(x, both_c,     label="Both correct",    color="#55A868")
    ax.bar(x, rg_only,    bottom=both_c,            label="RG only correct", color="#4C72B0")
    ax.bar(x, dpr_only,   bottom=[a + b for a, b in zip(both_c, rg_only)],
           label="DPR only correct", color="#DD8452")
    ax.bar(x, both_wrong,
           bottom=[a + b + c for a, b, c in zip(both_c, rg_only, dpr_only)],
           label="Both wrong", color="#C44E52", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=15, ha="right")
    ax.set_ylabel("Fraction of Queries")
    ax.set_title("Routing Outcome Breakdown by Regime")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = _FIGURES_OUT / "routing_outcome_stacked_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _record_generated(out)


def _figure_threshold_tradeoff_curve() -> None:
    """threshold_tradeoff_curve.png – accuracy vs avg_cost for threshold sweep."""
    plt = _setup_matplotlib()

    if not _THRESHOLD_SWEEP_FILE.exists():
        _record_skipped("threshold_tradeoff_curve.png", f"missing {_THRESHOLD_SWEEP_FILE}")
        return

    rows = _read_csv(_THRESHOLD_SWEEP_FILE)
    from collections import defaultdict  # noqa: PLC0415
    by_regime: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_regime[r["regime"]].append(r)

    colors = _setup_matplotlib().cm.tab10.colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (regime, regime_rows) in enumerate(sorted(by_regime.items())):
        costs = [float(r["avg_cost"]) for r in regime_rows]
        accs  = [float(r["accuracy"]) for r in regime_rows]
        ax.plot(costs, accs, "o-", color=colors[i % len(colors)], label=regime, markersize=3)

    ax.set_xlabel("Average Cost")
    ax.set_ylabel("Accuracy")
    ax.set_title("Confidence Threshold Trade-off: Accuracy vs. Cost")
    ax.legend(fontsize=8)
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = _FIGURES_OUT / "threshold_tradeoff_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _record_generated(out)


# ---------------------------------------------------------------------------
# 10.  Final generation report
# ---------------------------------------------------------------------------

def _write_generation_report() -> None:
    lines = [
        "# Lightweight Artifact Generation Report",
        "",
        "Generated by `scripts/generate_lightweight_paper_artifacts.py`.",
        "All computations use only pre-existing stored outputs; no new LLM inference.",
        "",
        "## Generated Artifacts",
        "",
    ]

    if _generated:
        for p in _generated:
            lines.append(f"- `{p}`")
    else:
        lines.append("*(none)*")

    lines += [
        "",
        "## Skipped Artifacts",
        "",
    ]

    if _skipped:
        for name, reason in _skipped:
            lines.append(f"- **{name}**: {reason}")
    else:
        lines.append("*(none skipped)*")

    lines += [
        "",
        "## Most Useful for Paper Strengthening",
        "",
        "- `outputs/paper_tables/paired_policy_comparison.csv` – "
        "Core comparison table with gains and cost deltas vs. Always-RG and Always-DPR.",
        "- `outputs/paper_tables/bootstrap_accuracy_ci.csv` – "
        "95% confidence intervals for all key policy accuracies.",
        "- `outputs/paper_tables/paired_difference_tests.csv` – "
        "Statistical significance of adaptive vs. baseline differences.",
        "- `outputs/paper_tables/oracle_headroom_table.csv` – "
        "Upper-bound analysis showing remaining oracle gap.",
        "- `outputs/paper_figures/routing_headroom_barplot.png` – "
        "Clear visual summary for paper main results figure.",
        "- `outputs/paper_figures/routing_outcome_stacked_bar.png` – "
        "Routing headroom interpretation (how many queries adaptive routing can help).",
        "- `outputs/paper_tables/cost_ratio_sensitivity.csv` + "
        "`outputs/paper_tables/policy_ranking_stability.csv` – "
        "Robustness check for cost model assumptions.",
        "",
        "## Regeneration Command",
        "",
        "```bash",
        "python scripts/generate_lightweight_paper_artifacts.py",
        "```",
    ]

    out = _TABLES_OUT / "LIGHTWEIGHT_ARTIFACT_GENERATION_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Report written: {out.relative_to(_REPO)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== Lightweight Paper Artifact Generation ===")
    print(f"Repo root : {_REPO}")
    print(f"Tables out: {_TABLES_OUT}")
    print(f"Figures out: {_FIGURES_OUT}")
    print()

    # Tables
    print("[1/8] Generating paired_policy_comparison.csv ...")
    _generate_paired_policy_comparison()

    print("[2/8] Generating routing_outcome_breakdown.csv ...")
    _generate_routing_outcome_breakdown()

    print("[3/8] Generating policy_efficiency_table.csv ...")
    _generate_policy_efficiency_table()

    print("[4/8] Generating oracle_headroom_table.csv ...")
    _generate_oracle_headroom_table()

    print("[5/8] Generating bootstrap CI and paired difference tests ...")
    _generate_bootstrap_artifacts()

    print("[6/8] Generating threshold_sweep_summary.csv, cost_ratio_sensitivity.csv, policy_ranking_stability.csv ...")
    _generate_threshold_sweep_summary()
    _generate_cost_ratio_sensitivity()
    _generate_policy_ranking_stability()

    # Figures
    print("[7/8] Generating figures ...")
    _figure_headroom_barplot()
    _figure_efficiency_scatter()
    _figure_oracle_gap_barplot()
    _figure_routing_outcome_stacked_bar()
    _figure_threshold_tradeoff_curve()

    # Report
    print("[8/8] Writing generation report ...")
    _write_generation_report()

    print()
    print(f"Done. Generated: {len(_generated)}, Skipped: {len(_skipped)}")
    if _skipped:
        print("Skipped items:")
        for name, reason in _skipped:
            print(f"  - {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
