#!/usr/bin/env python3
"""Generate authoritative final manuscript artifacts from committed outputs only.

This script is deterministic and repository-grounded:
- no API calls
- no new experiments
- fail fast on missing canonical inputs
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure export") from exc


REGIME_ORDER = [
    "gsm8k_random_100",
    "hard_gsm8k_100",
    "hard_gsm8k_b2",
    "math500_100",
]

REGIME_LABEL = {
    "gsm8k_random_100": "GSM8K Random-100",
    "hard_gsm8k_100": "Hard GSM8K-100",
    "hard_gsm8k_b2": "Hard GSM8K-B2",
    "math500_100": "MATH500-100",
}

SUMMARY_PATHS = {
    "gsm8k_random_100": "outputs/real_policy_eval/summary.json",
    "hard_gsm8k_100": "outputs/real_hard_gsm8k_policy_eval/summary.json",
    "hard_gsm8k_b2": "outputs/real_hard_gsm8k_b2_policy_eval/summary.json",
    "math500_100": "outputs/real_math500_policy_eval/summary.json",
}

ORACLE_PATHS = {
    "gsm8k_random_100": "outputs/oracle_routing_eval/gsm8k_random100_oracle_summary.json",
    "hard_gsm8k_100": "outputs/oracle_routing_eval/hard_gsm8k_100_oracle_summary.json",
    "hard_gsm8k_b2": "outputs/oracle_routing_eval/hard_gsm8k_b2_oracle_summary.json",
    "math500_100": "outputs/oracle_routing_eval/math500_100_oracle_summary.json",
}

BUDGET_PATHS = {
    "gsm8k_random_100": "outputs/budget_sweep/gsm8k_random100_budget_curve.csv",
    "hard_gsm8k_100": "outputs/budget_sweep/hard_gsm8k_100_budget_curve.csv",
    "hard_gsm8k_b2": "outputs/budget_sweep/hard_gsm8k_b2_budget_curve.csv",
    "math500_100": "outputs/budget_sweep/math500_100_budget_curve.csv",
}

REQUIRED_FILES = [
    *SUMMARY_PATHS.values(),
    *ORACLE_PATHS.values(),
    *BUDGET_PATHS.values(),
    "outputs/paper_tables/oracle_headroom_table.csv",
    "outputs/paper_tables/routing_outcome_breakdown.csv",
    "outputs/paper_tables/bootstrap_accuracy_ci.csv",
    "outputs/paper_tables/paired_difference_tests.csv",
    "outputs/baselines/confidence_threshold/confidence_threshold_summary.csv",
    "outputs/baselines/confidence_threshold/confidence_threshold_sweep.csv",
    "outputs/paper_tables/baselines/baselines_gsm8k_strategies.csv",
    "outputs/paper_tables/baselines/baselines_hard_gsm8k_strategies.csv",
    "outputs/paper_tables/baselines/baselines_math500_strategies.csv",
]


@dataclass
class RegimeMetrics:
    regime: str
    n: int
    revise_helpful_rate: float
    rg_acc: float
    rg_cost: float
    dpr_acc: float
    dpr_cost: float
    v5_acc: float
    v5_cost: float
    v6_acc: float
    v6_cost: float
    v7_acc: float
    v7_cost: float
    oracle_acc: float
    oracle_cost: float
    conf_acc: float
    conf_cost: float
    conf_threshold: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _budget_cost(row: dict[str, str]) -> float:
    if "avg_cost" in row:
        return float(row["avg_cost"])
    return float(row["achieved_avg_cost"])


def _budget_target(row: dict[str, str]) -> float:
    if "target_cost" in row:
        return float(row["target_cost"])
    return float(row["target_avg_cost"])


def _require_inputs(root: Path) -> None:
    missing = [p for p in REQUIRED_FILES if not (root / p).is_file()]
    if missing:
        msg = "Missing canonical input files:\n- " + "\n- ".join(missing)
        raise FileNotFoundError(msg)


def _comparison_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["route"]: r for r in summary.get("comparison", []) if isinstance(r, dict)}


def _load_confidence(root: Path) -> dict[str, dict[str, Any]]:
    rows = _read_csv(root / "outputs/baselines/confidence_threshold/confidence_threshold_summary.csv")
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r["regime"]] = {
            "threshold": float(r["threshold"]),
            "accuracy": float(r["accuracy"]),
            "avg_cost": float(r["avg_cost"]),
            "revise_rate": float(r["revise_rate"]),
            "n": int(float(r["n"])),
        }
    return out


def load_regime_metrics(root: Path) -> list[RegimeMetrics]:
    conf = _load_confidence(root)
    metrics: list[RegimeMetrics] = []
    for regime in REGIME_ORDER:
        summary = _read_json(root / SUMMARY_PATHS[regime])
        oracle = _read_json(root / ORACLE_PATHS[regime])
        comp = _comparison_map(summary)
        c = conf[regime]
        metrics.append(
            RegimeMetrics(
                regime=regime,
                n=int(summary["num_rows"]),
                revise_helpful_rate=float(summary["revise_helpful_prevalence"]),
                rg_acc=float(comp["reasoning_greedy"]["accuracy"]),
                rg_cost=float(comp["reasoning_greedy"]["avg_cost"]),
                dpr_acc=float(comp["direct_plus_revise"]["accuracy"]),
                dpr_cost=float(comp["direct_plus_revise"]["avg_cost"]),
                v5_acc=float(comp["adaptive_policy_v5"]["accuracy"]),
                v5_cost=float(comp["adaptive_policy_v5"]["avg_cost"]),
                v6_acc=float(comp["adaptive_policy_v6"]["accuracy"]),
                v6_cost=float(comp["adaptive_policy_v6"]["avg_cost"]),
                v7_acc=float(comp["adaptive_policy_v7"]["accuracy"]),
                v7_cost=float(comp["adaptive_policy_v7"]["avg_cost"]),
                oracle_acc=float(oracle["accuracy"]),
                oracle_cost=float(oracle["avg_cost"]),
                conf_acc=float(c["accuracy"]),
                conf_cost=float(c["avg_cost"]),
                conf_threshold=float(c["threshold"]),
            )
        )
    return metrics


def _best_adaptive_acc_cost(m: RegimeMetrics) -> tuple[str, float, float]:
    candidates = [
        ("adaptive_policy_v5", m.v5_acc, m.v5_cost),
        ("adaptive_policy_v6", m.v6_acc, m.v6_cost),
        ("adaptive_policy_v7", m.v7_acc, m.v7_cost),
    ]
    # Highest accuracy, then lower cost
    best = sorted(candidates, key=lambda x: (-x[1], x[2]))[0]
    return best


def generate_tables(root: Path, out_dir: Path, metrics: list[RegimeMetrics]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1 main_results_summary.csv
    rows = []
    for m in metrics:
        best_name, best_acc, best_cost = _best_adaptive_acc_cost(m)
        rows.append(
            {
                "regime": m.regime,
                "regime_label": REGIME_LABEL[m.regime],
                "n": m.n,
                "cheap_accuracy": round(m.rg_acc, 4),
                "cheap_cost": round(m.rg_cost, 4),
                "always_revise_accuracy": round(m.dpr_acc, 4),
                "always_revise_cost": round(m.dpr_cost, 4),
                "adaptive_primary_policy": "adaptive_policy_v5",
                "adaptive_primary_accuracy": round(m.v5_acc, 4),
                "adaptive_primary_cost": round(m.v5_cost, 4),
                "adaptive_best_accuracy_policy": best_name,
                "adaptive_best_accuracy": round(best_acc, 4),
                "adaptive_best_accuracy_cost": round(best_cost, 4),
                "oracle_accuracy": round(m.oracle_acc, 4),
                "oracle_cost": round(m.oracle_cost, 4),
                "revise_helpful_rate": round(m.revise_helpful_rate, 4),
            }
        )
    p = out_dir / "main_results_summary.csv"
    _write_csv(p, rows, list(rows[0].keys()))
    written.append(p)

    # 2 cross_regime_summary.csv
    rows = []
    for m in metrics:
        best_name, best_acc, best_cost = _best_adaptive_acc_cost(m)
        rows.append(
            {
                "regime": m.regime,
                "rg_acc": round(m.rg_acc, 4),
                "dpr_acc": round(m.dpr_acc, 4),
                "adaptive_v5_acc": round(m.v5_acc, 4),
                "adaptive_v6_acc": round(m.v6_acc, 4),
                "adaptive_v7_acc": round(m.v7_acc, 4),
                "best_adaptive_policy": best_name,
                "best_adaptive_acc": round(best_acc, 4),
                "best_adaptive_cost": round(best_cost, 4),
                "oracle_acc": round(m.oracle_acc, 4),
                "oracle_gap_vs_best_adaptive": round(m.oracle_acc - best_acc, 4),
                "revise_helpful_rate": round(m.revise_helpful_rate, 4),
            }
        )
    p = out_dir / "cross_regime_summary.csv"
    _write_csv(p, rows, list(rows[0].keys()))
    written.append(p)

    # 3 policy_comparison_main.csv
    rows = []
    for m in metrics:
        policy_rows = [
            ("reasoning_greedy", m.rg_acc, m.rg_cost, 0.0),
            ("adaptive_policy_v5", m.v5_acc, m.v5_cost, m.v5_cost - 1.0),
            ("adaptive_policy_v6", m.v6_acc, m.v6_cost, m.v6_cost - 1.0),
            ("adaptive_policy_v7", m.v7_acc, m.v7_cost, m.v7_cost - 1.0),
            ("direct_plus_revise", m.dpr_acc, m.dpr_cost, m.dpr_cost - 1.0),
            ("oracle", m.oracle_acc, m.oracle_cost, m.oracle_cost - 1.0),
        ]
        for pol, acc, cost, extra in policy_rows:
            rows.append(
                {
                    "regime": m.regime,
                    "policy": pol,
                    "accuracy": round(acc, 4),
                    "avg_cost": round(cost, 4),
                    "extra_cost_vs_cheap": round(extra, 4),
                    "oracle_gap": round(m.oracle_acc - acc, 4),
                }
            )
    p = out_dir / "policy_comparison_main.csv"
    _write_csv(p, rows, list(rows[0].keys()))
    written.append(p)

    # 4 oracle_headroom_main.csv (from existing headroom file, normalized order)
    src_headroom = _read_csv(root / "outputs/paper_tables/oracle_headroom_table.csv")
    src_map = {r["regime"]: r for r in src_headroom}
    rows = [src_map[r] for r in REGIME_ORDER if r in src_map]
    p = out_dir / "oracle_headroom_main.csv"
    _write_csv(p, rows, list(rows[0].keys()))
    written.append(p)

    # 5 routing_outcome_breakdown_main.csv
    src_outcome = _read_csv(root / "outputs/paper_tables/routing_outcome_breakdown.csv")
    src_map = {r["regime"]: r for r in src_outcome}
    rows = [src_map[r] for r in REGIME_ORDER if r in src_map]
    p = out_dir / "routing_outcome_breakdown_main.csv"
    _write_csv(p, rows, list(rows[0].keys()))
    written.append(p)

    # 6 budget_curve_main_points.csv (selected canonical points)
    target_points = [1.0, 1.1, 1.2, 2.0]
    rows = []
    for regime in REGIME_ORDER:
        curve = _read_csv(root / BUDGET_PATHS[regime])
        # choose nearest achieved avg cost for each target
        for t in target_points:
            best = min(curve, key=lambda r: abs(_budget_cost(r) - t))
            rows.append(
                {
                    "regime": regime,
                    "target_cost": t,
                    "target_budget_point": round(_budget_target(best), 4),
                    "selected_threshold": float(best["threshold"]) if "threshold" in best else "",
                    "achieved_cost": round(_budget_cost(best), 4),
                    "accuracy": round(float(best["accuracy"]), 4),
                    "revise_rate": round(float(best["revise_rate"]), 4),
                    "n": int(float(best["n"])) if "n" in best else 100,
                }
            )
    p = out_dir / "budget_curve_main_points.csv"
    _write_csv(p, rows, list(rows[0].keys()))
    written.append(p)

    # 7 baseline_comparison_appendix.csv
    rows = []
    for fn in [
        "outputs/paper_tables/baselines/baselines_gsm8k_strategies.csv",
        "outputs/paper_tables/baselines/baselines_hard_gsm8k_strategies.csv",
        "outputs/paper_tables/baselines/baselines_math500_strategies.csv",
    ]:
        for r in _read_csv(root / fn):
            rows.append(
                {
                    "source_file": fn,
                    "dataset": r["dataset"],
                    "strategy": r["strategy"],
                    "accuracy": round(float(r["accuracy"]), 4),
                    "avg_cost_proxy": round(float(r["avg_cost_proxy"]), 4),
                    "sample_size_note": "n<=30 (appendix only; not directly comparable to main n=100)",
                }
            )
    conf_rows = _read_csv(root / "outputs/baselines/confidence_threshold/confidence_threshold_summary.csv")
    for r in conf_rows:
        rows.append(
            {
                "source_file": "outputs/baselines/confidence_threshold/confidence_threshold_summary.csv",
                "dataset": r["regime"],
                "strategy": "confidence_threshold",
                "accuracy": round(float(r["accuracy"]), 4),
                "avg_cost_proxy": round(float(r["avg_cost"]), 4),
                "sample_size_note": "n=100 (supplementary baseline; threshold chosen by sweep)",
            }
        )
    p = out_dir / "baseline_comparison_appendix.csv"
    _write_csv(p, rows, list(rows[0].keys()))
    written.append(p)

    # 8 statistical_support_main.csv
    ci_rows = _read_csv(root / "outputs/paper_tables/bootstrap_accuracy_ci.csv")
    pd_rows = _read_csv(root / "outputs/paper_tables/paired_difference_tests.csv")
    rows = []
    for r in ci_rows:
        rows.append(
            {
                "type": "bootstrap_ci",
                "regime": r["regime"],
                "item": r["policy"],
                "metric_1": r["accuracy"],
                "metric_2": r["ci_lower_95"],
                "metric_3": r["ci_upper_95"],
                "metric_4": r["n"],
                "source_file": "outputs/paper_tables/bootstrap_accuracy_ci.csv",
            }
        )
    for r in pd_rows:
        rows.append(
            {
                "type": "paired_difference_test",
                "regime": r["regime"],
                "item": r["comparison"],
                "metric_1": r["mean_diff"],
                "metric_2": r["ci_lower_95"],
                "metric_3": r["ci_upper_95"],
                "metric_4": r["p_value_two_sided"],
                "source_file": "outputs/paper_tables/paired_difference_tests.csv",
            }
        )
    p = out_dir / "statistical_support_main.csv"
    _write_csv(p, rows, list(rows[0].keys()))
    written.append(p)

    # final README
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Final Paper Tables",
                "",
                "Authoritative deterministic table set generated from committed artifacts only.",
                "",
                "- `main_results_summary.csv` — Main-paper core regime summary (includes v5-v7, cheap, DPR, oracle).",
                "- `cross_regime_summary.csv` — Canonical cross-regime comparison with normalized regime names.",
                "- `policy_comparison_main.csv` — Main comparison rows used for plots and manuscript table text.",
                "- `oracle_headroom_main.csv` — Oracle gap/headroom summary (main paper).",
                "- `routing_outcome_breakdown_main.csv` — RG-only/DPR-only outcome decomposition (main paper).",
                "- `budget_curve_main_points.csv` — Canonical budget points (1.0, 1.1, 1.2, 2.0).",
                "- `baseline_comparison_appendix.csv` — Supplementary baseline comparison (appendix only).",
                "- `statistical_support_main.csv` — Bootstrap CI + paired tests (supporting stats).",
                "",
                "Main paper: first six files + statistical support as needed.",
                "Appendix/supplementary: `baseline_comparison_appendix.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(readme)
    return written


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def generate_figures(root: Path, out_dir: Path, metrics: list[RegimeMetrics]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    written: list[Path] = []

    regimes = [m.regime for m in metrics]
    labels = [REGIME_LABEL[r] for r in regimes]

    # 1 cross_regime_accuracy_cost.png (two-panel)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = range(len(metrics))
    ax1.plot(x, [m.rg_acc for m in metrics], marker="o", label="Cheap")
    ax1.plot(x, [m.v5_acc for m in metrics], marker="o", label="Adaptive v5")
    ax1.plot(x, [m.dpr_acc for m in metrics], marker="o", label="Always revise")
    ax1.plot(x, [m.oracle_acc for m in metrics], marker="o", label="Oracle")
    ax1.set_xticks(list(x), labels, rotation=20, ha="right")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by Regime")
    ax1.legend(fontsize=8)
    ax2.plot(x, [m.rg_cost for m in metrics], marker="o", label="Cheap")
    ax2.plot(x, [m.v5_cost for m in metrics], marker="o", label="Adaptive v5")
    ax2.plot(x, [m.dpr_cost for m in metrics], marker="o", label="Always revise")
    ax2.plot(x, [m.oracle_cost for m in metrics], marker="o", label="Oracle")
    ax2.set_xticks(list(x), labels, rotation=20, ha="right")
    ax2.set_ylabel("Average cost")
    ax2.set_title("Cost by Regime")
    ax2.legend(fontsize=8)
    fig.suptitle("Cross-regime Accuracy and Cost (Canonical Comparison)")
    fig.tight_layout()
    p = out_dir / "cross_regime_accuracy_cost.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 2 routing_headroom_barplot.png
    hrows = _read_csv(root / "outputs/paper_tables/oracle_headroom_table.csv")
    hmap = {r["regime"]: r for r in hrows}
    vals = [float(hmap[r]["adaptive_to_oracle_gap"]) for r in regimes]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, vals, color="#4C78A8")
    ax.set_ylabel("Oracle gap (oracle - best adaptive)")
    ax.set_title("Routing Headroom by Regime")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p = out_dir / "routing_headroom_barplot.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 3 routing_outcome_stacked_bar.png
    orows = _read_csv(root / "outputs/paper_tables/routing_outcome_breakdown.csv")
    omap = {r["regime"]: r for r in orows}
    both_correct = [float(omap[r]["frac_both_correct"]) for r in regimes]
    rg_only = [float(omap[r]["frac_rg_only"]) for r in regimes]
    dpr_only = [float(omap[r]["frac_dpr_only"]) for r in regimes]
    both_wrong = [float(omap[r]["frac_both_wrong"]) for r in regimes]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, both_correct, label="Both correct")
    ax.bar(labels, rg_only, bottom=both_correct, label="RG only correct")
    bottom2 = [a + b for a, b in zip(both_correct, rg_only)]
    ax.bar(labels, dpr_only, bottom=bottom2, label="DPR only correct")
    bottom3 = [a + b + c for a, b, c in zip(both_correct, rg_only, dpr_only)]
    ax.bar(labels, both_wrong, bottom=bottom3, label="Both wrong")
    ax.set_ylabel("Fraction of queries")
    ax.set_title("Routing Outcome Breakdown")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = out_dir / "routing_outcome_stacked_bar.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 4 oracle_gap_barplot.png
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cheap_gap = [m.oracle_acc - m.rg_acc for m in metrics]
    adaptive_gap = [m.oracle_acc - m.v5_acc for m in metrics]
    x = list(range(len(metrics)))
    w = 0.35
    ax.bar([i - w / 2 for i in x], cheap_gap, width=w, label="Cheap -> Oracle gap")
    ax.bar([i + w / 2 for i in x], adaptive_gap, width=w, label="Adaptive v5 -> Oracle gap")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy gap")
    ax.set_title("Oracle Gap Reduction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "oracle_gap_barplot.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 5 budget_curve_main.png (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, regime in zip(axes.flatten(), regimes):
        rows = _read_csv(root / BUDGET_PATHS[regime])
        rows = sorted(rows, key=_budget_cost)
        ax.plot([_budget_cost(r) for r in rows], [float(r["accuracy"]) for r in rows], marker="o", ms=3)
        ax.set_title(REGIME_LABEL[regime], fontsize=10)
        ax.set_xlabel("Achieved average cost")
        ax.set_ylabel("Accuracy")
    fig.suptitle("Budget Curves Across Main Regimes")
    fig.tight_layout()
    p = out_dir / "budget_curve_main.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 6 threshold_tradeoff_curve.png
    trows = _read_csv(root / "outputs/baselines/confidence_threshold/confidence_threshold_sweep.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, regime in zip(axes.flatten(), regimes):
        rr = [r for r in trows if r["regime"] == regime]
        rr = sorted(rr, key=lambda r: float(r["threshold"]))
        ax.plot([float(r["threshold"]) for r in rr], [float(r["accuracy"]) for r in rr], label="accuracy")
        ax2 = ax.twinx()
        ax2.plot([float(r["threshold"]) for r in rr], [float(r["avg_cost"]) for r in rr], color="tab:orange", label="avg_cost")
        ax.set_title(REGIME_LABEL[regime], fontsize=10)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Accuracy")
        ax2.set_ylabel("Cost")
    fig.suptitle("Confidence-Threshold Tradeoff Curves")
    fig.tight_layout()
    p = out_dir / "threshold_tradeoff_curve.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 7 adaptive_efficiency_scatter.png
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for m in metrics:
        ax.scatter(m.v5_cost, m.v5_acc, s=80, label=f"{REGIME_LABEL[m.regime]} (v5)")
    ax.scatter([m.rg_cost for m in metrics], [m.rg_acc for m in metrics], marker="s", s=50, label="Cheap", color="black")
    ax.scatter([m.dpr_cost for m in metrics], [m.dpr_acc for m in metrics], marker="^", s=50, label="Always revise", color="tab:red")
    ax.set_xlabel("Average cost")
    ax.set_ylabel("Accuracy")
    ax.set_title("Adaptive v5 Efficiency vs Baselines")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = out_dir / "adaptive_efficiency_scatter.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # Graphic abstract PNG+PDF + caption
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")
    ax.text(0.03, 0.92, "Input: mixed-difficulty reasoning queries", fontsize=11, weight="bold")
    ax.text(0.03, 0.78, "Cheap route (cost 1x)", fontsize=10, bbox=dict(boxstyle="round", fc="#DDEEFF"))
    ax.text(0.30, 0.78, "Adaptive router\n(lightweight signals)", fontsize=10, bbox=dict(boxstyle="round", fc="#FFF4CC"))
    ax.text(0.58, 0.78, "Revise route (cost ~2x)", fontsize=10, bbox=dict(boxstyle="round", fc="#E8F7E8"))
    ax.annotate("", xy=(0.28, 0.80), xytext=(0.17, 0.80), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.56, 0.80), xytext=(0.42, 0.80), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.03, 0.58, "Observed repository-grounded pattern:", fontsize=11, weight="bold")
    ax.text(0.05, 0.48, "- Adaptive v5 matches or improves over cheap across all four regimes.")
    ax.text(0.05, 0.42, "- Hard regimes show strongest gains (higher revise-helpful rates).")
    ax.text(0.05, 0.36, "- Oracle remains an upper bound (routing headroom exists).")
    ax.text(0.03, 0.22, "Canonical policy naming in this release: cheap, adaptive_policy_v5, always_revise, oracle.", fontsize=9)
    ax.text(0.03, 0.15, "No new experiments were run for this artifact bundle.", fontsize=9, style="italic")
    p = out_dir / "graphic_abstract.png"
    fig.savefig(p, dpi=260, bbox_inches="tight")
    fig.savefig(out_dir / "graphic_abstract.pdf", bbox_inches="tight")
    plt.close(fig)
    written.append(p)
    written.append(out_dir / "graphic_abstract.pdf")
    (out_dir / "graphic_abstract_caption.txt").write_text(
        "Graphic abstract: A lightweight adaptive router chooses between cheap reasoning and "
        "revision, improving the accuracy-cost tradeoff over always-cheap while using less cost than "
        "always-revise. The effect is regime-dependent (stronger on hard regimes with higher revise-helpful "
        "rates), with oracle routing showing remaining headroom.",
        encoding="utf-8",
    )
    written.append(out_dir / "graphic_abstract_caption.txt")

    return written


def write_final_report(
    out_tables: Path,
    table_files: list[Path],
    fig_files: list[Path],
    canonical_notes: list[str],
    blockers: list[str],
) -> Path:
    p = out_tables / "FINAL_ARTIFACT_EXPORT_REPORT.md"
    lines = [
        "# Final Artifact Export Report",
        "",
        "## Generated",
        "",
        "### Tables",
    ]
    lines.extend([f"- `{f.as_posix()}`" for f in table_files])
    lines.extend(["", "### Figures", ""])
    lines.extend([f"- `{f.as_posix()}`" for f in fig_files])
    lines.extend(["", "## Canonical decisions applied", ""])
    lines.extend([f"- {n}" for n in canonical_notes])
    lines.extend(["", "## Final manuscript assets", ""])
    lines.extend(
        [
            "- Main tables: `main_results_summary.csv`, `cross_regime_summary.csv`, `policy_comparison_main.csv`,",
            "  `oracle_headroom_main.csv`, `routing_outcome_breakdown_main.csv`, `budget_curve_main_points.csv`.",
            "- Main figures: `cross_regime_accuracy_cost.png`, `routing_headroom_barplot.png`,",
            "  `routing_outcome_stacked_bar.png`, `oracle_gap_barplot.png`, `budget_curve_main.png`,",
            "  `adaptive_efficiency_scatter.png`, `graphic_abstract.png`/`.pdf`.",
            "- Supplementary/appendix: `baseline_comparison_appendix.csv`, `threshold_tradeoff_curve.png`,",
            "  `statistical_support_main.csv`.",
        ]
    )
    lines.extend(["", "## Blockers / unresolved author decisions", ""])
    if blockers:
        lines.extend([f"- {b}" for b in blockers])
    else:
        lines.append("- None from required canonical inputs.")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_tables = root / "outputs/paper_tables_final"
    out_figs = root / "outputs/paper_figures_final"
    _require_inputs(root)
    metrics = load_regime_metrics(root)
    table_files = generate_tables(root, out_tables, metrics)
    fig_files = generate_figures(root, out_figs, metrics)
    canonical_notes = [
        "Regime names canonicalized to: gsm8k_random_100, hard_gsm8k_100, hard_gsm8k_b2, math500_100.",
        "Adaptive policy canonical main comparator: adaptive_policy_v5 (highest/tying accuracy on all four regimes; ties broken by transparency).",
        "v6/v7 retained in main comparison tables for honesty; v5 is not hidden.",
        "AIME and GPQA are supplementary-only in this export pass.",
        "No API calls or new experiment reruns performed.",
    ]
    blockers: list[str] = []
    report = write_final_report(out_tables, table_files, fig_files, canonical_notes, blockers)
    print(f"Wrote {len(table_files)} table artifacts")
    print(f"Wrote {len(fig_files)} figure artifacts")
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()
