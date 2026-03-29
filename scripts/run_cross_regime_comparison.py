#!/usr/bin/env python3
"""Aggregate cross-regime metrics from committed run summaries."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _policy_row(comp: list[dict], name: str) -> tuple[float, float]:
    for r in comp:
        if r["route"] == name:
            return float(r["accuracy"]), float(r["avg_cost"])
    return float("nan"), float("nan")


def _best_v6_v7(comp: list[dict]) -> tuple[str, float, float]:
    heur = [r for r in comp if r["route"] in ("adaptive_policy_v6", "adaptive_policy_v7")]
    if not heur:
        return "", 0.0, 0.0
    best = max(heur, key=lambda x: (float(x["accuracy"]), -float(x["avg_cost"])))
    return str(best["route"]), float(best["accuracy"]), float(best["avg_cost"])


def _router_viable(summary: dict) -> str:
    if summary.get("run_status") != "OK":
        return "no"
    pos = int(summary.get("num_positive", 0))
    if pos < 2:
        return "no_signal"
    f1s = [float(m.get("f1", 0)) for m in summary.get("models", [])]
    best_f1 = max(f1s) if f1s else 0.0
    if best_f1 >= 0.5 and pos >= 6:
        return "yes_weak"
    if best_f1 >= 0.35 or pos >= 10:
        return "marginal_learner_signal"
    return "marginal_sparse"


def main() -> None:
    out = Path("outputs/cross_regime_comparison")
    out.mkdir(parents=True, exist_ok=True)

    regimes = [
        {
            "regime": "gsm8k_random100",
            "build": REPO_ROOT / "outputs/real_routing_dataset/gsm8k_subset_run_summary.json",
            "policy": REPO_ROOT / "outputs/real_policy_eval/summary.json",
            "router": REPO_ROOT / "outputs/real_routing_model/summary.json",
        },
        {
            "regime": "math500_100",
            "build": REPO_ROOT / "outputs/real_math500_routing/math500_run_summary.json",
            "policy": REPO_ROOT / "outputs/real_math500_policy_eval/summary.json",
            "router": REPO_ROOT / "outputs/real_math500_routing_model/summary.json",
        },
        {
            "regime": "hard_gsm8k_100",
            "build": REPO_ROOT / "outputs/real_hard_gsm8k_routing/hard_gsm8k_run_summary.json",
            "policy": REPO_ROOT / "outputs/real_hard_gsm8k_policy_eval/summary.json",
            "router": REPO_ROOT / "outputs/real_hard_gsm8k_routing_model/summary.json",
        },
    ]

    rows: list[dict[str, str | float]] = []
    for block in regimes:
        rname = str(block["regime"])
        row: dict[str, str | float] = {"regime": rname}
        bp, pp, rp = block["build"], block["policy"], block["router"]

        if bp.exists():
            b = _load_json(bp)
            row["reasoning_greedy_accuracy"] = float(b.get("reasoning_accuracy", 0))
            row["direct_plus_revise_accuracy"] = float(b.get("revise_accuracy", 0))
            row["revise_helpful_prevalence"] = float(b.get("revise_helpful_rate", 0))
            row["num_queries_ok"] = int(b.get("num_queries_ok", 0))
        else:
            row["reasoning_greedy_accuracy"] = float("nan")
            row["direct_plus_revise_accuracy"] = float("nan")
            row["revise_helpful_prevalence"] = float("nan")
            row["num_queries_ok"] = 0

        if pp.exists():
            p = _load_json(pp)
            comp = p.get("comparison", [])
            v6a, v6c = _policy_row(comp, "adaptive_policy_v6")
            v7a, v7c = _policy_row(comp, "adaptive_policy_v7")
            row["adaptive_v6_accuracy"] = v6a
            row["adaptive_v6_avg_cost"] = v6c
            row["adaptive_v7_accuracy"] = v7a
            row["adaptive_v7_avg_cost"] = v7c
            name, acc, cost = _best_v6_v7(comp)
            row["best_v6_v7_policy"] = name
            row["best_v6_v7_accuracy"] = acc
            row["best_v6_v7_avg_cost"] = cost
            row["v7_minus_v6_accuracy"] = float(p.get("v7_minus_v6_accuracy", 0))
        else:
            row["adaptive_v6_accuracy"] = float("nan")
            row["adaptive_v6_avg_cost"] = float("nan")
            row["adaptive_v7_accuracy"] = float("nan")
            row["adaptive_v7_avg_cost"] = float("nan")
            row["best_v6_v7_policy"] = ""
            row["best_v6_v7_accuracy"] = float("nan")
            row["best_v6_v7_avg_cost"] = float("nan")
            row["v7_minus_v6_accuracy"] = float("nan")

        if rp.exists():
            row["learned_router_viability"] = _router_viable(_load_json(rp))
        else:
            row["learned_router_viability"] = "missing"

        rows.append(row)

    csv_path = out / "cross_regime_summary.csv"
    json_path = out / "cross_regime_summary.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
