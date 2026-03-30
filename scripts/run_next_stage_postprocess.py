#!/usr/bin/env python3
"""Oracle routing, budget sweep, cascade from enriched routing CSV + policy summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.next_stage_experiments import (  # noqa: E402
    budget_curve_marginal_gain,
    cascade_curve,
    oracle_revise_helpful_summary,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-key", required=True, help="e.g. gsm8k_random100")
    p.add_argument("--routing-csv", required=True)
    p.add_argument("--policy-summary-json", default="")
    p.add_argument("--out-dir", default="outputs/next_stage_eval")
    p.add_argument("--oracle-dir", default="outputs/oracle_routing_eval")
    p.add_argument("--budget-dir", default="outputs/budget_sweep")
    args = p.parse_args()

    csv_path = Path(args.routing_csv)
    if not csv_path.exists():
        print(f"BLOCKED: missing {csv_path}", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("status", "ok") != "error"]

    out = Path(args.out_dir) / args.dataset_key
    out.mkdir(parents=True, exist_ok=True)

    oracle = oracle_revise_helpful_summary(rows)
    oracle_dir = Path(args.oracle_dir)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    oracle_payload = {"dataset": args.dataset_key, **oracle}
    (out / "oracle_revise_helpful_summary.json").write_text(
        json.dumps(oracle_payload, indent=2), encoding="utf-8"
    )
    (oracle_dir / f"{args.dataset_key}_oracle_summary.json").write_text(
        json.dumps(oracle_payload, indent=2), encoding="utf-8"
    )

    targets = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    curve = budget_curve_marginal_gain(rows, targets)
    budget_dir = Path(args.budget_dir)
    budget_dir.mkdir(parents=True, exist_ok=True)
    with (out / "budget_curve.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()) if curve else [])
        if curve:
            w.writeheader()
            w.writerows(curve)
    with (budget_dir / f"{args.dataset_key}_budget_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()) if curve else [])
        if curve:
            w.writeheader()
            w.writerows(curve)

    thr = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    casc = cascade_curve(rows, thr)
    with (out / "cascade_curve.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(casc[0].keys()) if casc else [])
        if casc:
            w.writeheader()
            w.writerows(casc)

    best_pol = ""
    best_acc = 0.0
    best_cost = 0.0
    if args.policy_summary_json and Path(args.policy_summary_json).exists():
        ps = json.loads(Path(args.policy_summary_json).read_text(encoding="utf-8"))
        for row in ps.get("comparison", []):
            name = str(row.get("route", ""))
            if name not in ("adaptive_policy_v6", "adaptive_policy_v7"):
                continue
            acc = float(row.get("accuracy", 0))
            cost = float(row.get("avg_cost", 0))
            if acc > best_acc or (acc == best_acc and cost < best_cost):
                best_acc, best_cost, best_pol = acc, cost, name

    rtr_acc = None
    if rows and "reasoning_then_revise_correct" in rows[0]:
        rtr_acc = sum(int(r.get("reasoning_then_revise_correct", 0)) for r in rows) / len(rows)

    merged = {
        "dataset": args.dataset_key,
        "oracle": oracle,
        "budget_curve": curve,
        "cascade_sample": casc,
        "best_v6_v7_policy": best_pol,
        "best_v6_v7_accuracy": best_acc,
        "best_v6_v7_cost": best_cost,
        "reasoning_then_revise_accuracy": rtr_acc,
    }
    (out / "next_stage_merged.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
