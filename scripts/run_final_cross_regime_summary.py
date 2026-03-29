#!/usr/bin/env python3
"""Single CSV summarizing key metrics across datasets."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.next_stage_experiments import (  # noqa: E402
    best_policy_v6_v7_from_eval_summary,
    oracle_revise_helpful_summary,
)


def _mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get(key, 0) or 0) for r in rows) / len(rows)


def load_ok_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    return [r for r in rows if r.get("status", "ok") != "error"]


def main() -> int:
    out = Path("outputs/cross_regime_comparison/final_cross_regime_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    def pick_csv(base: Path, enriched: Path) -> Path:
        return enriched if enriched.exists() else base

    blocks = [
        {
            "dataset": "gsm8k_random100",
            "routing_csv": pick_csv(
                REPO_ROOT / "data/real_gsm8k_routing_dataset.csv",
                REPO_ROOT / "data/real_gsm8k_routing_dataset_enriched.csv",
            ),
            "policy_json": REPO_ROOT / "outputs/real_policy_eval/summary.json",
        },
        {
            "dataset": "hard_gsm8k_100",
            "routing_csv": pick_csv(
                REPO_ROOT / "data/real_hard_gsm8k_routing_dataset.csv",
                REPO_ROOT / "data/real_hard_gsm8k_routing_dataset_enriched.csv",
            ),
            "policy_json": REPO_ROOT / "outputs/real_hard_gsm8k_policy_eval/summary.json",
        },
        {
            "dataset": "math500_100",
            "routing_csv": pick_csv(
                REPO_ROOT / "data/real_math500_routing_dataset.csv",
                REPO_ROOT / "data/real_math500_routing_dataset_enriched.csv",
            ),
            "policy_json": REPO_ROOT / "outputs/real_math500_policy_eval/summary.json",
        },
    ]
    aime_csv = REPO_ROOT / "data/real_aime2024_routing_dataset.csv"
    if aime_csv.exists():
        blocks.append(
            {
                "dataset": "aime2024",
                "routing_csv": aime_csv,
                "policy_json": "",
            }
        )

    out_rows: list[dict[str, str | float]] = []
    for b in blocks:
        rows = load_ok_rows(Path(b["routing_csv"]))
        if not rows:
            print(f"WARNING: missing or empty {b['routing_csv']}", file=sys.stderr)
            continue
        oracle = oracle_revise_helpful_summary(rows)
        best_n, best_a, best_c = "", 0.0, 0.0
        pj = b.get("policy_json") or ""
        if pj and Path(pj).exists():
            best_n, best_a, best_c = best_policy_v6_v7_from_eval_summary(Path(pj))
        rtr = None
        if rows and "reasoning_then_revise_correct" in rows[0]:
            rtr = sum(int(r.get("reasoning_then_revise_correct", 0)) for r in rows) / len(rows)
        out_rows.append(
            {
                "dataset": b["dataset"],
                "reasoning_csv_used": str(b["routing_csv"].relative_to(REPO_ROOT)),
                "reasoning_accuracy": round(_mean(rows, "reasoning_correct"), 4),
                "revise_accuracy": round(_mean(rows, "revise_correct"), 4),
                "oracle_accuracy": round(oracle["accuracy"], 4),
                "best_policy_accuracy": round(best_a, 4) if best_n else "",
                "best_policy_cost": round(best_c, 4) if best_n else "",
                "best_policy_name": best_n,
                "revise_helpful_rate": round(_mean(rows, "revise_helpful"), 4),
                "reasoning_then_revise_accuracy": round(rtr, 4) if rtr is not None else "",
            }
        )

    if not out_rows:
        print("BLOCKED: no routing CSVs found", file=sys.stderr)
        return 2

    fieldnames = list(out_rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(json.dumps(out_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
