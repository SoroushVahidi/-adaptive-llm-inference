"""Evaluate routing policies on rows from real_gsm8k_routing_dataset.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.policies.adaptive_policy_v5 import (
    AdaptivePolicyV5Config,
    extract_question_features_v5,
)
from src.policies.adaptive_policy_v5 import (
    choose_strategy as c5,
)
from src.policies.adaptive_policy_v6 import (
    AdaptivePolicyV6Config,
    extract_question_features_v6,
)
from src.policies.adaptive_policy_v6 import (
    choose_strategy as c6,
)
from src.policies.adaptive_policy_v7 import (
    AdaptivePolicyV7Config,
)
from src.policies.adaptive_policy_v7 import (
    choose_strategy as c7,
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _int(x: Any) -> int:
    try:
        return int(float(str(x).strip()))
    except (ValueError, TypeError):
        return 0


def run_real_policy_eval(
    dataset_csv: str | Path = "data/real_gsm8k_routing_dataset.csv",
    output_dir: str | Path = "outputs/real_policy_eval",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = Path(dataset_csv)

    if not p.exists():
        summary = {"run_status": "BLOCKED", "evidence_status": "blocked", "reason": str(p)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return {"summary": summary}

    rows = _read_rows(p)
    if not rows:
        summary = {"run_status": "BLOCKED", "evidence_status": "blocked", "reason": "empty csv"}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return {"summary": summary}

    v5c, v6c, v7c = AdaptivePolicyV5Config(), AdaptivePolicyV6Config(), AdaptivePolicyV7Config()

    per_rows: list[dict[str, Any]] = []
    for r in rows:
        q = str(r.get("question", ""))
        first = str(r.get("reasoning_raw", ""))
        if not q or not first:
            continue
        f5 = extract_question_features_v5(q, v5c)
        f6 = extract_question_features_v6(q, v6c)
        f7 = extract_question_features_v6(q, v7c)
        ch5 = c5(q, f5, first, v5c)
        ch6 = c6(q, f6, first, v6c)
        ch7 = c7(q, f7, first, v7c)
        rc = _int(r.get("reasoning_correct"))
        vc = _int(r.get("revise_correct"))
        per_rows.append(
            {
                "question_id": r.get("question_id", ""),
                "reasoning_correct": rc,
                "revise_correct": vc,
                "revise_helpful": _int(r.get("revise_helpful")),
                "policy_v5": ch5,
                "policy_v6": ch6,
                "policy_v7": ch7,
                "correct_if_v5": vc if ch5 == "direct_plus_revise" else rc,
                "correct_if_v6": vc if ch6 == "direct_plus_revise" else rc,
                "correct_if_v7": vc if ch7 == "direct_plus_revise" else rc,
                "cost_v5": 2 if ch5 == "direct_plus_revise" else 1,
                "cost_v6": 2 if ch6 == "direct_plus_revise" else 1,
                "cost_v7": 2 if ch7 == "direct_plus_revise" else 1,
            }
        )

    n = len(per_rows)
    if n == 0:
        summary = {
            "run_status": "BLOCKED",
            "evidence_status": "blocked",
            "reason": "no scorable rows",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return {"summary": summary}

    comparison = [
        {
            "route": "reasoning_greedy",
            "accuracy": sum(_int(r["reasoning_correct"]) for r in per_rows) / n,
            "avg_cost": 1.0,
            "revise_rate": 0.0,
        },
        {
            "route": "direct_plus_revise",
            "accuracy": sum(_int(r["revise_correct"]) for r in per_rows) / n,
            "avg_cost": 2.0,
            "revise_rate": 1.0,
        },
    ]

    comparison.append(
        {
            "route": "adaptive_policy_v5",
            "accuracy": sum(r["correct_if_v5"] for r in per_rows) / n,
            "avg_cost": sum(r["cost_v5"] for r in per_rows) / n,
            "revise_rate": sum(1 for r in per_rows if r["policy_v5"] == "direct_plus_revise") / n,
        }
    )
    comparison.append(
        {
            "route": "adaptive_policy_v6",
            "accuracy": sum(r["correct_if_v6"] for r in per_rows) / n,
            "avg_cost": sum(r["cost_v6"] for r in per_rows) / n,
            "revise_rate": sum(1 for r in per_rows if r["policy_v6"] == "direct_plus_revise") / n,
        }
    )
    comparison.append(
        {
            "route": "adaptive_policy_v7",
            "accuracy": sum(r["correct_if_v7"] for r in per_rows) / n,
            "avg_cost": sum(r["cost_v7"] for r in per_rows) / n,
            "revise_rate": sum(1 for r in per_rows if r["policy_v7"] == "direct_plus_revise") / n,
        }
    )

    v6_acc = comparison[-2]["accuracy"]
    v7_acc = comparison[-1]["accuracy"]
    summary = {
        "run_status": "COMPLETED",
        "evidence_status": "measured_now",
        "num_rows": n,
        "revise_helpful_prevalence": sum(_int(r["revise_helpful"]) for r in per_rows) / n,
        "v7_accuracy": v7_acc,
        "v6_accuracy": v6_acc,
        "v7_minus_v6_accuracy": round(v7_acc - v6_acc, 6),
        "comparison": comparison,
    }

    with (out / "policy_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comparison[0].keys()))
        w.writeheader()
        w.writerows(comparison)

    with (out / "per_query_policy_decisions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary}
