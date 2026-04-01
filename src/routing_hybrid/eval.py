from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


POLICY_V5_PATHS = {
    "gsm8k_random_100": Path("outputs/real_policy_eval/per_query_policy_decisions.csv"),
    "hard_gsm8k_100": Path("outputs/real_hard_gsm8k_policy_eval/per_query_policy_decisions.csv"),
    "hard_gsm8k_b2": Path("outputs/real_hard_gsm8k_b2_policy_eval/per_query_policy_decisions.csv"),
    "math500_100": Path("outputs/real_math500_policy_eval/per_query_policy_decisions.csv"),
}


def evaluate_chosen_actions(
    candidate_rows: list[dict[str, Any]],
    chosen_by_prompt: dict[str, str],
) -> dict[str, Any]:
    rows_by_key = {(str(r["prompt_id"]), str(r["action_name"])): r for r in candidate_rows}
    chosen_rows: list[dict[str, Any]] = []
    for pid, action in chosen_by_prompt.items():
        row = rows_by_key.get((pid, action))
        if row is not None:
            chosen_rows.append(row)
    n = max(1, len(chosen_rows))
    acc = sum(float(r["correctness_label"]) for r in chosen_rows) / n
    avg_cost = sum(float(r["action_cost"]) for r in chosen_rows) / n
    avg_utility = sum(float(r["final_utility"]) for r in chosen_rows) / n
    action_counts: dict[str, int] = {}
    for r in chosen_rows:
        a = str(r["action_name"])
        action_counts[a] = action_counts.get(a, 0) + 1
    return {
        "num_prompts": len(chosen_rows),
        "final_accuracy": float(acc),
        "average_cost": float(avg_cost),
        "average_utility": float(avg_utility),
        "action_distribution": action_counts,
    }


def compute_simple_baselines(candidate_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for r in candidate_rows:
        by_prompt.setdefault(str(r["prompt_id"]), []).append(r)
    cheapest = {}
    expensive = {}
    oracle = {}
    for pid, rows in by_prompt.items():
        cheapest[pid] = min(rows, key=lambda x: (float(x["action_cost"]), str(x["action_name"])))
        expensive[pid] = max(rows, key=lambda x: (float(x["action_cost"]), str(x["action_name"])))
        oracle[pid] = max(rows, key=lambda x: (float(x["correctness_label"]), -float(x["action_cost"]), str(x["action_name"])))

    def _metrics(sel: dict[str, dict[str, Any]]) -> dict[str, float]:
        vals = list(sel.values())
        n = max(1, len(vals))
        return {
            "accuracy": sum(float(r["correctness_label"]) for r in vals) / n,
            "avg_cost": sum(float(r["action_cost"]) for r in vals) / n,
        }

    baselines = {
        "cheapest_only": _metrics(cheapest),
        "always_expensive": _metrics(expensive),
        "oracle_upper_bound": _metrics(oracle),
    }
    # Optional comparable baseline: adaptive policy v5 where files exist.
    policy_map: dict[tuple[str, str], str] = {}
    for regime, path in POLICY_V5_PATHS.items():
        if not path.is_file():
            continue
        for row in _read_csv(path):
            policy_map[(regime, row["question_id"])] = row.get("policy_v5", "")
    if policy_map:
        selected: dict[str, dict[str, Any]] = {}
        for pid, rows in by_prompt.items():
            regime = str(rows[0].get("regime", ""))
            action = policy_map.get((regime, pid))
            if not action:
                continue
            match = next((r for r in rows if str(r["action_name"]) == action), None)
            if match is not None:
                selected[pid] = match
        if selected:
            baselines["adaptive_policy_v5"] = _metrics(selected)
    return baselines

