from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _action_names(row: dict[str, str]) -> list[str]:
    actions: list[str] = []
    for k in row:
        if k.startswith("action_") and k.endswith("_correct"):
            actions.append(k[len("action_") : -len("_correct")])
    return sorted(actions)


def _family(action_name: str) -> str:
    if "revise" in action_name:
        return "revise"
    if "self_consistency" in action_name or "best_of" in action_name:
        return "multi_sample"
    if "direct" in action_name:
        return "direct"
    return "reasoning"


def _to_float(v: str | float | int, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def build_candidate_rows(
    rows: list[dict[str, str]],
    utility_lambdas: list[float],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    action_names = _action_names(rows[0])
    out: list[dict[str, Any]] = []
    for r in rows:
        costs = {a: _to_float(r.get(f"action_{a}_cost", 0.0)) for a in action_names}
        corrects = {a: int(_to_float(r.get(f"action_{a}_correct", 0))) for a in action_names}
        cheapest_action = min(action_names, key=lambda a: (costs[a], a))
        baseline_action = "reasoning_greedy" if "reasoning_greedy" in action_names else cheapest_action
        for a in action_names:
            utility_labels = {
                f"u_correct_minus_lambda_cost_{lam:g}": float(corrects[a]) - lam * costs[a]
                for lam in utility_lambdas
            }
            row_out = {
                "prompt_id": r["question_id"],
                "regime": r.get("regime", "unknown"),
                "question": r.get("question", ""),
                "split": r.get("split", "train"),
                "action_name": a,
                "action_family": _family(a),
                "action_cost": costs[a],
                "correctness_label": corrects[a],
                "first_pass_correct": int(_to_float(r.get("action_reasoning_greedy_correct", 0))),
                "revise_correct": int(_to_float(r.get("action_direct_plus_revise_correct", 0))),
                "answer_format": "numeric",
                "gain_vs_cheapest": corrects[a] - corrects[cheapest_action],
                "gain_vs_baseline": corrects[a] - corrects[baseline_action],
                "baseline_action": baseline_action,
                "cheapest_action": cheapest_action,
                "metadata": json.dumps(
                    {
                        "source": "routing_ml_dataset",
                        "label_policy": r.get("label_policy", ""),
                        "label_source": r.get("label_source", ""),
                    },
                    ensure_ascii=False,
                ),
            }
            row_out.update(utility_labels)
            for k, v in r.items():
                if k.startswith("feat_"):
                    row_out[k] = _to_float(v)
            out.append(row_out)
    return out


def write_candidate_artifacts(
    candidate_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cand_path = output_dir / "candidate_rows.csv"
    if not candidate_rows:
        raise RuntimeError("No candidate rows to write.")

    fieldnames = list(candidate_rows[0].keys())
    with cand_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(candidate_rows)

    # summaries
    by_action: dict[str, list[dict[str, Any]]] = {}
    by_regime: dict[str, list[dict[str, Any]]] = {}
    for r in candidate_rows:
        by_action.setdefault(str(r["action_name"]), []).append(r)
        by_regime.setdefault(str(r["regime"]), []).append(r)

    class_balance_path = output_dir / "class_balance_summary.csv"
    with class_balance_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["action_name", "rows", "correct_rate", "avg_cost"],
        )
        w.writeheader()
        for action, rows in sorted(by_action.items()):
            w.writerow(
                {
                    "action_name": action,
                    "rows": len(rows),
                    "correct_rate": mean(float(x["correctness_label"]) for x in rows),
                    "avg_cost": mean(float(x["action_cost"]) for x in rows),
                }
            )

    utility_cols = [c for c in candidate_rows[0].keys() if c.startswith("u_correct_minus_lambda_cost_")]
    utility_path = output_dir / "utility_summary.csv"
    with utility_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regime", "action_name", "utility_col", "mean_utility"])
        w.writeheader()
        for regime, rows in sorted(by_regime.items()):
            action_groups: dict[str, list[dict[str, Any]]] = {}
            for r in rows:
                action_groups.setdefault(str(r["action_name"]), []).append(r)
            for action, arows in sorted(action_groups.items()):
                for ucol in utility_cols:
                    w.writerow(
                        {
                            "regime": regime,
                            "action_name": action,
                            "utility_col": ucol,
                            "mean_utility": mean(float(x[ucol]) for x in arows),
                        }
                    )

    summary_path = output_dir / "dataset_summary.json"
    summary = {
        "num_candidate_rows": len(candidate_rows),
        "num_prompts": len({str(r["prompt_id"]) for r in candidate_rows}),
        "regimes": sorted(by_regime.keys()),
        "actions": sorted(by_action.keys()),
        "candidate_rows_csv": str(cand_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "candidate_rows_csv": str(cand_path),
        "dataset_summary_json": str(summary_path),
        "class_balance_summary_csv": str(class_balance_path),
        "utility_summary_csv": str(utility_path),
    }

