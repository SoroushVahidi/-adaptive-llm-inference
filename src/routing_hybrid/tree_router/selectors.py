from __future__ import annotations

from typing import Any

from src.routing_hybrid.optimizers.registry import make_optimizer
from src.routing_hybrid.utility import compute_candidate_utility


def assign_predicted_utility(
    rows: list[dict[str, Any]],
    pred_score_by_index: list[float],
    utility_name: str,
    lambda_cost: float,
    beta_uncertainty: float = 0.0,
) -> list[dict[str, Any]]:
    out = [dict(r) for r in rows]
    # baseline cost per prompt
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for r in out:
        by_prompt.setdefault(str(r["prompt_id"]), []).append(r)
    for prompt_rows in by_prompt.values():
        b = min(prompt_rows, key=lambda x: (float(x.get("action_cost", 0.0)), str(x.get("action_name", ""))))
        bc = float(b.get("action_cost", 0.0))
        for r in prompt_rows:
            r["baseline_cost"] = bc
    for i, r in enumerate(out):
        p = float(pred_score_by_index[i])
        r["pred_p_success"] = p
        r["pred_gain"] = p - 0.5
        r["pred_utility"] = p
        r["pred_reward"] = 1.0
        r["pred_uncertainty"] = abs(0.5 - p)
        r["final_utility"] = compute_candidate_utility(
            r,
            utility_name=utility_name,
            lambda_cost=lambda_cost,
            beta_uncertainty=beta_uncertainty,
        )
    return out


def select_actions(
    rows: list[dict[str, Any]],
    selector: str,
    budget: float,
    optimizer_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    optimizer = make_optimizer(selector, optimizer_params=optimizer_params)
    return optimizer.solve(rows, budget=budget)

