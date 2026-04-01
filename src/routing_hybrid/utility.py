from __future__ import annotations

from typing import Any


def compute_candidate_utility(
    candidate_row: dict[str, Any],
    utility_name: str,
    lambda_cost: float,
    beta_uncertainty: float = 0.0,
) -> float:
    p_success = float(candidate_row.get("pred_p_success", 0.0))
    pred_gain = float(candidate_row.get("pred_gain", p_success))
    pred_utility = float(candidate_row.get("pred_utility", p_success))
    reward = float(candidate_row.get("pred_reward", pred_utility))
    cost = float(candidate_row.get("action_cost", 0.0))
    baseline_cost = float(candidate_row.get("baseline_cost", cost))
    heuristic_adj = float(candidate_row.get("heur_utility_adjustment", 0.0))
    uncertainty = float(candidate_row.get("pred_uncertainty", 0.0))

    if utility_name == "expected_correct_minus_lambda_cost":
        u = p_success - lambda_cost * cost
    elif utility_name == "gain_vs_baseline_minus_lambda_delta_cost":
        u = pred_gain - lambda_cost * (cost - baseline_cost)
    elif utility_name == "expected_reward_minus_cost":
        u = pred_utility - lambda_cost * cost
    elif utility_name == "heuristic_adjusted":
        u = pred_utility + heuristic_adj
    elif utility_name == "p_correct_times_reward_minus_cost":
        u = p_success * reward - lambda_cost * cost
    else:
        raise ValueError(f"Unknown utility '{utility_name}'")
    u = u + heuristic_adj - beta_uncertainty * uncertainty
    return float(u)

