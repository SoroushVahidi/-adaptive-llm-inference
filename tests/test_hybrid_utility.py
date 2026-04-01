from __future__ import annotations

from src.routing_hybrid.utility import compute_candidate_utility


def test_utility_computation() -> None:
    row = {
        "pred_p_success": 0.8,
        "pred_gain": 0.2,
        "pred_utility": 0.7,
        "action_cost": 1.5,
        "baseline_cost": 1.0,
        "heur_utility_adjustment": 0.05,
        "pred_uncertainty": 0.1,
    }
    u = compute_candidate_utility(
        row,
        utility_name="expected_correct_minus_lambda_cost",
        lambda_cost=0.2,
        beta_uncertainty=0.1,
    )
    assert isinstance(u, float)


def test_new_utility_formula() -> None:
    row = {
        "pred_p_success": 0.8,
        "pred_reward": 1.0,
        "action_cost": 1.0,
        "heur_utility_adjustment": 0.0,
        "pred_uncertainty": 0.0,
    }
    u = compute_candidate_utility(
        row,
        utility_name="p_correct_times_reward_minus_cost",
        lambda_cost=0.5,
        beta_uncertainty=0.0,
    )
    assert abs(u - 0.3) < 1e-9
