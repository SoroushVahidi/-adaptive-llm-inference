from __future__ import annotations

from src.routing_hybrid.tree_router.selectors import assign_predicted_utility, select_actions


def test_selector_integration_smoke() -> None:
    rows = [
        {"prompt_id": "p1", "action_name": "a0", "action_cost": 1.0, "heur_utility_adjustment": 0.0},
        {"prompt_id": "p1", "action_name": "a1", "action_cost": 2.0, "heur_utility_adjustment": 0.0},
        {"prompt_id": "p2", "action_name": "a0", "action_cost": 1.0, "heur_utility_adjustment": 0.0},
        {"prompt_id": "p2", "action_name": "a1", "action_cost": 2.0, "heur_utility_adjustment": 0.0},
    ]
    scored = assign_predicted_utility(
        rows=rows,
        pred_score_by_index=[0.6, 0.7, 0.5, 0.8],
        utility_name="expected_correct_minus_lambda_cost",
        lambda_cost=0.2,
    )
    out = select_actions(scored, selector="mckp_exact", budget=3.0, optimizer_params={"cost_scale": 10})
    assert "chosen_by_prompt" in out
