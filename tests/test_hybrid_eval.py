from __future__ import annotations

from src.routing_hybrid.eval import evaluate_chosen_actions


def test_end_to_end_eval_smoke() -> None:
    rows = [
        {"prompt_id": "p1", "action_name": "a0", "correctness_label": 1, "action_cost": 1.0, "final_utility": 0.5},
        {"prompt_id": "p1", "action_name": "a1", "correctness_label": 0, "action_cost": 2.0, "final_utility": 0.2},
        {"prompt_id": "p2", "action_name": "a0", "correctness_label": 0, "action_cost": 1.0, "final_utility": 0.1},
        {"prompt_id": "p2", "action_name": "a1", "correctness_label": 1, "action_cost": 2.0, "final_utility": 0.8},
    ]
    chosen = {"p1": "a0", "p2": "a1"}
    out = evaluate_chosen_actions(rows, chosen)
    assert out["num_prompts"] == 2
    assert out["final_accuracy"] == 1.0
