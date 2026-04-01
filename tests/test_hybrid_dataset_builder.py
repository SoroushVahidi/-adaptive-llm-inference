from __future__ import annotations

from src.routing_hybrid.dataset_builder import build_candidate_rows


def test_candidate_row_construction() -> None:
    rows = [
        {
            "question_id": "q1",
            "question": "What is 1+1?",
            "regime": "toy",
            "split": "train",
            "action_reasoning_greedy_correct": "1",
            "action_reasoning_greedy_cost": "1.0",
            "action_direct_plus_revise_correct": "0",
            "action_direct_plus_revise_cost": "2.0",
            "feat_unified_confidence_score": "0.8",
        }
    ]
    out = build_candidate_rows(rows, utility_lambdas=[1.0])
    assert len(out) == 2
    assert {r["action_name"] for r in out} == {"reasoning_greedy", "direct_plus_revise"}
    assert all("u_correct_minus_lambda_cost_1" in r for r in out)
