from __future__ import annotations

from src.routing_hybrid.features.registry import apply_feature_families


def test_feature_generation() -> None:
    rows = [
        {
            "prompt_id": "q1",
            "regime": "toy",
            "question": "Compute 10% of 50?",
            "split": "train",
            "action_name": "reasoning_greedy",
            "action_family": "reasoning",
            "action_cost": 1.0,
            "correctness_label": 1,
            "feat_unified_confidence_score": 0.7,
            "feat_unified_error_score": 0.2,
            "feat_fp_first_pass_parse_success": 1.0,
        }
    ]
    out = apply_feature_families(
        rows,
        ["prompt_features", "action_features", "interaction_features", "heuristic_features", "risk_features"],
    )
    r = out[0]
    assert "hfeat_prompt_length_chars" in r
    assert "hfeat_action_cost_rank" in r
    assert "hfeat_interaction_complexity_x_revise" in r
    assert "hfeat_heuristic_cheap_safe" in r
    assert "hfeat_risk_composite" in r
