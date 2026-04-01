from __future__ import annotations

from src.routing_hybrid.heuristics.registry import apply_heuristics


def test_heuristic_pruning_and_bonus() -> None:
    rows = [
        {
            "prompt_id": "q1",
            "action_name": "reasoning_greedy",
            "action_cost": 1.0,
            "gain_vs_cheapest": 0.0,
            "hfeat_action_cost_rank": 0.0,
            "hfeat_prompt_unified_confidence": 0.9,
            "hfeat_action_is_revise": 0.0,
            "hfeat_action_is_direct": 0.0,
            "hfeat_heuristic_parse_risk": 0.1,
            "feat_fp_first_pass_parse_success": 1.0,
        },
        {
            "prompt_id": "q1",
            "action_name": "direct_plus_revise",
            "action_cost": 2.0,
            "gain_vs_cheapest": 0.0,
            "hfeat_action_cost_rank": 1.0,
            "hfeat_prompt_unified_confidence": 0.9,
            "hfeat_action_is_revise": 1.0,
            "hfeat_action_is_direct": 1.0,
            "hfeat_heuristic_parse_risk": 0.1,
            "feat_fp_first_pass_parse_success": 1.0,
        },
    ]
    out = apply_heuristics(rows, ["prune_dominated_cost", "cheap_safe", "ambiguity_penalty"])
    dominated = [r for r in out if r["action_name"] == "direct_plus_revise"][0]
    cheap = [r for r in out if r["action_name"] == "reasoning_greedy"][0]
    assert float(dominated["heur_forbidden"]) == 1.0
    assert float(cheap["heur_cheap_safe_flag"]) == 1.0
