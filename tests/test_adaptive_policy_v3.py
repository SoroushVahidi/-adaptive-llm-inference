from __future__ import annotations

from src.evaluation.adaptive_policy_v3_eval import _choose_best_setting
from src.policies.adaptive_policy_v3 import (
    AdaptivePolicyV3Config,
    choose_strategy,
    compute_revise_score,
    extract_question_features,
    extract_violation_signals,
)


def test_compute_revise_score_respects_weighted_signals() -> None:
    config = AdaptivePolicyV3Config(
        weight_final_answer_missing_or_unclear=2,
        weight_parse_failure=3,
        weight_malformed_output=0,
        weight_uncertainty_phrase_present=1,
        weight_too_many_intermediate_numbers_without_clear_final=0,
        weight_contradiction_like_phrase_present=0,
        weight_target_mismatch_suspected=0,
        weight_unit_mismatch_suspected=0,
        weight_impossible_value_suspected=0,
        revise_threshold=4,
    )
    signals = {
        "final_answer_missing_or_unclear": True,
        "parse_failure": False,
        "malformed_output": True,
        "uncertainty_phrase_present": True,
        "too_many_intermediate_numbers_without_clear_final": True,
        "contradiction_like_phrase_present": False,
        "target_mismatch_suspected": False,
        "unit_mismatch_suspected": False,
        "impossible_value_suspected": False,
    }

    score, contributing = compute_revise_score(signals, config)

    assert score == 3
    assert contributing == [
        "final_answer_missing_or_unclear",
        "uncertainty_phrase_present",
    ]


def test_choose_strategy_v3_prefers_reasoning_when_score_below_threshold() -> None:
    question = (
        "A train makes 3 trips per day for 4 days, then misses 2 trips. "
        "How many trips total?"
    )
    features = extract_question_features(question)
    config = AdaptivePolicyV3Config(revise_threshold=2)
    first_pass_output = (
        "Step 1: 3 trips/day * 4 days = 12.\n"
        "Step 2: 12 - 2 = 10.\n"
        "Final answer: 10"
    )

    assert choose_strategy(
        question_text=question,
        features=features,
        first_pass_output=first_pass_output,
        config=config,
    ) == "reasoning_greedy"


def test_choose_strategy_v3_triggers_revise_when_score_crosses_threshold() -> None:
    question = "How many apples are left after buying and sharing?"
    features = extract_question_features(question)
    config = AdaptivePolicyV3Config(revise_threshold=2)
    first_pass_output = "We compute 12 and 8, but maybe 6."

    assert choose_strategy(
        question_text=question,
        features=features,
        first_pass_output=first_pass_output,
        config=config,
    ) == "direct_plus_revise"


def test_choose_strategy_v3_optional_best_of_3_fallback() -> None:
    question = (
        "First calculate the total number of red marbles in 3 boxes, then subtract the 2 damaged "
        "bags, then multiply by 4 before dividing among 5 friends."
    )
    features = extract_question_features(question)
    config = AdaptivePolicyV3Config(
        revise_threshold=1,
        allow_reasoning_best_of_3=True,
        allow_strong_direct=False,
    )
    first_pass_output = (
        "We compute 12, 24, 36, 48, 60, 72, and 84 in different steps, "
        "but there is no final response summary."
    )

    assert choose_strategy(
        question_text=question,
        features=features,
        first_pass_output=first_pass_output,
        config=config,
    ) == "reasoning_best_of_3"


def test_pick_best_setting_prefers_balanced_non_extreme_revise_rate() -> None:
    rows = [
        {
            "setting": "conservative",
            "accuracy": 0.65,
            "avg_cost": 1.0,
            "revise_trigger_fraction": 0.0,
            "matches_or_beats_reasoning_greedy": True,
            "uses_less_cost_than_always_revise": True,
            "nontrivial_selectivity": False,
            "distance_to_oracle": -0.1,
        },
        {
            "setting": "medium",
            "accuracy": 0.70,
            "avg_cost": 1.5,
            "revise_trigger_fraction": 0.25,
            "matches_or_beats_reasoning_greedy": True,
            "uses_less_cost_than_always_revise": True,
            "nontrivial_selectivity": True,
            "distance_to_oracle": -0.05,
        },
        {
            "setting": "aggressive",
            "accuracy": 0.70,
            "avg_cost": 2.0,
            "revise_trigger_fraction": 1.0,
            "matches_or_beats_reasoning_greedy": True,
            "uses_less_cost_than_always_revise": False,
            "nontrivial_selectivity": False,
            "distance_to_oracle": -0.05,
        },
    ]

    best = _choose_best_setting(rows)

    assert best["setting"] == "medium"


def test_extract_violation_signals_are_reused_from_v2_shape() -> None:
    question = "How many apples are left?"
    output = "I might be wrong, but maybe 12 apples remain."
    signals = extract_violation_signals(question, output)

    assert "final_answer_missing_or_unclear" in signals
    assert "uncertainty_phrase_present" in signals
    assert "parse_failure" in signals
