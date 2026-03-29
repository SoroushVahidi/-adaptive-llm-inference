from __future__ import annotations

from src.policies.adaptive_policy_v5 import (
    AdaptivePolicyV5Config,
    choose_strategy,
    extract_question_features_v5,
    extract_weighted_role_state,
)


def test_choose_strategy_v5_defaults_to_reasoning_for_non_simple_question() -> None:
    question = "A machine makes 9 parts per hour for 7 hours. How many parts in total?"
    features = extract_question_features_v5(question)
    assert choose_strategy(question, features) == "reasoning_greedy"


def test_choose_strategy_v5_triggers_revise_on_missing_required_numbers() -> None:
    question = "A bus has 40 seats. 26 are occupied. How many seats are left?"
    features = extract_question_features_v5(question)
    first_pass = "Final answer: 26 seats are occupied."
    chosen = choose_strategy(
        question_text=question,
        features=features,
        first_pass_output=first_pass,
        config=AdaptivePolicyV5Config(maybe_escalate_threshold=1, strong_escalate_threshold=2),
    )
    assert chosen == "direct_plus_revise"


def test_extract_weighted_role_state_contains_calibrated_role_features() -> None:
    question = "A class has 31 students and vans hold at most 7 students each. Minimum vans?"
    first_pass = "31 divided by 7 is about 4.4. Final answer: 4"
    state = extract_weighted_role_state(question, first_pass, parsed_answer="4")
    assert "role_calibrated" in state
    assert "constraint_features" in state
    assert "unified_state" in state
    assert isinstance(state["revise_recommended"], bool)
