from __future__ import annotations

from src.policies.adaptive_policy_v5 import (
    AdaptivePolicyV5Config,
    extract_question_features_v5,
)
from src.policies.adaptive_policy_v5 import (
    choose_strategy as choose_v5,
)
from src.policies.adaptive_policy_v6 import (
    choose_strategy,
    compute_v6_scores,
    extract_question_features_v6,
)


def test_choose_strategy_v6_defaults_to_reasoning_for_non_simple_question() -> None:
    question = "A machine makes 9 parts per hour for 7 hours. How many parts in total?"
    features = extract_question_features_v6(question)
    assert choose_strategy(question, features) == "reasoning_greedy"


def test_v6_does_not_revise_on_documented_concise_correct_answer() -> None:
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
        "clips in May. How many clips did Natalia sell altogether in April and May?"
    )
    first_pass = "Worked it out.\nFinal answer: 72"
    features = extract_question_features_v6(question)
    assert choose_strategy(question, features, first_pass) == "reasoning_greedy"
    state = compute_v6_scores(question, first_pass)
    assert state["revise_recommended"] is False
    assert state["answer_error_score"] == 0
    assert state["final_answer_confident"] is True


def test_v6_does_not_revise_on_weekday_answer_when_parsed() -> None:
    question_full = (
        "Jasmine had 3 paperclips on Monday, then she had 6 on Tuesday, and her number of "
        "paperclips proceeded to double on each subsequent day. On what day of the week did "
        "she first have more than 100 paperclips?"
    )
    first_pass = "Final answer: Sunday"
    features = extract_question_features_v6(question_full)
    state = compute_v6_scores(question_full, first_pass)
    assert state["categorical_question"] is True
    assert state["answer_error_score"] == 0
    assert choose_strategy(question_full, features, first_pass) == "reasoning_greedy"


def test_v6_revises_when_parsed_answer_echoes_intermediate_under_remaining_ask() -> None:
    question = "A bus has 40 seats. 26 are occupied. How many seats are left?"
    first_pass = "Final answer: 26"
    features = extract_question_features_v6(question)
    assert choose_strategy(question, features, first_pass) == "direct_plus_revise"
    state = compute_v6_scores(question, first_pass)
    assert state["revise_recommended"] is True
    assert "consistency_intermediate_echo_risk" in state["contributing_answer_error_signals"]


def test_v5_still_revise_on_same_concise_correct_fixture() -> None:
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
        "clips in May. How many clips did Natalia sell altogether in April and May?"
    )
    first_pass = "Worked it out.\nFinal answer: 72"
    feats = extract_question_features_v5(question)
    assert choose_v5(question, feats, first_pass, AdaptivePolicyV5Config()) == "direct_plus_revise"
