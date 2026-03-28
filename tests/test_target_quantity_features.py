"""Unit tests for src/features/target_quantity_features.py."""

from __future__ import annotations

from src.features.target_quantity_features import extract_target_quantity_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_EXPECTED_KEYS = {
    "asks_remaining_or_left",
    "asks_total",
    "asks_difference",
    "asks_rate_or_unit",
    "asks_money",
    "asks_time",
    "has_subtraction_trap_verb",
    "has_addition_trap_structure",
    "has_multi_operation_hint",
    "likely_intermediate_quantity_ask",
    "potential_answer_echo_risk",
}


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


def test_returns_all_expected_keys() -> None:
    feats = extract_target_quantity_features("Janet has 3 apples and buys 4 more.")
    assert set(feats.keys()) == _ALL_EXPECTED_KEYS


def test_all_values_are_bool() -> None:
    feats = extract_target_quantity_features("She spent $5 on apples.")
    for key, value in feats.items():
        assert isinstance(value, bool), f"Feature '{key}' is not bool: {type(value)}"


def test_empty_string_does_not_crash() -> None:
    feats = extract_target_quantity_features("")
    assert set(feats.keys()) == _ALL_EXPECTED_KEYS
    for value in feats.values():
        assert isinstance(value, bool)


def test_no_numbers_does_not_crash() -> None:
    feats = extract_target_quantity_features("What is the answer?")
    assert set(feats.keys()) == _ALL_EXPECTED_KEYS


# ---------------------------------------------------------------------------
# A. Target-type cues — asks_remaining_or_left
# ---------------------------------------------------------------------------


def test_asks_remaining_or_left_remaining_keyword() -> None:
    feats = extract_target_quantity_features(
        "Janet has 10 apples. She gives 3 away. How many are remaining?"
    )
    assert feats["asks_remaining_or_left"] is True


def test_asks_remaining_or_left_left_keyword() -> None:
    feats = extract_target_quantity_features(
        "She had 20 dollars and spent 7. How much is left?"
    )
    assert feats["asks_remaining_or_left"] is True


def test_asks_remaining_or_left_leftover_keyword() -> None:
    feats = extract_target_quantity_features(
        "There were 15 cookies. They ate 6. How many are left over?"
    )
    assert feats["asks_remaining_or_left"] is True


def test_asks_remaining_or_left_absent() -> None:
    feats = extract_target_quantity_features(
        "She bought 3 apples and 2 oranges. How many did she buy in total?"
    )
    assert feats["asks_remaining_or_left"] is False


def test_asks_remaining_or_left_case_insensitive() -> None:
    feats = extract_target_quantity_features("How many are REMAINING?")
    assert feats["asks_remaining_or_left"] is True


# ---------------------------------------------------------------------------
# A. Target-type cues — asks_total
# ---------------------------------------------------------------------------


def test_asks_total_total_keyword() -> None:
    feats = extract_target_quantity_features(
        "She earns $10 per day. What is her total earnings after 5 days?"
    )
    assert feats["asks_total"] is True


def test_asks_total_altogether_keyword() -> None:
    feats = extract_target_quantity_features(
        "He has 4 red balls and 6 blue balls. How many altogether?"
    )
    assert feats["asks_total"] is True


def test_asks_total_in_all_keyword() -> None:
    feats = extract_target_quantity_features(
        "She bought 3 apples and 2 oranges. How many in all?"
    )
    assert feats["asks_total"] is True


def test_asks_total_absent() -> None:
    feats = extract_target_quantity_features(
        "She had 20 dollars and spent 7. How much is left?"
    )
    assert feats["asks_total"] is False


# ---------------------------------------------------------------------------
# A. Target-type cues — asks_difference
# ---------------------------------------------------------------------------


def test_asks_difference_difference_keyword() -> None:
    feats = extract_target_quantity_features(
        "John has 15 marbles and Mary has 8. What is the difference?"
    )
    assert feats["asks_difference"] is True


def test_asks_difference_more_than_keyword() -> None:
    feats = extract_target_quantity_features(
        "Tom scored 30 and Jerry scored 20. How much more than Jerry did Tom score?"
    )
    assert feats["asks_difference"] is True


def test_asks_difference_less_than_keyword() -> None:
    feats = extract_target_quantity_features(
        "Box A has 12 items and Box B has 5 fewer than Box A."
    )
    assert feats["asks_difference"] is True


def test_asks_difference_absent() -> None:
    feats = extract_target_quantity_features(
        "She bought 3 apples. How many does she have in total?"
    )
    assert feats["asks_difference"] is False


# ---------------------------------------------------------------------------
# A. Target-type cues — asks_rate_or_unit
# ---------------------------------------------------------------------------


def test_asks_rate_or_unit_per_keyword() -> None:
    feats = extract_target_quantity_features(
        "A factory produces 50 units per day. How many in 5 days?"
    )
    assert feats["asks_rate_or_unit"] is True


def test_asks_rate_or_unit_each_keyword() -> None:
    feats = extract_target_quantity_features(
        "She gives 3 candies to each child. There are 8 children."
    )
    assert feats["asks_rate_or_unit"] is True


def test_asks_rate_or_unit_every_keyword() -> None:
    feats = extract_target_quantity_features(
        "The bus stops every 15 minutes. How many stops in 2 hours?"
    )
    assert feats["asks_rate_or_unit"] is True


def test_asks_rate_or_unit_absent() -> None:
    feats = extract_target_quantity_features(
        "She bought 5 apples and 3 oranges. What is the total?"
    )
    assert feats["asks_rate_or_unit"] is False


# ---------------------------------------------------------------------------
# A. Target-type cues — asks_money
# ---------------------------------------------------------------------------


def test_asks_money_dollar_symbol() -> None:
    feats = extract_target_quantity_features("She spent $25 on groceries.")
    assert feats["asks_money"] is True


def test_asks_money_dollars_word() -> None:
    feats = extract_target_quantity_features("He earned 100 dollars this week.")
    assert feats["asks_money"] is True


def test_asks_money_cents_word() -> None:
    feats = extract_target_quantity_features("The candy costs 50 cents.")
    assert feats["asks_money"] is True


def test_asks_money_euro_symbol() -> None:
    feats = extract_target_quantity_features("The ticket costs €30.")
    assert feats["asks_money"] is True


def test_asks_money_absent() -> None:
    feats = extract_target_quantity_features("She ran 5 miles on Monday.")
    assert feats["asks_money"] is False


# ---------------------------------------------------------------------------
# A. Target-type cues — asks_time
# ---------------------------------------------------------------------------


def test_asks_time_minutes_keyword() -> None:
    feats = extract_target_quantity_features(
        "He studied for 45 minutes. How many minutes did he study?"
    )
    assert feats["asks_time"] is True


def test_asks_time_hours_keyword() -> None:
    feats = extract_target_quantity_features("She worked 8 hours a day for 5 days.")
    assert feats["asks_time"] is True


def test_asks_time_days_keyword() -> None:
    feats = extract_target_quantity_features("The project takes 14 days to complete.")
    assert feats["asks_time"] is True


def test_asks_time_weeks_keyword() -> None:
    feats = extract_target_quantity_features("He saved money for 4 weeks.")
    assert feats["asks_time"] is True


def test_asks_time_absent() -> None:
    feats = extract_target_quantity_features("She bought 5 apples. How many total?")
    assert feats["asks_time"] is False


# ---------------------------------------------------------------------------
# B. Wording-trap signals — has_subtraction_trap_verb
# ---------------------------------------------------------------------------


def test_subtraction_trap_spent() -> None:
    feats = extract_target_quantity_features("She spent $10 on lunch.")
    assert feats["has_subtraction_trap_verb"] is True


def test_subtraction_trap_lost() -> None:
    feats = extract_target_quantity_features("He lost 3 marbles at school.")
    assert feats["has_subtraction_trap_verb"] is True


def test_subtraction_trap_gave_away() -> None:
    feats = extract_target_quantity_features("She gave away 5 cookies to her friends.")
    assert feats["has_subtraction_trap_verb"] is True


def test_subtraction_trap_sold() -> None:
    feats = extract_target_quantity_features("The farmer sold 20 apples at the market.")
    assert feats["has_subtraction_trap_verb"] is True


def test_subtraction_trap_used() -> None:
    feats = extract_target_quantity_features("He used 4 of his pencils.")
    assert feats["has_subtraction_trap_verb"] is True


def test_subtraction_trap_absent() -> None:
    feats = extract_target_quantity_features(
        "Janet has 3 apples and buys 4 more. How many does she have in total?"
    )
    assert feats["has_subtraction_trap_verb"] is False


def test_subtraction_trap_case_insensitive() -> None:
    feats = extract_target_quantity_features("She SPENT $5 at the store.")
    assert feats["has_subtraction_trap_verb"] is True


# ---------------------------------------------------------------------------
# B. Wording-trap signals — has_addition_trap_structure
# ---------------------------------------------------------------------------


def test_addition_trap_also_keyword() -> None:
    feats = extract_target_quantity_features(
        "She has 5 apples. She also has 3 oranges."
    )
    assert feats["has_addition_trap_structure"] is True


def test_addition_trap_then_keyword() -> None:
    feats = extract_target_quantity_features(
        "He bought 4 shirts. Then he bought 2 more."
    )
    assert feats["has_addition_trap_structure"] is True


def test_addition_trap_together_keyword() -> None:
    feats = extract_target_quantity_features(
        "John and Mary worked together on the project."
    )
    assert feats["has_addition_trap_structure"] is True


def test_addition_trap_absent() -> None:
    feats = extract_target_quantity_features(
        "She ran 5 miles and rested. How far did she run?"
    )
    assert feats["has_addition_trap_structure"] is False


# ---------------------------------------------------------------------------
# B. Wording-trap signals — has_multi_operation_hint
# ---------------------------------------------------------------------------


def test_multi_operation_hint_two_verbs() -> None:
    feats = extract_target_quantity_features(
        "She bought 10 apples and sold 4 of them."
    )
    assert feats["has_multi_operation_hint"] is True


def test_multi_operation_hint_three_verbs() -> None:
    feats = extract_target_quantity_features(
        "He earned $50, spent $20, and saved the rest."
    )
    assert feats["has_multi_operation_hint"] is True


def test_multi_operation_hint_single_verb() -> None:
    feats = extract_target_quantity_features(
        "She bought 10 apples. How many does she have?"
    )
    assert feats["has_multi_operation_hint"] is False


def test_multi_operation_hint_no_verbs() -> None:
    feats = extract_target_quantity_features("3 plus 4 equals what?")
    assert feats["has_multi_operation_hint"] is False


# ---------------------------------------------------------------------------
# C. Answer-risk signals — likely_intermediate_quantity_ask
# ---------------------------------------------------------------------------


def test_likely_intermediate_quantity_ask_triggered() -> None:
    # ≥3 sentences, ≥3 numbers, no total/remaining anchor
    question = (
        "Tom had 8 apples. "
        "He gave 3 to his sister. "
        "His sister then gave 2 to a friend. "
        "How many does his sister now have?"
    )
    feats = extract_target_quantity_features(question)
    assert feats["likely_intermediate_quantity_ask"] is True


def test_likely_intermediate_quantity_ask_suppressed_by_total_cue() -> None:
    # Would otherwise trigger but "total" cue suppresses it
    question = (
        "Tom had 8 apples. "
        "He received 3 more. "
        "Then he got 2 more. "
        "What is the total?"
    )
    feats = extract_target_quantity_features(question)
    assert feats["likely_intermediate_quantity_ask"] is False


def test_likely_intermediate_quantity_ask_suppressed_by_remaining_cue() -> None:
    question = (
        "Tom had 10 apples. "
        "He gave away 3 yesterday. "
        "He gave away 2 today. "
        "How many are remaining?"
    )
    feats = extract_target_quantity_features(question)
    assert feats["likely_intermediate_quantity_ask"] is False


def test_likely_intermediate_quantity_ask_too_few_sentences() -> None:
    # Only 2 sentences — below threshold of 3
    feats = extract_target_quantity_features(
        "She has 4 apples and 3 oranges. How many fruits does she have?"
    )
    assert feats["likely_intermediate_quantity_ask"] is False


def test_likely_intermediate_quantity_ask_too_few_numbers() -> None:
    # Only 2 numbers — below threshold of 3
    question = (
        "Tom went to the store. "
        "He spent $5 on apples. "
        "He walked home. "
        "How much did he spend in total?"
    )
    feats = extract_target_quantity_features(question)
    # "total" anchor also suppresses this
    assert feats["likely_intermediate_quantity_ask"] is False


# ---------------------------------------------------------------------------
# C. Answer-risk signals — potential_answer_echo_risk
# ---------------------------------------------------------------------------


def test_potential_answer_echo_risk_triggered() -> None:
    # ≥4 numbers AND short final sentence
    question = (
        "A shop sells apples for $2, oranges for $3, bananas for $1, "
        "and grapes for $4. "
        "What does an apple cost?"
    )
    feats = extract_target_quantity_features(question)
    assert feats["potential_answer_echo_risk"] is True


def test_potential_answer_echo_risk_few_numbers() -> None:
    # Only 2 numbers — below threshold of 4
    feats = extract_target_quantity_features(
        "She has 5 apples. She gives 2 away. How many are left?"
    )
    assert feats["potential_answer_echo_risk"] is False


def test_potential_answer_echo_risk_long_final_sentence() -> None:
    # 4+ numbers but final sentence is long (> 12 tokens)
    question = (
        "Items cost $2, $3, $4, and $5. "
        "If she buys one of each item and pays with a $20 bill, "
        "how much change does she receive in total from the cashier?"
    )
    feats = extract_target_quantity_features(question)
    assert feats["potential_answer_echo_risk"] is False


# ---------------------------------------------------------------------------
# Integration: composable with extract_query_features
# ---------------------------------------------------------------------------


def test_composable_with_extract_query_features() -> None:
    """Merging both feature dicts should not produce key conflicts."""
    from src.features.precompute_features import extract_query_features

    question = "She spent $5 on 3 apples and has $2 remaining."
    base = extract_query_features(question)
    tq = extract_target_quantity_features(question)

    # No key collisions
    assert set(base.keys()).isdisjoint(set(tq.keys())), (
        "Key collision between extract_query_features and "
        "extract_target_quantity_features"
    )

    merged = {**base, **tq}
    assert len(merged) == len(base) + len(tq)


def test_importable_from_features_package() -> None:
    """extract_target_quantity_features must be importable from src.features."""
    from src.features import extract_target_quantity_features as fn

    feats = fn("She earned $10 per day for 5 days.")
    assert feats["asks_money"] is True
    assert feats["asks_rate_or_unit"] is True


# ---------------------------------------------------------------------------
# Realistic GSM8K-style examples
# ---------------------------------------------------------------------------


def test_gsm8k_remaining_style_problem() -> None:
    """Typical 'remaining' problem: Janet buys apples, eats some, asks remainder."""
    question = (
        "Janet had 20 apples. She gave 5 to her friend and ate 3 herself. "
        "How many apples does she have left?"
    )
    feats = extract_target_quantity_features(question)
    assert feats["asks_remaining_or_left"] is True
    assert feats["has_subtraction_trap_verb"] is True   # "gave"
    assert feats["has_multi_operation_hint"] is True    # gave + ate = 2 verbs


def test_gsm8k_total_earnings_problem() -> None:
    """Total-earnings problem: rate × days."""
    question = (
        "Maria earns $15 per hour. She works 8 hours a day for 5 days. "
        "What are her total earnings?"
    )
    feats = extract_target_quantity_features(question)
    assert feats["asks_total"] is True
    assert feats["asks_rate_or_unit"] is True
    assert feats["asks_money"] is True
    assert feats["asks_time"] is True


def test_gsm8k_rate_problem() -> None:
    """Rate problem: how many widgets per hour."""
    question = (
        "A machine produces 120 widgets in 4 hours. "
        "How many widgets does it produce per hour?"
    )
    feats = extract_target_quantity_features(question)
    assert feats["asks_rate_or_unit"] is True
    assert feats["asks_remaining_or_left"] is False
