"""Unit tests for src/features/precompute_features.py."""

import pytest

from src.features.precompute_features import extract_first_pass_features, extract_query_features

# ---------------------------------------------------------------------------
# extract_query_features — basic structure
# ---------------------------------------------------------------------------


def test_returns_all_expected_keys() -> None:
    feats = extract_query_features("Janet has 3 apples.")
    expected_keys = {
        "question_length_chars",
        "question_length_tokens_approx",
        "num_numeric_mentions",
        "num_sentences_approx",
        "has_multi_step_cue",
        "has_equation_like_pattern",
        "has_percent_symbol",
        "has_fraction_pattern",
        "has_currency_symbol",
        "max_numeric_value_approx",
        "min_numeric_value_approx",
        "numeric_range_approx",
        "repeated_number_flag",
    }
    assert set(feats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Numeric counting
# ---------------------------------------------------------------------------


def test_numeric_counting_single() -> None:
    feats = extract_query_features("There are 5 dogs.")
    assert feats["num_numeric_mentions"] == 1


def test_numeric_counting_multiple() -> None:
    feats = extract_query_features("She bought 3 apples, 7 oranges, and 2 bananas.")
    assert feats["num_numeric_mentions"] == 3


def test_numeric_counting_with_decimals() -> None:
    feats = extract_query_features("The price is 3.50 and discount is 0.25.")
    assert feats["num_numeric_mentions"] == 2


def test_numeric_counting_with_commas() -> None:
    # 1,000 should count as one number
    feats = extract_query_features("He earned $1,000 last year.")
    assert feats["num_numeric_mentions"] == 1


def test_max_min_numeric_values() -> None:
    feats = extract_query_features("Values are 10, 50, and 5.")
    assert feats["max_numeric_value_approx"] == pytest.approx(50.0)
    assert feats["min_numeric_value_approx"] == pytest.approx(5.0)
    assert feats["numeric_range_approx"] == pytest.approx(45.0)


def test_numeric_range_zero_for_single_number() -> None:
    feats = extract_query_features("There is only 42 items.")
    assert feats["numeric_range_approx"] == pytest.approx(0.0)


def test_numeric_range_zero_for_no_numbers() -> None:
    feats = extract_query_features("No numbers here.")
    assert feats["numeric_range_approx"] == pytest.approx(0.0)
    assert feats["max_numeric_value_approx"] == pytest.approx(0.0)
    assert feats["min_numeric_value_approx"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Repeated number flag
# ---------------------------------------------------------------------------


def test_repeated_number_flag_true() -> None:
    feats = extract_query_features("She ran 5 miles on Monday and 5 miles on Friday.")
    assert feats["repeated_number_flag"] is True


def test_repeated_number_flag_false() -> None:
    feats = extract_query_features("She ran 5 miles on Monday and 7 miles on Friday.")
    assert feats["repeated_number_flag"] is False


# ---------------------------------------------------------------------------
# Multi-step cue detection
# ---------------------------------------------------------------------------


def test_multi_step_cue_total() -> None:
    feats = extract_query_features("What is the total cost?")
    assert feats["has_multi_step_cue"] is True


def test_multi_step_cue_remaining() -> None:
    feats = extract_query_features("How many apples are remaining?")
    assert feats["has_multi_step_cue"] is True


def test_multi_step_cue_case_insensitive() -> None:
    feats = extract_query_features("Find the AVERAGE of these numbers.")
    assert feats["has_multi_step_cue"] is True


def test_multi_step_cue_absent() -> None:
    feats = extract_query_features("What color is the sky?")
    assert feats["has_multi_step_cue"] is False


def test_multi_step_cue_percent_keyword() -> None:
    feats = extract_query_features("Express the answer as a percent.")
    assert feats["has_multi_step_cue"] is True


# ---------------------------------------------------------------------------
# Fraction / percent / currency detection
# ---------------------------------------------------------------------------


def test_fraction_pattern_detected() -> None:
    feats = extract_query_features("He ate 3/4 of the pizza.")
    assert feats["has_fraction_pattern"] is True


def test_fraction_pattern_absent() -> None:
    feats = extract_query_features("He ate half of the pizza.")
    assert feats["has_fraction_pattern"] is False


def test_percent_symbol_detected() -> None:
    feats = extract_query_features("The discount is 20%.")
    assert feats["has_percent_symbol"] is True


def test_percent_symbol_absent() -> None:
    feats = extract_query_features("The discount is twenty percent.")
    assert feats["has_percent_symbol"] is False


def test_currency_symbol_dollar() -> None:
    feats = extract_query_features("She spent $50 on groceries.")
    assert feats["has_currency_symbol"] is True


def test_currency_symbol_euro() -> None:
    feats = extract_query_features("The item costs €30.")
    assert feats["has_currency_symbol"] is True


def test_currency_symbol_absent() -> None:
    feats = extract_query_features("The item costs 30 dollars.")
    assert feats["has_currency_symbol"] is False


# ---------------------------------------------------------------------------
# Equation-like pattern
# ---------------------------------------------------------------------------


def test_equation_like_pattern_detected() -> None:
    feats = extract_query_features("We know that 3 + 4 = 7.")
    assert feats["has_equation_like_pattern"] is True


def test_equation_like_pattern_absent() -> None:
    feats = extract_query_features("Add the numbers together.")
    assert feats["has_equation_like_pattern"] is False


# ---------------------------------------------------------------------------
# Length features
# ---------------------------------------------------------------------------


def test_question_length_chars() -> None:
    text = "Hello world"
    feats = extract_query_features(text)
    assert feats["question_length_chars"] == len(text)


def test_question_length_tokens_approx() -> None:
    text = "One two three four five"
    feats = extract_query_features(text)
    assert feats["question_length_tokens_approx"] == 5


def test_num_sentences_approx() -> None:
    text = "First sentence. Second sentence! Third sentence?"
    feats = extract_query_features(text)
    assert feats["num_sentences_approx"] == 3


# ---------------------------------------------------------------------------
# extract_first_pass_features — structure
# ---------------------------------------------------------------------------


def test_first_pass_returns_all_expected_keys() -> None:
    feats = extract_first_pass_features("Some question.", "The answer is 42.")
    expected_keys = {
        "first_pass_parse_success",
        "first_pass_output_length",
        "first_pass_has_final_answer_cue",
        "first_pass_has_uncertainty_phrase",
        "first_pass_num_numeric_mentions",
        "first_pass_empty_or_malformed_flag",
    }
    assert set(feats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Empty / malformed first-pass output
# ---------------------------------------------------------------------------


def test_first_pass_empty_string() -> None:
    feats = extract_first_pass_features("Some question.", "")
    assert feats["first_pass_empty_or_malformed_flag"] is True
    assert feats["first_pass_parse_success"] is False


def test_first_pass_whitespace_only() -> None:
    feats = extract_first_pass_features("Some question.", "   ")
    assert feats["first_pass_empty_or_malformed_flag"] is True


def test_first_pass_very_short_output() -> None:
    feats = extract_first_pass_features("Some question.", "ok")
    assert feats["first_pass_empty_or_malformed_flag"] is True


def test_first_pass_normal_output_not_malformed() -> None:
    feats = extract_first_pass_features("Some question.", "The answer is 42.")
    assert feats["first_pass_empty_or_malformed_flag"] is False


# ---------------------------------------------------------------------------
# Final-answer cue detection
# ---------------------------------------------------------------------------


def test_first_pass_final_answer_cue_detected() -> None:
    feats = extract_first_pass_features(
        "What is 2 + 2?",
        "We compute step by step. Final answer: 4.",
    )
    assert feats["first_pass_has_final_answer_cue"] is True


def test_first_pass_therefore_cue_detected() -> None:
    feats = extract_first_pass_features(
        "What is 2 + 2?",
        "Therefore, the result is 4.",
    )
    assert feats["first_pass_has_final_answer_cue"] is True


def test_first_pass_no_final_answer_cue() -> None:
    feats = extract_first_pass_features("Some question.", "I think it might be 42.")
    assert feats["first_pass_has_final_answer_cue"] is False


# ---------------------------------------------------------------------------
# Uncertainty phrase detection
# ---------------------------------------------------------------------------


def test_first_pass_uncertainty_phrase_detected() -> None:
    feats = extract_first_pass_features(
        "Some question.",
        "I'm not sure about this, but it might be 7.",
    )
    assert feats["first_pass_has_uncertainty_phrase"] is True


def test_first_pass_no_uncertainty_phrase() -> None:
    feats = extract_first_pass_features(
        "Some question.",
        "The answer is clearly 42.",
    )
    assert feats["first_pass_has_uncertainty_phrase"] is False


# ---------------------------------------------------------------------------
# Parse success with parsed_answer argument
# ---------------------------------------------------------------------------


def test_first_pass_parse_success_with_parsed_answer() -> None:
    feats = extract_first_pass_features("q", "output text", parsed_answer="42")
    assert feats["first_pass_parse_success"] is True


def test_first_pass_parse_failure_with_empty_parsed_answer() -> None:
    feats = extract_first_pass_features("q", "no numbers here", parsed_answer="")
    assert feats["first_pass_parse_success"] is False


def test_first_pass_parse_success_inferred_from_numbers() -> None:
    # No parsed_answer supplied; success is inferred from numeric tokens
    feats = extract_first_pass_features("q", "The result is 99.")
    assert feats["first_pass_parse_success"] is True


def test_first_pass_parse_failure_inferred_no_numbers() -> None:
    feats = extract_first_pass_features("q", "I have no idea.")
    assert feats["first_pass_parse_success"] is False


# ---------------------------------------------------------------------------
# Numeric mentions in first-pass output
# ---------------------------------------------------------------------------


def test_first_pass_num_numeric_mentions() -> None:
    # "Step 1: 5. Step 2: 10. Total: 15." — regex finds 1, 5, 2, 10, 15 = 5 tokens
    feats = extract_first_pass_features("q", "Step 1: 5. Step 2: 10. Total: 15.")
    assert feats["first_pass_num_numeric_mentions"] == 5


def test_first_pass_output_length() -> None:
    output = "The answer is 42."
    feats = extract_first_pass_features("q", output)
    assert feats["first_pass_output_length"] == len(output)
