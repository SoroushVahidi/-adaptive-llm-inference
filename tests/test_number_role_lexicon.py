"""Unit tests for src/features/number_role_lexicon.py."""

from __future__ import annotations

import pytest

from src.features.number_role_lexicon import (
    classify_local_role_cue,
    extract_number_word_matches,
    get_role_cues,
    normalize_number_word,
)

# ---------------------------------------------------------------------------
# normalize_number_word — structure
# ---------------------------------------------------------------------------


def test_normalize_returns_expected_keys() -> None:
    result = normalize_number_word("three")
    assert result is not None
    assert set(result.keys()) == {"surface_text", "normalized_value", "source_type"}


def test_normalize_returns_none_for_unknown_token() -> None:
    assert normalize_number_word("elephant") is None


def test_normalize_returns_none_for_empty_string() -> None:
    assert normalize_number_word("") is None


def test_normalize_returns_none_for_numeric_digit() -> None:
    assert normalize_number_word("3") is None


# ---------------------------------------------------------------------------
# normalize_number_word — cardinal number_words (one … twelve)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,expected_value",
    [
        ("one", 1.0),
        ("two", 2.0),
        ("three", 3.0),
        ("four", 4.0),
        ("five", 5.0),
        ("six", 6.0),
        ("seven", 7.0),
        ("eight", 8.0),
        ("nine", 9.0),
        ("ten", 10.0),
        ("eleven", 11.0),
        ("twelve", 12.0),
    ],
)
def test_normalize_cardinal_number_words(word: str, expected_value: float) -> None:
    result = normalize_number_word(word)
    assert result is not None
    assert result["normalized_value"] == expected_value
    assert result["source_type"] == "number_word"
    assert result["surface_text"] == word


def test_normalize_case_insensitive_cardinal() -> None:
    result = normalize_number_word("THREE")
    assert result is not None
    assert result["normalized_value"] == 3.0
    assert result["surface_text"] == "three"


# ---------------------------------------------------------------------------
# normalize_number_word — multiplicative_word
# ---------------------------------------------------------------------------


def test_normalize_double() -> None:
    result = normalize_number_word("double")
    assert result is not None
    assert result["normalized_value"] == 2.0
    assert result["source_type"] == "multiplicative_word"


def test_normalize_twice() -> None:
    result = normalize_number_word("twice")
    assert result is not None
    assert result["normalized_value"] == 2.0
    assert result["source_type"] == "multiplicative_word"


def test_normalize_triple() -> None:
    result = normalize_number_word("triple")
    assert result is not None
    assert result["normalized_value"] == 3.0
    assert result["source_type"] == "multiplicative_word"


def test_normalize_case_insensitive_multiplicative() -> None:
    result = normalize_number_word("TWICE")
    assert result is not None
    assert result["normalized_value"] == 2.0
    assert result["source_type"] == "multiplicative_word"


# ---------------------------------------------------------------------------
# normalize_number_word — fraction_word
# ---------------------------------------------------------------------------


def test_normalize_half() -> None:
    result = normalize_number_word("half")
    assert result is not None
    assert result["normalized_value"] == 0.5
    assert result["source_type"] == "fraction_word"


# ---------------------------------------------------------------------------
# normalize_number_word — quantity_word
# ---------------------------------------------------------------------------


def test_normalize_dozen() -> None:
    result = normalize_number_word("dozen")
    assert result is not None
    assert result["normalized_value"] == 12.0
    assert result["source_type"] == "quantity_word"


# ---------------------------------------------------------------------------
# extract_number_word_matches — structure and empty cases
# ---------------------------------------------------------------------------


def test_extract_empty_string_returns_empty_list() -> None:
    assert extract_number_word_matches("") == []


def test_extract_no_number_words_returns_empty_list() -> None:
    assert extract_number_word_matches("The sky is blue.") == []


def test_extract_returns_list_of_dicts() -> None:
    results = extract_number_word_matches("She has three apples.")
    assert isinstance(results, list)
    for item in results:
        assert set(item.keys()) == {"surface_text", "normalized_value", "source_type"}


# ---------------------------------------------------------------------------
# extract_number_word_matches — single matches
# ---------------------------------------------------------------------------


def test_extract_single_cardinal() -> None:
    results = extract_number_word_matches("She bought five pencils.")
    assert len(results) == 1
    assert results[0]["surface_text"] == "five"
    assert results[0]["normalized_value"] == 5.0
    assert results[0]["source_type"] == "number_word"


def test_extract_single_multiplicative() -> None:
    results = extract_number_word_matches("He earns double his usual salary.")
    assert len(results) == 1
    assert results[0]["surface_text"] == "double"
    assert results[0]["source_type"] == "multiplicative_word"


def test_extract_fraction_word() -> None:
    results = extract_number_word_matches("She gave half the cake to her sister.")
    assert len(results) == 1
    assert results[0]["surface_text"] == "half"
    assert results[0]["normalized_value"] == 0.5


def test_extract_quantity_word_dozen() -> None:
    results = extract_number_word_matches("He ordered a dozen eggs.")
    assert len(results) == 1
    assert results[0]["surface_text"] == "dozen"
    assert results[0]["normalized_value"] == 12.0
    assert results[0]["source_type"] == "quantity_word"


# ---------------------------------------------------------------------------
# extract_number_word_matches — multiple matches
# ---------------------------------------------------------------------------


def test_extract_multiple_cardinals() -> None:
    results = extract_number_word_matches("She has two cats and three dogs.")
    surfaces = [m["surface_text"] for m in results]
    assert surfaces == ["two", "three"]


def test_extract_mixed_types() -> None:
    results = extract_number_word_matches(
        "She has three apples and twice as many oranges."
    )
    surfaces = [m["surface_text"] for m in results]
    assert "three" in surfaces
    assert "twice" in surfaces


def test_extract_preserves_order() -> None:
    results = extract_number_word_matches("one two three four five")
    values = [m["normalized_value"] for m in results]
    assert values == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_extract_case_insensitive() -> None:
    results = extract_number_word_matches("She has THREE apples.")
    assert len(results) == 1
    assert results[0]["surface_text"] == "three"


def test_extract_does_not_match_substrings() -> None:
    # "oten" should not match "ten", "eleven" should not match in "eleventh"
    results = extract_number_word_matches("Listen to the threesome play.")
    # "three" is not a standalone word here — boundary check
    surfaces = [m["surface_text"] for m in results]
    assert "three" not in surfaces


# ---------------------------------------------------------------------------
# get_role_cues — structure
# ---------------------------------------------------------------------------


def test_get_role_cues_returns_all_keys() -> None:
    cues = get_role_cues("She spent $5.")
    assert set(cues.keys()) == {"add", "subtract", "multiply_rate", "ratio", "capacity"}


def test_get_role_cues_empty_text() -> None:
    cues = get_role_cues("")
    for v in cues.values():
        assert v == []


def test_get_role_cues_no_matches_all_empty() -> None:
    cues = get_role_cues("The sky is blue today.")
    for v in cues.values():
        assert v == []


def test_get_role_cues_values_are_lists() -> None:
    cues = get_role_cues("She spent five dollars.")
    for v in cues.values():
        assert isinstance(v, list)


# ---------------------------------------------------------------------------
# get_role_cues — add cues
# ---------------------------------------------------------------------------


def test_get_role_cues_add_received() -> None:
    cues = get_role_cues("She received a gift.")
    assert "received" in cues["add"]


def test_get_role_cues_add_gained() -> None:
    cues = get_role_cues("He gained three points.")
    assert "gained" in cues["add"]


def test_get_role_cues_add_bought() -> None:
    cues = get_role_cues("She bought five apples.")
    assert "bought" in cues["add"]


def test_get_role_cues_add_found() -> None:
    cues = get_role_cues("He found two coins on the floor.")
    assert "found" in cues["add"]


def test_get_role_cues_add_earned() -> None:
    cues = get_role_cues("She earned $20 today.")
    assert "earned" in cues["add"]


def test_get_role_cues_add_got_more() -> None:
    cues = get_role_cues("He got more apples from the store.")
    assert "got more" in cues["add"]


# ---------------------------------------------------------------------------
# get_role_cues — subtract cues
# ---------------------------------------------------------------------------


def test_get_role_cues_subtract_spent() -> None:
    cues = get_role_cues("She spent $10 on lunch.")
    assert "spent" in cues["subtract"]


def test_get_role_cues_subtract_lost() -> None:
    cues = get_role_cues("He lost 3 marbles.")
    assert "lost" in cues["subtract"]


def test_get_role_cues_subtract_gave_away() -> None:
    cues = get_role_cues("She gave away 5 cookies.")
    assert "gave away" in cues["subtract"]


def test_get_role_cues_subtract_sold() -> None:
    cues = get_role_cues("The farmer sold 20 apples.")
    assert "sold" in cues["subtract"]


def test_get_role_cues_subtract_used() -> None:
    cues = get_role_cues("He used four pencils.")
    assert "used" in cues["subtract"]


def test_get_role_cues_subtract_ate() -> None:
    cues = get_role_cues("She ate two cookies.")
    assert "ate" in cues["subtract"]


def test_get_role_cues_subtract_removed() -> None:
    cues = get_role_cues("They removed three items from the list.")
    assert "removed" in cues["subtract"]


# ---------------------------------------------------------------------------
# get_role_cues — multiply_rate cues
# ---------------------------------------------------------------------------


def test_get_role_cues_rate_each() -> None:
    cues = get_role_cues("She gives 3 candies to each child.")
    assert "each" in cues["multiply_rate"]


def test_get_role_cues_rate_every() -> None:
    cues = get_role_cues("The bus stops every 15 minutes.")
    assert "every" in cues["multiply_rate"]


def test_get_role_cues_rate_per() -> None:
    cues = get_role_cues("She earns $10 per hour.")
    assert "per" in cues["multiply_rate"]


# ---------------------------------------------------------------------------
# get_role_cues — ratio cues
# ---------------------------------------------------------------------------


def test_get_role_cues_ratio_half() -> None:
    cues = get_role_cues("She spent half her savings.")
    assert "half" in cues["ratio"]


def test_get_role_cues_ratio_double() -> None:
    cues = get_role_cues("He earns double what she makes.")
    assert "double" in cues["ratio"]


def test_get_role_cues_ratio_twice() -> None:
    cues = get_role_cues("The box holds twice as many items.")
    assert "twice" in cues["ratio"]


def test_get_role_cues_ratio_triple() -> None:
    cues = get_role_cues("The new machine is triple the speed.")
    assert "triple" in cues["ratio"]


# ---------------------------------------------------------------------------
# get_role_cues — capacity cues
# ---------------------------------------------------------------------------


def test_get_role_cues_capacity_at_least() -> None:
    cues = get_role_cues("She needs at least 3 boxes.")
    assert "at least" in cues["capacity"]


def test_get_role_cues_capacity_enough() -> None:
    cues = get_role_cues("Do we have enough seats for everyone?")
    assert "enough" in cues["capacity"]


def test_get_role_cues_capacity_trips() -> None:
    cues = get_role_cues("How many trips does it take?")
    assert "trips" in cues["capacity"]


def test_get_role_cues_capacity_buses() -> None:
    cues = get_role_cues("How many buses are needed?")
    assert "buses" in cues["capacity"]


def test_get_role_cues_capacity_boxes() -> None:
    cues = get_role_cues("How many boxes fit 10 apples each?")
    assert "boxes" in cues["capacity"]


# ---------------------------------------------------------------------------
# get_role_cues — mixed text
# ---------------------------------------------------------------------------


def test_get_role_cues_multiple_roles_in_one_sentence() -> None:
    cues = get_role_cues("She spent $5 and earned $10 each week.")
    assert "spent" in cues["subtract"]
    assert "earned" in cues["add"]
    assert "each" in cues["multiply_rate"]


def test_get_role_cues_no_cross_contamination() -> None:
    cues = get_role_cues("She spent $5.")
    assert cues["add"] == []
    assert cues["multiply_rate"] == []
    assert cues["ratio"] == []
    assert cues["capacity"] == []


# ---------------------------------------------------------------------------
# classify_local_role_cue — basic cases
# ---------------------------------------------------------------------------


def test_classify_returns_none_for_empty_string() -> None:
    assert classify_local_role_cue("") is None


def test_classify_returns_none_for_no_cues() -> None:
    assert classify_local_role_cue("the sky is blue") is None


def test_classify_subtract_dominant() -> None:
    assert classify_local_role_cue("she spent three dollars on lunch") == "subtract"


def test_classify_add_dominant() -> None:
    assert classify_local_role_cue("he earned five extra dollars") == "add"


def test_classify_multiply_rate_dominant() -> None:
    assert classify_local_role_cue("ten dollars per box") == "multiply_rate"


def test_classify_ratio_dominant() -> None:
    # Only ratio cues present — no add/subtract verbs in this window
    assert classify_local_role_cue("double the usual amount") == "ratio"


def test_classify_capacity_dominant() -> None:
    assert classify_local_role_cue("how many trips are needed") == "capacity"


def test_classify_returns_string() -> None:
    result = classify_local_role_cue("she spent $5")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# classify_local_role_cue — tie-breaking (subtract > add)
# ---------------------------------------------------------------------------


def test_classify_tie_break_subtract_over_add() -> None:
    # "gave" is subtract, "bought" is add — one each → subtract wins by priority
    result = classify_local_role_cue("she gave and bought one item")
    assert result == "subtract"


# ---------------------------------------------------------------------------
# Import from package
# ---------------------------------------------------------------------------


def test_importable_from_features_package() -> None:
    from src.features import (
        classify_local_role_cue as clf,
    )
    from src.features import (
        extract_number_word_matches as ext,
    )
    from src.features import (
        get_role_cues as grc,
    )
    from src.features import (
        normalize_number_word as norm,
    )

    assert norm("five") is not None
    assert ext("three apples") != []
    assert grc("spent $5")["subtract"] != []
    assert clf("spent three dollars") == "subtract"


# ---------------------------------------------------------------------------
# Realistic GSM8K-style examples
# ---------------------------------------------------------------------------


def test_gsm8k_remaining_problem_cues() -> None:
    question = (
        "Janet had 20 apples. She gave 5 to her friend and ate 3 herself. "
        "How many apples does she have left?"
    )
    cues = get_role_cues(question)
    assert "ate" in cues["subtract"]
    assert "gave" in cues["subtract"]


def test_gsm8k_rate_problem_number_words() -> None:
    question = "She earns double her usual pay. The base pay is $10 per hour."
    results = extract_number_word_matches(question)
    surfaces = [m["surface_text"] for m in results]
    assert "double" in surfaces
    cues = get_role_cues(question)
    assert "per" in cues["multiply_rate"]


def test_gsm8k_capacity_problem() -> None:
    question = (
        "A bus can hold 40 passengers. There are 120 students. "
        "How many trips does the bus need to make?"
    )
    cues = get_role_cues(question)
    assert "trips" in cues["capacity"]
    role = classify_local_role_cue("how many trips does the bus need")
    assert role == "capacity"


def test_gsm8k_mixed_number_words_and_cues() -> None:
    question = (
        "She received a dozen eggs and used half of them for baking. "
        "Each egg costs $0.50. How many eggs does she have left?"
    )
    results = extract_number_word_matches(question)
    surfaces = [m["surface_text"] for m in results]
    assert "dozen" in surfaces
    assert "half" in surfaces
    cues = get_role_cues(question)
    assert "received" in cues["add"]
    assert "used" in cues["subtract"]
    assert "each" in cues["multiply_rate"]
