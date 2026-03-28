from src.utils.answer_extraction import (
    extract_math_answer,
    extract_numeric_answer,
    normalize_math_answer,
)


def test_direct_final_numeric_answer() -> None:
    assert extract_numeric_answer("Answer the following question. Final answer: 42") == "42"


def test_reasoning_with_intermediate_numbers_prefers_final_answer_marker() -> None:
    text = (
        "Step 1: 16 eggs.\n"
        "Step 2: 3 + 4 = 7 eggs used.\n"
        "Step 3: 16 - 7 = 9 eggs sold.\n"
        "Final answer: 18"
    )
    assert extract_numeric_answer(text) == "18"


def test_decimal_normalization() -> None:
    assert extract_numeric_answer("The original price was 26.00.") == "26"
    assert extract_numeric_answer("Final answer: 26.0") == "26"


def test_comma_separated_number() -> None:
    assert extract_numeric_answer("The total is $57,500.") == "57500"


def test_boxed_answer_is_supported() -> None:
    assert extract_numeric_answer(r"We compute carefully and get \boxed{104}.") == "104"


def test_extract_math_answer_prefers_boxed_symbolic_answer() -> None:
    text = r"We simplify carefully and conclude \boxed{\left( 3, \frac{1}{2} \right)}."
    assert extract_math_answer(text) == r"(3,\frac{1}{2})"


def test_normalize_math_answer_removes_lightweight_latex_wrappers() -> None:
    assert normalize_math_answer(r"\left( 3, \frac{2}{4} \right).") == r"(3,\frac{2}{4})"


def test_reasoning_output_does_not_latch_onto_intermediate_prefix_numbers() -> None:
    text = (
        "1. Determine total eggs laid: 16.\n"
        "2. Determine eggs eaten: 3.\n"
        "3. Determine eggs baked with: 4.\n"
        "Therefore, Janet makes 18 dollars.\n"
        "Answer: 18"
    )
    assert extract_numeric_answer(text) == "18"


def test_final_lines_are_preferred_when_no_explicit_marker_exists() -> None:
    text = (
        "The distance for one day is 180 meters.\n"
        "He runs 3 days per week.\n"
        "So the total weekly distance is 540 meters.\n"
        "540"
    )
    assert extract_numeric_answer(text) == "540"


def test_negative_number() -> None:
    assert extract_numeric_answer("Result is -7") == "-7"


def test_decimal_value_is_preserved() -> None:
    assert extract_numeric_answer("The answer is 3.14") == "3.14"


def test_extraction_fails_cleanly_when_no_number_exists() -> None:
    assert extract_numeric_answer("I am not sure.") == ""
