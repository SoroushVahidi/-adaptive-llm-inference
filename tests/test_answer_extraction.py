from src.utils.answer_extraction import extract_numeric_answer


def test_simple_number():
    assert extract_numeric_answer("The answer is 42") == "42"


def test_number_with_commas():
    assert extract_numeric_answer("Total: 1,234") == "1234"


def test_negative_number():
    assert extract_numeric_answer("Result is -7") == "-7"


def test_decimal():
    assert extract_numeric_answer("The answer is 3.14") == "3.14"


def test_last_number_wins():
    assert extract_numeric_answer("Step 1: 5, Step 2: 10, Final: 15") == "15"


def test_no_number():
    assert extract_numeric_answer("no numbers here") == "no numbers here"
