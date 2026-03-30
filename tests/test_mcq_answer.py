from src.utils.mcq_answer import extract_mcq_letter, gold_letter_from_solution, normalize_mcq_letter


def test_normalize_mcq_letter() -> None:
    assert normalize_mcq_letter("d") == "D"
    assert normalize_mcq_letter("x") == ""


def test_extract_boxed() -> None:
    assert extract_mcq_letter(r"Thus \boxed{D}") == "D"
    assert extract_mcq_letter(r"Answer: \boxed{\mathrm{B}}") == "B"


def test_gold_from_solution() -> None:
    assert gold_letter_from_solution(r"\boxed{C}") == "C"
