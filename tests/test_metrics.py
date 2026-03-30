from src.baselines.base import BaselineResult
from src.evaluation.metrics import compute_accuracy, exact_match


def test_exact_match_identical():
    assert exact_match("42", "42")


def test_exact_match_case_insensitive():
    assert exact_match("ABC", "abc")


def test_exact_match_whitespace():
    assert exact_match(" 42 ", "42")


def test_exact_match_different():
    assert not exact_match("41", "42")


def _make_result(correct: bool, samples: int = 1) -> BaselineResult:
    return BaselineResult(
        query_id="q",
        question="q?",
        candidates=["a"],
        final_answer="42" if correct else "0",
        ground_truth="42",
        correct=correct,
        samples_used=samples,
        self_consistency_ambiguous=False,
        self_consistency_tie=False,
        metadata={},
    )


def test_compute_accuracy_all_correct():
    results = [_make_result(True) for _ in range(5)]
    stats = compute_accuracy(results)
    assert stats["accuracy"] == 1.0
    assert stats["total_queries"] == 5


def test_compute_accuracy_half_correct():
    results = [_make_result(True), _make_result(False)]
    stats = compute_accuracy(results)
    assert stats["accuracy"] == 0.5


def test_compute_accuracy_empty():
    stats = compute_accuracy([])
    assert stats["accuracy"] == 0.0
