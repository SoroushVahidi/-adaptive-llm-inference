from __future__ import annotations

from src.features.calibration_features import extract_calibration_features
from src.features.selective_prediction_features import extract_selective_prediction_features
from src.features.self_verification_features import extract_self_verification_features
from src.features.step_verification_features import extract_step_verification_features
from src.features.unified_error_signal import compute_unified_error_signal


def test_self_verification_features_detects_conflict() -> None:
    out = extract_self_verification_features(
        question_text="What is 5+5?",
        reasoning_text="I think it is 10. But actually maybe 11.",
        parsed_answer="11",
    )
    assert out["reasoning_contains_backtracking_or_conflict"] is True


def test_selective_prediction_features_has_confidence_proxy() -> None:
    out = extract_selective_prediction_features("Final answer: 42", parsed_answer="42")
    assert 0.0 <= out["confidence_proxy_score"] <= 1.0


def test_calibration_features_outputs_bin() -> None:
    out = extract_calibration_features("Final answer: 12", parsed_answer="12")
    assert out["calibration_bin"] in {"high_confidence", "medium_confidence", "low_confidence"}


def test_step_verification_features_outputs_step_count() -> None:
    out = extract_step_verification_features(
        question_text="A has 10 and gives away 3, then buys 2.",
        reasoning_text="10 - 3 = 7. Then 7 + 2 = 9. Final answer: 9",
    )
    assert out["number_of_steps_detected"] >= 1


def test_unified_error_signal_scores_in_range() -> None:
    out = compute_unified_error_signal(
        question_text="A bus has 40 seats. 26 are occupied. How many are left?",
        reasoning_text="Final answer: 26",
        parsed_answer="26",
    )
    assert 0.0 <= out["unified_error_score"] <= 1.0
    assert 0.0 <= out["unified_confidence_score"] <= 1.0
