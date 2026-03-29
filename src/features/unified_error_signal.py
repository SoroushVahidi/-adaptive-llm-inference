"""Unified interpretable error/confidence aggregation across feature families."""

from __future__ import annotations

from typing import Any

from src.features.calibration_features import extract_calibration_features
from src.features.constraint_violation_features import extract_constraint_violation_features
from src.features.number_role_features import compute_calibrated_role_decision
from src.features.selective_prediction_features import extract_selective_prediction_features
from src.features.self_verification_features import extract_self_verification_features
from src.features.step_verification_features import extract_step_verification_features
from src.features.target_quantity_features import extract_target_quantity_features


def compute_unified_error_signal(
    question_text: str,
    reasoning_text: str,
    parsed_answer: str | None = None,
    shallow_pass_answers: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate interpretable signals into unified error/confidence scores."""
    role = compute_calibrated_role_decision(
        question_text=question_text,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    constraint = extract_constraint_violation_features(
        question_text=question_text,
        reasoning_output=reasoning_text,
        predicted_answer=parsed_answer,
    )
    target = extract_target_quantity_features(question_text)
    self_verify = extract_self_verification_features(
        question_text=question_text,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    selective = extract_selective_prediction_features(
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
        shallow_pass_answers=shallow_pass_answers,
    )
    calibration = extract_calibration_features(
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    step = extract_step_verification_features(
        question_text=question_text,
        reasoning_text=reasoning_text,
    )

    role_error = min(1.0, 0.2 * role["role_warning_score"] + 0.3 * role["role_strong_error_score"])
    constraint_error = min(
        1.0,
        0.25 * int(constraint.get("target_quantity_mismatch_suspected", False))
        + 0.2 * int(constraint.get("constraint_word_conflict_suspected", False))
        + 0.2 * int(constraint.get("answer_type_mismatch_suspected", False)),
    )
    target_error = 0.15 * int(target.get("likely_intermediate_quantity_ask", False)) + 0.1 * int(
        target.get("potential_answer_echo_risk", False)
    )
    self_error = 1.0 - float(self_verify["self_verification_score"])
    selective_uncertainty = 1.0 - float(selective["confidence_proxy_score"])
    calibration_uncertainty = 1.0 - float(calibration["predicted_answer_format_confidence"])
    step_error = 1.0 - float(step["step_consistency_score"])

    unified_error_score = (
        0.24 * role_error
        + 0.19 * constraint_error
        + 0.08 * target_error
        + 0.14 * self_error
        + 0.12 * selective_uncertainty
        + 0.11 * calibration_uncertainty
        + 0.12 * step_error
    )
    unified_error_score = max(0.0, min(1.0, unified_error_score))

    unified_confidence_score = (
        0.3 * selective["confidence_proxy_score"]
        + 0.25 * calibration["predicted_answer_format_confidence"]
        + 0.25 * self_verify["self_verification_score"]
        + 0.2 * step["step_consistency_score"]
    )
    unified_confidence_score = max(0.0, min(1.0, unified_confidence_score))

    return {
        "unified_error_score": unified_error_score,
        "unified_confidence_score": unified_confidence_score,
        "signal_breakdown": {
            "role": role,
            "constraint": constraint,
            "target": target,
            "self_verification": self_verify,
            "selective_prediction": selective,
            "calibration": calibration,
            "step_verification": step,
            "component_error": {
                "role_error": role_error,
                "constraint_error": constraint_error,
                "target_error": target_error,
                "self_error": self_error,
                "selective_uncertainty": selective_uncertainty,
                "calibration_uncertainty": calibration_uncertainty,
                "step_error": step_error,
            },
        },
    }
