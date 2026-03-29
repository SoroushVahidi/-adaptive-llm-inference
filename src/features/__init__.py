"""Feature extraction helpers."""

from src.features.calibration_features import extract_calibration_features
from src.features.constraint_violation_features import (
    ConstraintFeatureConfig,
    extract_constraint_violation_features,
    summarize_constraint_signal_firing,
)
from src.features.number_role_features import (
    assign_number_roles,
    compute_calibrated_role_decision,
    compute_role_coverage_features,
    extract_problem_numbers,
)
from src.features.precompute_features import extract_first_pass_features, extract_query_features
from src.features.selective_prediction_features import extract_selective_prediction_features
from src.features.self_verification_features import extract_self_verification_features
from src.features.step_verification_features import extract_step_verification_features
from src.features.target_quantity_features import extract_target_quantity_features
from src.features.unified_error_signal import compute_unified_error_signal

__all__ = [
    "ConstraintFeatureConfig",
    "extract_constraint_violation_features",
    "summarize_constraint_signal_firing",
    "extract_query_features",
    "extract_first_pass_features",
    "extract_target_quantity_features",
    "extract_problem_numbers",
    "assign_number_roles",
    "compute_role_coverage_features",
    "compute_calibrated_role_decision",
    "extract_self_verification_features",
    "extract_selective_prediction_features",
    "extract_calibration_features",
    "extract_step_verification_features",
    "compute_unified_error_signal",
]
