"""Feature extraction helpers."""

from src.features.constraint_violation_features import (
    ConstraintFeatureConfig,
    extract_constraint_violation_features,
    summarize_constraint_signal_firing,
)
from src.features.number_role_lexicon import (
    classify_local_role_cue,
    extract_number_word_matches,
    get_role_cues,
    normalize_number_word,
)
from src.features.precompute_features import extract_first_pass_features, extract_query_features
from src.features.target_quantity_features import extract_target_quantity_features

__all__ = [
    "ConstraintFeatureConfig",
    "extract_constraint_violation_features",
    "summarize_constraint_signal_firing",
    "extract_query_features",
    "extract_first_pass_features",
    "extract_target_quantity_features",
    "normalize_number_word",
    "extract_number_word_matches",
    "get_role_cues",
    "classify_local_role_cue",
]
