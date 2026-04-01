"""Feature loading and extraction for the learned router.

Loads per-query features from the enriched routing CSV files that are already
committed to the repository.  No LLM calls are required; all signals come from
the cheap-route output (RG) that is already recorded.

Binary routing label
--------------------
``y = 1`` (escalate to DPR) if DPR is strictly better than RG:
    ``revise_correct == 1 AND reasoning_correct == 0``
``y = 0`` otherwise (RG is already correct, or DPR does not help).

This matches the RouteLLM / FrugalGPT labelling convention: escalate only when
the expensive action strictly dominates.

Public API
----------
- ``FEATURE_COLS``  — ordered list of feature column names.
- ``REGIME_FILES``  — mapping from regime id to enriched CSV path.
- ``load_regime_df(path)``  → ``pd.DataFrame``
- ``build_feature_matrix(df)``  → ``(X, y, feature_names)``
- ``build_training_dataset(regime_files, test_regime)``
    → ``(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------

#: All numeric features available in the enriched routing CSVs.
#: These are computed solely from the cheap-route (RG) output and question
#: text — no additional API calls are needed.
FEATURE_COLS: list[str] = [
    # Question-level structural features
    "q_question_length_chars",
    "q_question_length_tokens_approx",
    "q_num_numeric_mentions",
    "q_num_sentences_approx",
    "q_has_multi_step_cue",
    "q_has_equation_like_pattern",
    "q_has_percent_symbol",
    "q_has_fraction_pattern",
    "q_has_currency_symbol",
    "q_max_numeric_value_approx",
    "q_min_numeric_value_approx",
    "q_numeric_range_approx",
    "q_repeated_number_flag",
    # Target-quantity-oriented signals
    "tq_asks_remaining_or_left",
    "tq_asks_total",
    "tq_asks_difference",
    "tq_asks_rate_or_unit",
    "tq_asks_money",
    "tq_asks_time",
    "tq_has_subtraction_trap_verb",
    "tq_has_addition_trap_structure",
    "tq_has_multi_operation_hint",
    "tq_likely_intermediate_quantity_ask",
    "tq_potential_answer_echo_risk",
    # Answer-error / constraint-violation signals (strong escalation hints)
    "cons_answer_type_mismatch_suspected",
    "cons_target_quantity_mismatch_suspected",
    "cons_unit_mismatch_suspected",
    "cons_impossible_sign_suspected",
    "cons_integer_expected_but_noninteger_suspected",
    "cons_percent_or_ratio_mismatch_suspected",
    "cons_answer_not_mentioned_in_final_statement_suspected",
    "cons_constraint_word_conflict_suspected",
    "cons_bound_violation_suspected",
    "cons_obvious_upper_bound_exceeded_suspected",
    "cons_obvious_lower_bound_violated_suspected",
    # Explanation-warning / quality signals
    "role_warning_score",
    "role_strong_error_score",
    "unified_error_score",
    "unified_confidence_score",
    # Self-verification features
    "self_self_verification_score",
    # Step-consistency features
    "step_step_consistency_score",
    # First-pass parse features
    "fp_first_pass_parse_success",
    "fp_first_pass_has_final_answer_cue",
    "fp_first_pass_has_uncertainty_phrase",
    "fp_first_pass_empty_or_malformed_flag",
    # Calibration features
    "cal_predicted_answer_format_confidence",
    "cal_format_clean_numeric",
    "cal_format_many_numeric_candidates",
]

#: Mapping from manuscript regime id to the enriched routing dataset CSV path.
#: Covers the four main regimes reported in the paper.
REGIME_FILES: dict[str, str] = {
    "gsm8k_random_100": "data/real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": "data/real_math500_routing_dataset_enriched.csv",
}

#: Column for the routing label (1 = DPR is strictly better, 0 = otherwise).
LABEL_COL = "escalate_label"

#: Required columns from the raw enriched CSV.
_REQUIRED_COLS = {"reasoning_correct", "revise_correct"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_regime_df(path: str | Path) -> pd.DataFrame:
    """Load an enriched routing CSV into a DataFrame with the routing label.

    The label column ``escalate_label`` is derived as:
        ``escalate_label = (revise_correct == 1) & (reasoning_correct == 0)``
    i.e. escalate to DPR only when DPR is correct and RG is not.

    Parameters
    ----------
    path:
        Absolute or relative path to the enriched routing CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with all original columns plus ``escalate_label``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns (``reasoning_correct``, ``revise_correct``) are absent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Routing dataset not found: {p}")

    df = pd.read_csv(p)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from {p.name}: {missing}")

    df[LABEL_COL] = (
        (df["revise_correct"].fillna(0).astype(int) == 1)
        & (df["reasoning_correct"].fillna(0).astype(int) == 0)
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a numeric feature matrix and label vector from a routing DataFrame.

    Missing feature columns are zero-filled.  All values are coerced to float.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`load_regime_df`.
    feature_cols:
        Ordered list of feature column names.  Defaults to ``FEATURE_COLS``.

    Returns
    -------
    X : np.ndarray of shape (n, d)
        Feature matrix.
    y : np.ndarray of shape (n,)
        Binary label vector (``escalate_label``).
    feature_names : list[str]
        Column names corresponding to the columns of *X*.
    """
    cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS

    available = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]

    X_df = df[available].copy()
    for c in missing:
        X_df[c] = 0.0
    X_df = X_df[cols].fillna(0.0)

    X = X_df.values.astype(np.float32)
    y = df[LABEL_COL].values.astype(int) if LABEL_COL in df.columns else np.zeros(len(df), dtype=int)

    return X, y, cols


# ---------------------------------------------------------------------------
# Train / validation / test splitting
# ---------------------------------------------------------------------------


def build_training_dataset(
    regime_files: dict[str, str] | None = None,
    test_regime: str | None = None,
    val_fraction: float = 0.2,
    random_seed: int = 42,
    feature_cols: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Assemble training, validation, and test datasets from routing artifacts.

    Strategy
    --------
    - The *test_regime* (e.g. ``"hard_gsm8k_100"``) is held out entirely.
    - All other regimes are pooled; ``val_fraction`` of that pool is reserved as
      the validation set (for threshold selection and hyperparameter tuning).
    - If *test_regime* is ``None``, all regimes are used for training/validation
      and test arrays are empty.

    This respects the manuscript's artifact structure: evaluation numbers are
    only reported on regimes that were NOT seen during training.

    Parameters
    ----------
    regime_files:
        Mapping from regime id to CSV path.  Defaults to ``REGIME_FILES``.
    test_regime:
        Regime to hold out as the test set.  If ``None``, no test split is made.
    val_fraction:
        Fraction of pooled training data to use as validation.
    random_seed:
        Random seed for reproducible splits.
    feature_cols:
        Feature columns to use.  Defaults to ``FEATURE_COLS``.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
    feature_names : list[str]
    """
    files = regime_files if regime_files is not None else REGIME_FILES
    cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS

    train_Xs, train_ys = [], []
    test_X: np.ndarray = np.empty((0, len(cols)), dtype=np.float32)
    test_y: np.ndarray = np.empty(0, dtype=int)

    for regime, path in files.items():
        df = load_regime_df(path)
        X, y, names = build_feature_matrix(df, feature_cols=cols)
        if regime == test_regime:
            test_X = X
            test_y = y
        else:
            train_Xs.append(X)
            train_ys.append(y)

    if not train_Xs:
        empty = np.empty((0, len(cols)), dtype=np.float32)
        empty_y = np.empty(0, dtype=int)
        return empty, empty_y, empty, empty_y, test_X, test_y, cols

    X_pool = np.concatenate(train_Xs, axis=0)
    y_pool = np.concatenate(train_ys, axis=0)

    # Shuffle before splitting
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(len(X_pool))
    X_pool = X_pool[idx]
    y_pool = y_pool[idx]

    n_val = int(len(X_pool) * val_fraction) if val_fraction > 0 else 0
    X_val = X_pool[:n_val]
    y_val = y_pool[:n_val]
    X_train = X_pool[n_val:]
    y_train = y_pool[n_val:]

    return X_train, y_train, X_val, y_val, test_X, test_y, cols
