"""Lightweight learned routing baselines for the four main manuscript regimes.

Trains **logistic regression** and a **shallow decision tree** (max depth 4) to
predict ``revise_helpful`` (should the policy apply revision?) from the
pre-computed features in the enriched routing datasets.  Both models use
``class_weight='balanced'`` to handle the strong class imbalance (revise_helpful
positive rate ranges from 2 % to 12 % across regimes).

Evaluation strategy
-------------------
Each regime has exactly 100 labelled examples.  To obtain honest held-out
estimates we use **5-fold stratified cross-validation** within each regime.
Where class imbalance makes stratification infeasible (fewer than 5 positives),
we fall back to un-stratified k-fold and flag it in the output.

The routing *decision* is:
    predicted_revise = 1  →  ``direct_plus_revise``  (cost 2.0, accuracy = revise_correct)
    predicted_revise = 0  →  ``reasoning_greedy``     (cost 1.0, accuracy = reasoning_correct)

Metrics reported
----------------
- ``accuracy`` — fraction of correctly classified queries (routing outcome)
- ``avg_cost``  — 1.0 + revise_rate  (matches manuscript cost model)
- ``revise_rate`` — fraction of queries routed to revise
- ``cv_folds`` — number of folds actually used
- ``degenerate`` — True if all predictions are the majority class

Public API
----------
- ``FEATURE_COLS`` — list of feature column names used as model inputs.
- ``REGIME_FILES`` — mapping from regime id to enriched CSV path.
- ``LearnedRouterResult`` — dataclass for per-regime results.
- ``evaluate_regime(regime, df, model_name, cv_folds)`` → LearnedRouterResult
- ``run_all_regimes(regime_files, output_dir)``
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Feature columns used by the learned router.  These are all present in the
#: enriched routing CSVs and require no additional API calls to compute.
FEATURE_COLS: list[str] = [
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
    "role_warning_score",
    "role_strong_error_score",
    "unified_error_score",
    "unified_confidence_score",
    "self_self_verification_score",
    "step_step_consistency_score",
    "fp_first_pass_parse_success",
    "fp_first_pass_has_final_answer_cue",
    "fp_first_pass_has_uncertainty_phrase",
    "fp_first_pass_empty_or_malformed_flag",
    "cal_predicted_answer_format_confidence",
    "cal_format_clean_numeric",
    "cal_format_many_numeric_candidates",
]

REGIME_FILES: dict[str, str] = {
    "gsm8k_random_100": "data/real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": "data/real_math500_routing_dataset_enriched.csv",
}

LABEL_COL = "revise_helpful"
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LearnedRouterResult:
    """Per-regime, per-model learned-router evaluation result."""

    regime: str
    model_name: str
    accuracy: float
    avg_cost: float
    revise_rate: float
    cv_folds: int
    degenerate: bool
    n: int
    note: str = ""

    def to_summary_dict(self) -> dict:
        return {
            "regime": self.regime,
            "baseline": f"learned_router_{self.model_name}",
            "accuracy": self.accuracy,
            "avg_cost": self.avg_cost,
            "revise_rate": self.revise_rate,
            "cv_folds": self.cv_folds,
            "degenerate": self.degenerate,
            "n": self.n,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# sklearn helpers (graceful fallback when unavailable)
# ---------------------------------------------------------------------------

_SKLEARN_AVAILABLE: bool | None = None


def _has_sklearn() -> bool:
    global _SKLEARN_AVAILABLE  # noqa: PLW0603
    if _SKLEARN_AVAILABLE is None:
        try:
            import sklearn  # noqa: F401

            _SKLEARN_AVAILABLE = True
        except ImportError:
            _SKLEARN_AVAILABLE = False
    return _SKLEARN_AVAILABLE


def _make_model(model_name: str) -> Any:
    """Instantiate a balanced sklearn classifier by name."""
    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    elif model_name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")


# ---------------------------------------------------------------------------
# Simple fallback cross-validation (no sklearn required for the eval loop)
# ---------------------------------------------------------------------------


def _cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int = CV_FOLDS,
) -> tuple[np.ndarray, int, bool]:
    """Run stratified (or plain) k-fold CV; return (oof_predictions, folds_used, stratified).

    Returns predictions aligned with the input indices.
    """
    n = len(y)
    preds = np.zeros(n, dtype=int)
    n_pos = int(np.sum(y))

    # Determine fold count and stratification
    actual_folds = n_folds
    stratified = n_pos >= n_folds
    if n_pos < 2:
        # Cannot fit any meaningful model; predict majority everywhere
        majority = int(np.bincount(y).argmax())
        preds[:] = majority
        return preds, 1, False

    if not stratified:
        actual_folds = max(2, n_pos)

    try:
        from sklearn.model_selection import StratifiedKFold, KFold

        if stratified:
            kf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
            splits = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
            splits = list(kf.split(X))
    except ImportError:
        # Manual stratified split fallback (no sklearn)
        splits = _manual_splits(n, actual_folds)

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        clf = _make_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        preds[test_idx] = clf.predict(X_test)

    return preds, actual_folds, stratified


def _manual_splits(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n)
    fold_sizes = np.full(k, n // k)
    fold_sizes[: n % k] += 1
    splits = []
    current = 0
    for size in fold_sizes:
        test = indices[current : current + size]
        train = np.concatenate([indices[:current], indices[current + size :]])
        splits.append((train, test))
        current += size
    return splits


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_regime(
    regime: str,
    df: pd.DataFrame,
    model_name: str = "logistic_regression",
    cv_folds: int = CV_FOLDS,
) -> LearnedRouterResult:
    """Fit and CV-evaluate a learned router on one regime's dataset.

    Columns required in *df*:
    - All columns in ``FEATURE_COLS`` (missing ones are zero-filled with a note)
    - ``revise_helpful``, ``reasoning_correct``, ``revise_correct``
    """
    if not _has_sklearn():
        return LearnedRouterResult(
            regime=regime,
            model_name=model_name,
            accuracy=float("nan"),
            avg_cost=float("nan"),
            revise_rate=float("nan"),
            cv_folds=0,
            degenerate=True,
            n=len(df),
            note="scikit-learn not available; skipped",
        )

    # Build feature matrix
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    X_df = df[available_cols].copy()
    for c in missing_cols:
        X_df[c] = 0.0
    X_df = X_df.fillna(0.0)
    X = X_df.values.astype(float)
    y = df[LABEL_COL].values.astype(int)

    note = ""
    if missing_cols:
        note = f"zero-filled {len(missing_cols)} missing feature cols"

    preds, folds_used, stratified = _cv_predict(X, y, model_name, cv_folds)

    degenerate = len(np.unique(preds)) == 1

    # Compute routing outcomes
    revise_mask = preds == 1
    correct = revise_mask * df["revise_correct"].values + (~revise_mask) * df["reasoning_correct"].values
    accuracy = float(correct.mean())
    revise_rate = float(revise_mask.mean())
    avg_cost = 1.0 + revise_rate

    if degenerate and not note:
        note = "degenerate: all predictions are the same class"
    if not stratified and not note:
        note = f"non-stratified CV due to low positive count ({int(y.sum())})"

    return LearnedRouterResult(
        regime=regime,
        model_name=model_name,
        accuracy=accuracy,
        avg_cost=avg_cost,
        revise_rate=revise_rate,
        cv_folds=folds_used,
        degenerate=degenerate,
        n=len(df),
        note=note,
    )


def run_all_regimes(
    regime_files: dict[str, str] | None = None,
    output_dir: str | Path = "outputs/baselines/learned_router",
    cv_folds: int = CV_FOLDS,
) -> list[LearnedRouterResult]:
    """Evaluate logistic regression and decision tree on all four regimes.

    Writes
    ------
    ``<output_dir>/learned_router_summary.csv``
        Per-regime, per-model summary (accuracy, avg_cost, revise_rate, etc.)
    ``<output_dir>/learned_router_summary.json``
        Same data as JSON list.
    """
    files = regime_files if regime_files is not None else REGIME_FILES
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[LearnedRouterResult] = []
    for regime, csv_path in files.items():
        df = pd.read_csv(csv_path)
        for model_name in ["logistic_regression", "decision_tree"]:
            result = evaluate_regime(regime, df, model_name=model_name, cv_folds=cv_folds)
            all_results.append(result)

    # Write CSV
    summary_path = out_dir / "learned_router_summary.csv"
    rows = [r.to_summary_dict() for r in all_results]
    if rows:
        with summary_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Write JSON
    json_path = out_dir / "learned_router_summary.json"
    json_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2))

    return all_results
