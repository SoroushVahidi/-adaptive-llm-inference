"""Feature-method-fit offline analysis.

Builds a unified per-query analysis table from the four main manuscript
regimes and runs lightweight analyses to identify which features are
associated with different routing outcomes and method choices.

No API calls, no new LLM inference. All data comes from:
  - data/real_*_routing_dataset.csv
  - outputs/real_*_policy_eval/per_query_policy_decisions.csv

See docs/FEATURE_METHOD_FIT_EXPERIMENT.md for the full feature schema.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Regime configuration
# ---------------------------------------------------------------------------

REGIMES: list[dict[str, str]] = [
    {
        "regime": "gsm8k_random_100",
        "routing_csv": "data/real_gsm8k_routing_dataset.csv",
        "policy_csv": "outputs/real_policy_eval/per_query_policy_decisions.csv",
    },
    {
        "regime": "hard_gsm8k_100",
        "routing_csv": "data/real_hard_gsm8k_routing_dataset.csv",
        "policy_csv": "outputs/real_hard_gsm8k_policy_eval/per_query_policy_decisions.csv",
    },
    {
        "regime": "hard_gsm8k_b2",
        "routing_csv": "data/real_hard_gsm8k_b2_routing_dataset.csv",
        "policy_csv": "outputs/real_hard_gsm8k_b2_policy_eval/per_query_policy_decisions.csv",
    },
    {
        "regime": "math500_100",
        "routing_csv": "data/real_math500_routing_dataset.csv",
        "policy_csv": "outputs/real_math500_policy_eval/per_query_policy_decisions.csv",
    },
]

# ---------------------------------------------------------------------------
# Feature definitions (see docs/FEATURE_METHOD_FIT_EXPERIMENT.md)
# ---------------------------------------------------------------------------

QUESTION_SIDE_FEATURES = [
    "prompt_number_count",
    "prompt_token_length",
    "target_quantity_type",
    "multi_stepness_proxy",
    "explicit_constraint_presence",
    "relational_wording_presence",
    "special_structure_presence",
]

OUTPUT_SIDE_FEATURES = [
    "final_answer_parseable",
    "body_final_numeric_mismatch",
    "target_quantity_mismatch",
    "constraint_violation_signal",
    "copied_question_number_as_final_answer",
    "cheap_route_confidence",
    "explanation_warning_signal",
    "answer_error_signal",
]

ALL_FEATURES = QUESTION_SIDE_FEATURES + OUTPUT_SIDE_FEATURES

NUMERIC_FEATURES = ["prompt_number_count", "prompt_token_length"]
BINARY_FEATURES = [f for f in ALL_FEATURES if f not in NUMERIC_FEATURES + ["target_quantity_type"]]
CATEGORICAL_FEATURES = ["target_quantity_type"]

OUTCOME_LABELS = [
    "revise_helpful",
    "safe_cheap",
    "both_wrong",
    "unnecessary_revise_candidate",
]

METHODS = ["reasoning_greedy", "direct_plus_revise", "v5", "v6", "v7"]


# ---------------------------------------------------------------------------
# CSV helpers (stdlib only)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(val: str | None, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "nan") else default
    except (ValueError, TypeError):
        return default


def _int(val: str | None, default: int = 0) -> int:
    return int(_float(val, default))


def _bool_col(val: str | None) -> int:
    """Return 1 if truthy (non-zero float, 'True', '1'), else 0."""
    s = str(val).strip().lower()
    if s in ("true", "1", "1.0"):
        return 1
    if s in ("false", "0", "0.0", "", "nan", "none"):
        return 0
    try:
        return int(float(s) != 0)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Feature derivation for a single row
# ---------------------------------------------------------------------------

def _derive_features(row: dict[str, str]) -> dict[str, Any]:
    """Derive the 15 conceptual features from a routing-dataset row."""

    # --- Question-side -------------------------------------------------------

    # F1: prompt_number_count → q_num_numeric_mentions
    f1 = _float(row.get("q_num_numeric_mentions"), 0.0)

    # F2: prompt_token_length → q_question_length_tokens_approx
    f2 = _float(row.get("q_question_length_tokens_approx"), 0.0)

    # F3: target_quantity_type (categorical, priority order)
    tq_map = [
        ("tq_asks_rate_or_unit", "rate_or_unit"),
        ("tq_asks_remaining_or_left", "remaining_or_left"),
        ("tq_asks_difference", "difference"),
        ("tq_asks_total", "total"),
        ("tq_asks_money", "money"),
        ("tq_asks_time", "time"),
    ]
    f3 = "other"
    for col, label in tq_map:
        if _bool_col(row.get(col)):
            f3 = label
            break

    # F4: multi_stepness_proxy → tq_has_multi_operation_hint
    f4 = _bool_col(row.get("tq_has_multi_operation_hint"))

    # F5: explicit_constraint_presence (subtraction OR addition trap verb)
    f5 = int(
        _bool_col(row.get("tq_has_subtraction_trap_verb"))
        or _bool_col(row.get("tq_has_addition_trap_structure"))
    )

    # F6: relational_wording_presence (difference OR remaining/left phrasing)
    f6 = int(
        _bool_col(row.get("tq_asks_difference"))
        or _bool_col(row.get("tq_asks_remaining_or_left"))
    )

    # F7: special_structure_presence (percent / fraction / rate / time)
    f7 = int(
        _bool_col(row.get("q_has_percent_symbol"))
        or _bool_col(row.get("q_has_fraction_pattern"))
        or _bool_col(row.get("tq_asks_rate_or_unit"))
        or _bool_col(row.get("tq_asks_time"))
    )

    # --- Output-side ---------------------------------------------------------

    # F8: final_answer_parseable → fp_first_pass_parse_success
    f8 = _bool_col(row.get("fp_first_pass_parse_success"))

    # F9: body_final_numeric_mismatch → v7_extra_answer_error > 0
    #     (V7 structural signals: weekday+numeric, need_more, tail_equals)
    f9 = int(_float(row.get("v7_extra_answer_error"), 0.0) > 0)

    # F10: target_quantity_mismatch → cons_target_quantity_mismatch_suspected
    f10 = _bool_col(row.get("cons_target_quantity_mismatch_suspected"))

    # F11: constraint_violation_signal (any strong constraint flag, excl. F10)
    f11 = int(
        _bool_col(row.get("cons_answer_type_mismatch_suspected"))
        or _bool_col(row.get("cons_unit_mismatch_suspected"))
        or _bool_col(row.get("cons_impossible_sign_suspected"))
        or _bool_col(row.get("cons_integer_expected_but_noninteger_suspected"))
        or _bool_col(row.get("cons_constraint_word_conflict_suspected"))
        or _bool_col(row.get("cons_bound_violation_suspected"))
    )

    # F12: copied_question_number_as_final_answer → tq_potential_answer_echo_risk
    #      (question-side proxy; best available column)
    f12 = _bool_col(row.get("tq_potential_answer_echo_risk"))

    # F13: cheap_route_confidence → v6_final_answer_confident
    f13 = _bool_col(row.get("v6_final_answer_confident"))

    # F14: explanation_warning_signal → v6_explanation_warning_score > 0
    f14 = int(_float(row.get("v6_explanation_warning_score"), 0.0) > 0)

    # F15: answer_error_signal → v6_answer_error_score > 0
    f15 = int(_float(row.get("v6_answer_error_score"), 0.0) > 0)

    return {
        "prompt_number_count": f1,
        "prompt_token_length": f2,
        "target_quantity_type": f3,
        "multi_stepness_proxy": f4,
        "explicit_constraint_presence": f5,
        "relational_wording_presence": f6,
        "special_structure_presence": f7,
        "final_answer_parseable": f8,
        "body_final_numeric_mismatch": f9,
        "target_quantity_mismatch": f10,
        "constraint_violation_signal": f11,
        "copied_question_number_as_final_answer": f12,
        "cheap_route_confidence": f13,
        "explanation_warning_signal": f14,
        "answer_error_signal": f15,
    }


# ---------------------------------------------------------------------------
# Outcome label derivation
# ---------------------------------------------------------------------------

def _derive_outcomes(
    routing_row: dict[str, str],
    policy_row: dict[str, str] | None,
) -> dict[str, Any]:
    """Derive outcome labels for a single query."""
    rg_correct = _int(routing_row.get("reasoning_correct"))
    dpr_correct = _int(routing_row.get("revise_correct"))

    revise_helpful = int(rg_correct == 0 and dpr_correct == 1)
    safe_cheap = int(rg_correct == 1)
    both_wrong = int(rg_correct == 0 and dpr_correct == 0)

    # unnecessary_revise_candidate: RG correct but policy would trigger revise
    v6_rev = _bool_col(routing_row.get("v6_revise_recommended"))
    v7_rev = _bool_col(routing_row.get("v7_revise_recommended"))
    unnecessary_revise_candidate = int(rg_correct == 1 and (v6_rev or v7_rev))

    # Policy outcomes (from policy_eval CSVs, if available)
    policy_cols: dict[str, Any] = {
        "policy_v5": None,
        "policy_v6": None,
        "policy_v7": None,
        "correct_if_v5": None,
        "correct_if_v6": None,
        "correct_if_v7": None,
        "cost_v5": None,
        "cost_v6": None,
        "cost_v7": None,
    }
    if policy_row is not None:
        for k in policy_cols:
            v = policy_row.get(k, "")
            policy_cols[k] = v if v != "" else None

    # method_best_label: highest correctness, lowest cost tie-break
    # Build candidate list: (method, correct, cost)
    candidates: list[tuple[str, int, float]] = [
        ("reasoning_greedy", rg_correct, 1.0),
        ("direct_plus_revise", dpr_correct, 2.0),
    ]
    if policy_row is not None:
        for suffix, name in [("v5", "adaptive_v5"), ("v6", "adaptive_v6"), ("v7", "adaptive_v7")]:
            c = policy_row.get(f"correct_if_{suffix}", "")
            cost = policy_row.get(f"cost_{suffix}", "")
            if c != "" and cost != "":
                candidates.append((name, _int(c), _float(cost)))

    best_correct = max(t[1] for t in candidates)
    best_candidates = [t for t in candidates if t[1] == best_correct]
    best_cost = min(t[2] for t in best_candidates)
    best = [t for t in best_candidates if t[2] == best_cost]
    method_best_label = best[0][0] if best else "reasoning_greedy"

    return {
        "reasoning_correct": rg_correct,
        "revise_correct": dpr_correct,
        "revise_helpful": revise_helpful,
        "safe_cheap": safe_cheap,
        "both_wrong": both_wrong,
        "unnecessary_revise_candidate": unnecessary_revise_candidate,
        "method_best_label": method_best_label,
        **policy_cols,
    }


# ---------------------------------------------------------------------------
# Build unified analysis dataset
# ---------------------------------------------------------------------------

def build_analysis_dataset(repo_root: Path) -> list[dict[str, Any]]:
    """Merge per-regime data into one flat list of row dicts."""
    rows: list[dict[str, Any]] = []

    for cfg in REGIMES:
        regime = cfg["regime"]
        routing_path = repo_root / cfg["routing_csv"]
        policy_path = repo_root / cfg["policy_csv"]

        if not routing_path.exists():
            warnings.warn(
                f"[{regime}] routing CSV not found: {routing_path}. Skipping regime.",
                stacklevel=2,
            )
            continue

        routing_rows = _read_csv(routing_path)

        # Build policy lookup by question_id
        policy_lookup: dict[str, dict[str, str]] = {}
        if policy_path.exists():
            for pr in _read_csv(policy_path):
                qid = pr.get("question_id", "")
                if qid:
                    policy_lookup[qid] = pr
        else:
            warnings.warn(
                f"[{regime}] policy CSV not found: {policy_path}. "
                "Policy columns will be null.",
                stacklevel=2,
            )

        for rr in routing_rows:
            qid = rr.get("question_id", "")
            policy_row = policy_lookup.get(qid)

            features = _derive_features(rr)
            outcomes = _derive_outcomes(rr, policy_row)

            row: dict[str, Any] = {
                "regime": regime,
                "question_id": qid,
                **outcomes,
                **features,
                # Raw v6 scores for reference
                "v6_answer_error_score_raw": _float(rr.get("v6_answer_error_score")),
                "v6_explanation_warning_score_raw": _float(
                    rr.get("v6_explanation_warning_score")
                ),
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _group(rows: list[dict], outcome_col: str, val: int = 1) -> list[dict]:
    return [r for r in rows if r.get(outcome_col) == val]


def compute_univariate_summary(rows: list[dict]) -> list[dict[str, Any]]:
    """
    For each binary/numeric feature, compute group-wise means for each
    outcome class and a simple effect-size proxy (max_rate - min_rate).
    """
    results = []
    for feat in ALL_FEATURES:
        if feat == "target_quantity_type":
            continue  # categorical — handled separately

        side = "question" if feat in QUESTION_SIDE_FEATURES else "output"

        outcome_means: dict[str, float] = {}
        for outcome in OUTCOME_LABELS:
            grp = _group(rows, outcome)
            vals = [float(r[feat]) for r in grp if feat in r and str(r[feat]) != "nan"]
            outcome_means[f"mean_{outcome}"] = _mean(vals)

        all_means = [v for v in outcome_means.values() if v == v]  # drop NaN
        effect_size_proxy = (max(all_means) - min(all_means)) if len(all_means) >= 2 else 0.0

        # Overall prevalence
        all_vals = [float(r[feat]) for r in rows if feat in r and str(r[feat]) != "nan"]
        overall_mean = _mean(all_vals)

        results.append(
            {
                "feature": feat,
                "side": side,
                "overall_mean": round(overall_mean, 4),
                **{k: round(v, 4) for k, v in outcome_means.items()},
                "effect_size_proxy": round(effect_size_proxy, 4),
            }
        )
    return results


def compute_target_quantity_type_breakdown(rows: list[dict]) -> list[dict[str, Any]]:
    """Frequency breakdown of target_quantity_type by outcome."""
    all_types = sorted({r.get("target_quantity_type", "other") for r in rows})
    result = []
    for tqt in all_types:
        subset = [r for r in rows if r.get("target_quantity_type") == tqt]
        n = len(subset)
        entry: dict[str, Any] = {"target_quantity_type": tqt, "n": n}
        for outcome in OUTCOME_LABELS:
            cnt = sum(1 for r in subset if r.get(outcome) == 1)
            entry[f"rate_{outcome}"] = round(cnt / n, 4) if n > 0 else 0.0
        result.append(entry)
    return result


def compute_method_fit_summary(rows: list[dict]) -> list[dict[str, Any]]:
    """
    For each best-method label, compute feature means to characterize
    queries that are best served by each routing method.
    """
    methods = sorted({r.get("method_best_label", "") for r in rows if r.get("method_best_label")})
    result = []
    for method in methods:
        subset = [r for r in rows if r.get("method_best_label") == method]
        n = len(subset)
        entry: dict[str, Any] = {"method_best_label": method, "n": n}
        for feat in BINARY_FEATURES + NUMERIC_FEATURES:
            vals = [float(r[feat]) for r in subset if feat in r and str(r[feat]) != "nan"]
            entry[f"mean_{feat}"] = round(_mean(vals), 4)
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Lightweight model: logistic regression for revise_helpful
# ---------------------------------------------------------------------------

def run_logistic_model(
    rows: list[dict],
) -> dict[str, Any]:
    """
    Fit a L2-regularised logistic regression to predict revise_helpful.
    Uses binary + numeric features only (excludes target_quantity_type).
    Returns coefficients, class balance, and cross-val accuracy (if sklearn
    is available; otherwise returns a coefficient-free stub).
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"blocker": "scikit-learn not available", "model": "logistic_regression"}

    feature_cols = BINARY_FEATURES + NUMERIC_FEATURES
    X_raw = []
    y = []
    for r in rows:
        row_feats = [float(r.get(f, 0.0)) for f in feature_cols]
        label = int(r.get("revise_helpful", 0))
        X_raw.append(row_feats)
        y.append(label)

    X = np.array(X_raw)
    y_arr = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Suppress convergence warnings on small data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_scaled, y_arr)

    coefs = dict(zip(feature_cols, clf.coef_[0].tolist()))

    # Cross-val accuracy (3-fold; stratified if possible)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=500, random_state=42),
            X_scaled,
            y_arr,
            cv=3,
            scoring="accuracy",
        )

    pos = int(y_arr.sum())
    neg = int(len(y_arr) - pos)

    return {
        "model": "logistic_regression",
        "n_total": len(y),
        "n_positive": pos,
        "n_negative": neg,
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "coefficients": {k: round(v, 4) for k, v in coefs.items()},
    }


def run_decision_tree_model(
    rows: list[dict],
) -> dict[str, Any]:
    """
    Fit a shallow decision tree to predict revise_helpful.
    Returns feature importances.
    """
    try:
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        return {"blocker": "scikit-learn not available", "model": "decision_tree"}

    feature_cols = BINARY_FEATURES + NUMERIC_FEATURES
    X_raw = []
    y = []
    for r in rows:
        row_feats = [float(r.get(f, 0.0)) for f in feature_cols]
        label = int(r.get("revise_helpful", 0))
        X_raw.append(row_feats)
        y.append(label)

    X = np.array(X_raw)
    y_arr = np.array(y)

    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    clf.fit(X, y_arr)

    importances = dict(zip(feature_cols, clf.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    return {
        "model": "decision_tree",
        "n_total": len(y),
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
    }


# ---------------------------------------------------------------------------
# Final feature ranking
# ---------------------------------------------------------------------------

# Manual interpretability scores (1–5): 5 = most interpretable
_INTERPRETABILITY = {
    "prompt_number_count": 5,
    "prompt_token_length": 5,
    "target_quantity_type": 5,
    "multi_stepness_proxy": 4,
    "explicit_constraint_presence": 4,
    "relational_wording_presence": 4,
    "special_structure_presence": 4,
    "final_answer_parseable": 5,
    "body_final_numeric_mismatch": 3,
    "target_quantity_mismatch": 4,
    "constraint_violation_signal": 4,
    "copied_question_number_as_final_answer": 3,
    "cheap_route_confidence": 5,
    "explanation_warning_signal": 4,
    "answer_error_signal": 5,
}

# Redundancy notes
_REDUNDANCY = {
    "prompt_number_count": "",
    "prompt_token_length": "Correlated with prompt_number_count",
    "target_quantity_type": "",
    "multi_stepness_proxy": "Overlaps with explicit_constraint_presence",
    "explicit_constraint_presence": "Subset of relational_wording_presence signals",
    "relational_wording_presence": "",
    "special_structure_presence": "",
    "final_answer_parseable": "",
    "body_final_numeric_mismatch": "Captures V7-specific cases only; low base rate",
    "target_quantity_mismatch": "Subset of constraint_violation_signal",
    "constraint_violation_signal": "",
    "copied_question_number_as_final_answer": (
        "Question-side proxy; overlaps with answer_error_signal at output"
    ),
    "cheap_route_confidence": "Summary signal; correlated with answer_error_signal",
    "explanation_warning_signal": "Weak association expected from V6 architecture",
    "answer_error_signal": "",
}


def compute_feature_ranking(
    univariate: list[dict],
    logistic_result: dict,
    dt_result: dict,
    rows: list[dict],
) -> list[dict[str, Any]]:
    """Assemble final feature ranking table."""
    # Build lookup tables
    univ_by_feat = {r["feature"]: r for r in univariate}
    coefs = logistic_result.get("coefficients", {})
    dt_imp = dt_result.get("feature_importances", {})

    ranking = []
    for feat in ALL_FEATURES:
        side = "question" if feat in QUESTION_SIDE_FEATURES else "output"
        univ = univ_by_feat.get(feat, {})
        effect = univ.get("effect_size_proxy", float("nan"))
        mean_rh = univ.get("mean_revise_helpful", float("nan"))
        mean_sc = univ.get("mean_safe_cheap", float("nan"))

        # Association signals
        coef = coefs.get(feat, float("nan"))
        imp = dt_imp.get(feat, float("nan"))

        interp = _INTERPRETABILITY.get(feat, 3)
        redundancy = _REDUNDANCY.get(feat, "")

        # Manuscript recommendation heuristic:
        # "yes" if interpretable (>=4) and notable effect (>0.05) and not redundant
        # "maybe" if moderate
        # "no" if very low effect or highly redundant
        notable = effect > 0.05 if effect == effect else False
        redundant = bool(redundancy)
        if interp >= 4 and notable and not redundant:
            rec = "yes"
        elif interp >= 4 and (notable or not redundant):
            rec = "maybe"
        else:
            rec = "no"

        # Override: answer_error and cheap_route_confidence always recommended
        if feat in ("answer_error_signal", "cheap_route_confidence"):
            rec = "yes"
        # explanation_warning_signal: "maybe" — theoretically motivated
        if feat == "explanation_warning_signal":
            rec = "maybe"

        ranking.append(
            {
                "feature_name": feat,
                "question_or_output_side": side,
                "availability": "direct"
                if feat
                not in (
                    "body_final_numeric_mismatch",
                    "constraint_violation_signal",
                    "explicit_constraint_presence",
                    "relational_wording_presence",
                    "special_structure_presence",
                    "explanation_warning_signal",
                    "answer_error_signal",
                )
                else "derived",
                "mean_revise_helpful": round(mean_rh, 4) if mean_rh == mean_rh else "",
                "mean_safe_cheap": round(mean_sc, 4) if mean_sc == mean_sc else "",
                "effect_size_proxy": round(effect, 4) if effect == effect else "",
                "logistic_coef": round(coef, 4) if coef == coef else "",
                "dt_importance": round(imp, 4) if imp == imp else "",
                "association_with_revise_helpful": _qualitative_assoc(coef, effect, "revise"),
                "association_with_method_fit": _qualitative_assoc(imp, effect, "method"),
                "interpretability_score": interp,
                "redundancy_notes": redundancy,
                "manuscript_recommendation": rec,
            }
        )

    return sorted(ranking, key=lambda r: -(float(r["effect_size_proxy"] or 0)))


def _qualitative_assoc(main_stat: float, effect: float, _kind: str) -> str:
    """Convert numerical stat to qualitative label."""
    if main_stat != main_stat or effect != effect:
        return "unknown"
    if abs(effect) > 0.20:
        return "strong"
    if abs(effect) > 0.08:
        return "moderate"
    if abs(effect) > 0.03:
        return "weak"
    return "negligible"


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feature_method_fit_analysis(
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the full analysis and write all output files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build unified analysis dataset
    rows = build_analysis_dataset(repo_root)
    if not rows:
        raise RuntimeError("No data rows found — check that routing CSVs exist.")

    # Write main dataset
    dataset_path = output_dir / "feature_analysis_dataset.csv"
    write_csv(dataset_path, rows)

    # Step 2: Univariate summary
    univariate = compute_univariate_summary(rows)
    univ_path = output_dir / "univariate_feature_summary.csv"
    write_csv(univ_path, univariate)

    # Step 3: Target-quantity type breakdown
    tqt_breakdown = compute_target_quantity_type_breakdown(rows)
    tqt_path = output_dir / "target_quantity_type_breakdown.csv"
    write_csv(tqt_path, tqt_breakdown)

    # Step 4: Method-fit descriptive summary
    method_fit = compute_method_fit_summary(rows)
    method_fit_path = output_dir / "method_fit_descriptive_summary.csv"
    write_csv(method_fit_path, method_fit)

    # Step 5: Logistic regression model
    logistic_result = run_logistic_model(rows)
    lr_path = output_dir / "revise_helpful_model_summary.csv"
    lr_rows = [{"param": k, "value": str(v)} for k, v in logistic_result.items()
               if k not in ("coefficients",)]
    if "coefficients" in logistic_result:
        for feat, coef in logistic_result["coefficients"].items():
            lr_rows.append({"param": f"coef_{feat}", "value": str(coef)})
    write_csv(lr_path, lr_rows)

    # Step 6: Decision tree model
    dt_result = run_decision_tree_model(rows)
    dt_path = output_dir / "method_fit_model_summary.csv"
    dt_rows = [{"param": k, "value": str(v)} for k, v in dt_result.items()
               if k != "feature_importances"]
    if "feature_importances" in dt_result:
        for feat, imp in dt_result["feature_importances"].items():
            dt_rows.append({"param": f"importance_{feat}", "value": str(imp)})
    write_csv(dt_path, dt_rows)

    # Step 7: Feature importance (combined from LR + DT)
    feat_imp_rows = []
    for feat in ALL_FEATURES:
        if feat == "target_quantity_type":
            continue
        coef = logistic_result.get("coefficients", {}).get(feat, "")
        imp = dt_result.get("feature_importances", {}).get(feat, "")
        feat_imp_rows.append({
            "feature": feat,
            "side": "question" if feat in QUESTION_SIDE_FEATURES else "output",
            "logistic_coef": coef,
            "dt_importance": imp,
        })
    feat_imp_path = output_dir / "feature_importance.csv"
    write_csv(feat_imp_path, feat_imp_rows)

    # Step 8: Final feature ranking
    ranking = compute_feature_ranking(univariate, logistic_result, dt_result, rows)
    ranking_path = output_dir / "final_feature_ranking.csv"
    write_csv(ranking_path, ranking)

    # Step 9: Paper table (top features, compact)
    paper_table = [
        {
            "feature_name": r["feature_name"],
            "side": r["question_or_output_side"],
            "interpretability": r["interpretability_score"],
            "effect_size_proxy": r["effect_size_proxy"],
            "assoc_revise_helpful": r["association_with_revise_helpful"],
            "assoc_method_fit": r["association_with_method_fit"],
            "manuscript_recommendation": r["manuscript_recommendation"],
            "notes": r["redundancy_notes"] or "—",
        }
        for r in ranking
    ]
    paper_table_dir = repo_root / "outputs" / "paper_tables"
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    paper_table_path = paper_table_dir / "feature_method_fit_main_table.csv"
    write_csv(paper_table_path, paper_table)

    # Build summary stats for report generation
    n_total = len(rows)
    by_regime: dict[str, int] = {}
    for r in rows:
        by_regime[r["regime"]] = by_regime.get(r["regime"], 0) + 1

    outcome_counts: dict[str, int] = {}
    for outcome in OUTCOME_LABELS:
        outcome_counts[outcome] = sum(1 for r in rows if r.get(outcome) == 1)

    return {
        "n_total": n_total,
        "by_regime": by_regime,
        "outcome_counts": outcome_counts,
        "univariate": univariate,
        "method_fit": method_fit,
        "logistic_result": logistic_result,
        "dt_result": dt_result,
        "ranking": ranking,
        "output_paths": {
            "dataset": str(dataset_path),
            "univariate": str(univ_path),
            "tqt_breakdown": str(tqt_path),
            "method_fit": str(method_fit_path),
            "lr_model": str(lr_path),
            "dt_model": str(dt_path),
            "feature_importance": str(feat_imp_path),
            "ranking": str(ranking_path),
            "paper_table": str(paper_table_path),
        },
    }
