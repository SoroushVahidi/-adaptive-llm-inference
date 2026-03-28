"""Revise-help feature analysis using target-quantity and wording-trap features.

This module tests whether the target-quantity / wording-trap features added in
``src/features/target_quantity_features.py`` genuinely separate:

  - ``revise_helps``            direct_greedy wrong; direct_plus_revise correct
  - ``direct_already_enough``   direct_greedy correct; no extra compute needed
  - ``unique_other_strategy_case`` direct wrong; revise not helpful; other strategy correct
  - ``revise_not_enough``       no strategy was correct
  - ``reasoning_enough``        catch-all for remaining cases

For each query the module computes a combined feature vector from:
  - :func:`~src.features.precompute_features.extract_query_features`
    (13 existing features)
  - :func:`~src.features.target_quantity_features.extract_target_quantity_features`
    (11 new features)

It then aggregates feature rates per group and produces a ``feature_differences``
table (revise_helps − direct_already_enough for each feature) to identify the
strongest separating signals.

Inputs (all optional — missing files are handled gracefully):
    outputs/oracle_subset_eval/per_query_matrix.csv
    outputs/oracle_subset_eval/oracle_assignments.csv
    outputs/revise_case_analysis/case_table.csv
    src/datasets/bundled/gsm8k_test_sample.json       (fallback for question text)

Outputs written to outputs/revise_help_feature_analysis/:
    group_feature_rates.csv
    feature_differences.csv
    query_feature_table.csv
    example_cases.json

See docs/REVISE_HELP_FEATURE_ANALYSIS.md for full design rationale and results.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from src.features.precompute_features import extract_query_features
from src.features.target_quantity_features import extract_target_quantity_features

# ---------------------------------------------------------------------------
# Group labels
# ---------------------------------------------------------------------------

GROUP_REVISE_HELPS = "revise_helps"
GROUP_DIRECT_ALREADY_ENOUGH = "direct_already_enough"
GROUP_UNIQUE_OTHER_STRATEGY = "unique_other_strategy_case"
GROUP_REVISE_NOT_ENOUGH = "revise_not_enough"
GROUP_REASONING_ENOUGH = "reasoning_enough"

_ALL_GROUPS = [
    GROUP_REVISE_HELPS,
    GROUP_DIRECT_ALREADY_ENOUGH,
    GROUP_UNIQUE_OTHER_STRATEGY,
    GROUP_REVISE_NOT_ENOUGH,
    GROUP_REASONING_ENOUGH,
]

# ---------------------------------------------------------------------------
# Feature names used in the analysis
# ---------------------------------------------------------------------------

# All 11 target-quantity / wording-trap features
TARGET_QUANTITY_FEATURES: list[str] = [
    "asks_remaining_or_left",
    "asks_total",
    "asks_difference",
    "asks_rate_or_unit",
    "asks_money",
    "asks_time",
    "has_subtraction_trap_verb",
    "has_addition_trap_structure",
    "has_multi_operation_hint",
    "likely_intermediate_quantity_ask",
    "potential_answer_echo_risk",
]

# Selected base features to include in the analysis
BASE_BOOL_FEATURES: list[str] = [
    "has_multi_step_cue",
    "has_currency_symbol",
    "has_percent_symbol",
    "has_fraction_pattern",
    "has_equation_like_pattern",
    "repeated_number_flag",
]

BASE_NUMERIC_FEATURES: list[str] = [
    "question_length_chars",
    "question_length_tokens_approx",
    "num_numeric_mentions",
    "num_sentences_approx",
    "numeric_range_approx",
]

# Combined list of all boolean features used in rate comparisons
ALL_BOOL_FEATURES: list[str] = TARGET_QUANTITY_FEATURES + BASE_BOOL_FEATURES

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_bool_int(value: str | None, default: int = 0) -> int:
    """Parse a string 0/1/True/False into int (0 or 1)."""
    try:
        return int(float(value or "0"))
    except (ValueError, TypeError):
        return default


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV into a list of row dicts; return [] if file not found."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_oracle_assignments(path: str | Path) -> list[dict[str, str]]:
    """Load oracle_assignments.csv.  Returns [] if file not found."""
    return _read_csv(path)


def load_per_query_matrix(path: str | Path) -> list[dict[str, str]]:
    """Load per_query_matrix.csv.  Returns [] if file not found."""
    return _read_csv(path)


def load_case_table(path: str | Path) -> list[dict[str, str]]:
    """Load revise_case_analysis/case_table.csv if present."""
    return _read_csv(path)


def load_bundled_gsm8k(path: str | Path) -> list[dict[str, str]]:
    """Load the bundled GSM8K sample JSON.  Returns [] if file not found."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return []


# ---------------------------------------------------------------------------
# Question text helpers
# ---------------------------------------------------------------------------


def build_question_text_map(
    per_query_matrix: list[dict[str, str]],
    oracle_assignments: list[dict[str, str]],
    bundled_gsm8k: list[dict[str, str]],
) -> dict[str, str]:
    """Return {question_id: question_text} from all available sources.

    Priority order: per_query_matrix > oracle_assignments > bundled_gsm8k.
    The bundled sample uses ``id`` as the question identifier.
    """
    qmap: dict[str, str] = {}
    # Lowest priority: bundled GSM8K (id field)
    for item in bundled_gsm8k:
        qid = item.get("id", "")
        text = item.get("question", "")
        if qid and text:
            qmap[qid] = text
    # Mid priority: oracle_assignments
    for row in oracle_assignments:
        qid = row.get("question_id", "")
        text = row.get("question_text", row.get("question", ""))
        if qid and text:
            qmap[qid] = text
    # Highest priority: per_query_matrix
    for row in per_query_matrix:
        qid = row.get("question_id", "")
        text = row.get("question_text", row.get("question", ""))
        if qid and text:
            qmap[qid] = text
    return qmap


# ---------------------------------------------------------------------------
# Extended group assignment
# ---------------------------------------------------------------------------


def assign_extended_group(
    direct_greedy_correct: int,
    revise_correct: int | None,
    oracle_any_correct: int,
    direct_already_optimal: int,
    non_revise_strategy_correct: int,
) -> str:
    """Assign a query to one of five extended groups.

    Parameters
    ----------
    direct_greedy_correct:
        1 if direct_greedy answered correctly, else 0.
    revise_correct:
        1/0 if direct_plus_revise result is known, else None.
    oracle_any_correct:
        1 if at least one strategy was correct, else 0.
    direct_already_optimal:
        1 if oracle flagged direct_greedy as the cheapest correct strategy.
    non_revise_strategy_correct:
        1 if a strategy other than direct_greedy and direct_plus_revise was
        correct for this query.

    Groups
    ------
    direct_already_enough
        direct_greedy was correct; no extra compute needed.
    revise_helps
        direct_greedy wrong; direct_plus_revise correct.
    unique_other_strategy_case
        direct wrong; revise not helpful; a different strategy was correct.
    revise_not_enough
        no strategy was correct.
    reasoning_enough
        catch-all (e.g. direct wrong, revise unknown, oracle says possible).
    """
    if direct_greedy_correct:
        return GROUP_DIRECT_ALREADY_ENOUGH
    if revise_correct == 1:
        return GROUP_REVISE_HELPS
    if non_revise_strategy_correct:
        return GROUP_UNIQUE_OTHER_STRATEGY
    if oracle_any_correct == 0:
        return GROUP_REVISE_NOT_ENOUGH
    # revise unknown/wrong but oracle says solvable by other means
    return GROUP_REASONING_ENOUGH


def build_extended_group_map(
    oracle_assignments: list[dict[str, str]],
    per_query_matrix: list[dict[str, str]],
    case_category_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return {question_id: group_label} using the five-group classification.

    Parameters
    ----------
    oracle_assignments:
        Rows from oracle_assignments.csv.
    per_query_matrix:
        Rows from per_query_matrix.csv (strategy-level correct flags).
    case_category_overrides:
        Optional {question_id: category} from case_table.csv; values matching
        any of the five group labels override the computed assignment.
    """
    # Build per-query strategy→correct map from matrix
    strategy_correct: dict[str, dict[str, int]] = {}
    for row in per_query_matrix:
        qid = row.get("question_id", "")
        strat = row.get("strategy", "")
        correct = _parse_bool_int(row.get("correct", "0"))
        if qid and strat:
            if qid not in strategy_correct:
                strategy_correct[qid] = {}
            strategy_correct[qid][strat] = correct

    _REVISE_KEY = "direct_plus_revise"
    _DIRECT_KEY = "direct_greedy"
    _VALID_GROUPS = set(_ALL_GROUPS)

    group_map: dict[str, str] = {}
    for row in oracle_assignments:
        qid = row.get("question_id", "")
        if not qid:
            continue

        direct_correct = _parse_bool_int(row.get("direct_greedy_correct", "0"))
        direct_already_optimal = _parse_bool_int(row.get("direct_already_optimal", "0"))

        # Revise correct from matrix
        revise_correct: int | None = None
        if qid in strategy_correct and _REVISE_KEY in strategy_correct[qid]:
            revise_correct = strategy_correct[qid][_REVISE_KEY]

        # oracle_any_correct
        cheapest_strat = row.get("cheapest_correct_strategy", "")
        oracle_any_correct = 1 if cheapest_strat else 0
        if qid in strategy_correct and any(
            v == 1 for v in strategy_correct[qid].values()
        ):
            oracle_any_correct = 1

        # non_revise_strategy_correct: any strategy other than direct_greedy and revise
        non_revise = 0
        if qid in strategy_correct:
            for strat, val in strategy_correct[qid].items():
                if strat not in {_DIRECT_KEY, _REVISE_KEY} and val == 1:
                    non_revise = 1
                    break

        group_map[qid] = assign_extended_group(
            direct_correct,
            revise_correct,
            oracle_any_correct,
            direct_already_optimal,
            non_revise,
        )

    # Apply explicit overrides from case_table
    if case_category_overrides:
        for qid, cat in case_category_overrides.items():
            if cat in _VALID_GROUPS:
                group_map[qid] = cat

    return group_map


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_combined_features(qid: str, question_text: str) -> dict[str, Any]:
    """Return the full combined feature vector for a single query.

    Merges base query features with all 11 target-quantity features.  All
    boolean target-quantity features are stored as ``int`` (0/1) for
    uniformity with the base bool features.

    Parameters
    ----------
    qid:
        Query identifier (stored in the ``question_id`` key).
    question_text:
        Raw question string; may be empty (all features will be falsy).

    Returns
    -------
    dict
        Flat dict with keys: question_id, question_text, + all feature names.
    """
    base = extract_query_features(question_text)
    tq = extract_target_quantity_features(question_text)
    tq_int = {k: int(v) for k, v in tq.items()}
    return {"question_id": qid, "question_text": question_text, **base, **tq_int}


# ---------------------------------------------------------------------------
# Feature rate computation
# ---------------------------------------------------------------------------


def compute_group_feature_rates(
    group_features: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Compute per-group boolean feature rates and numeric feature means.

    Parameters
    ----------
    group_features:
        {group_label: [per-query feature dict, ...]}

    Returns
    -------
    list of dicts, one per group, sorted by group name.
    """
    rows: list[dict[str, Any]] = []
    for group_label in _ALL_GROUPS:
        feat_list = group_features.get(group_label, [])
        row: dict[str, Any] = {"group": group_label, "n": len(feat_list)}
        # Boolean features → rate
        for feat in ALL_BOOL_FEATURES:
            vals = [int(f.get(feat, 0)) for f in feat_list]
            rate = sum(vals) / len(vals) if vals else 0.0
            row[f"{feat}_rate"] = round(rate, 4)
        # Numeric features → mean
        for feat in BASE_NUMERIC_FEATURES:
            vals = [float(f[feat]) for f in feat_list if feat in f]
            mean = statistics.mean(vals) if vals else 0.0
            row[f"{feat}_mean"] = round(mean, 3)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Feature difference computation
# ---------------------------------------------------------------------------


def compute_feature_differences(
    rate_rows: list[dict[str, Any]],
    focus_group: str = GROUP_REVISE_HELPS,
    baseline_group: str = GROUP_DIRECT_ALREADY_ENOUGH,
) -> list[dict[str, Any]]:
    """Compute (focus_group_rate − baseline_group_rate) for each boolean feature.

    Parameters
    ----------
    rate_rows:
        Output of :func:`compute_group_feature_rates`.
    focus_group:
        The group we care about (default: ``revise_helps``).
    baseline_group:
        The comparison baseline (default: ``direct_already_enough``).

    Returns
    -------
    list of dicts sorted by |difference| descending; keys:
        feature, focus_rate, baseline_rate, difference.
    """
    focus_row = next((r for r in rate_rows if r["group"] == focus_group), {})
    baseline_row = next((r for r in rate_rows if r["group"] == baseline_group), {})

    diffs: list[dict[str, Any]] = []
    for feat in ALL_BOOL_FEATURES:
        key = f"{feat}_rate"
        focus_rate = float(focus_row.get(key, 0.0))
        baseline_rate = float(baseline_row.get(key, 0.0))
        diff = focus_rate - baseline_rate
        diffs.append({
            "feature": feat,
            "focus_group": focus_group,
            "baseline_group": baseline_group,
            "focus_rate": round(focus_rate, 4),
            "baseline_rate": round(baseline_rate, 4),
            "difference": round(diff, 4),
            "abs_difference": round(abs(diff), 4),
        })

    diffs.sort(key=lambda x: x["abs_difference"], reverse=True)
    return diffs


# ---------------------------------------------------------------------------
# Top separating features
# ---------------------------------------------------------------------------


def find_top_separating_features(
    diff_rows: list[dict[str, Any]],
    n: int = 5,
    min_abs_diff: float = 0.0,
) -> list[dict[str, Any]]:
    """Return the top-n features with the largest |difference|.

    Parameters
    ----------
    diff_rows:
        Output of :func:`compute_feature_differences`.
    n:
        Number of top features to return.
    min_abs_diff:
        Minimum absolute difference threshold; features below this are excluded.

    Returns
    -------
    list of dicts (subset of diff_rows), length ≤ n.
    """
    qualified = [r for r in diff_rows if r["abs_difference"] >= min_abs_diff]
    return qualified[:n]


# ---------------------------------------------------------------------------
# Example case selection
# ---------------------------------------------------------------------------


def build_example_cases(
    group_map: dict[str, str],
    per_query_features: dict[str, dict[str, Any]],
    focus_group: str = GROUP_REVISE_HELPS,
    n: int = 3,
) -> list[dict[str, Any]]:
    """Select up to n example queries from the focus group.

    Preference is given to queries that have the most target-quantity
    features firing (i.e. highest signal density), so that the examples
    illustrate meaningful patterns.

    Parameters
    ----------
    group_map:
        {question_id: group_label}
    per_query_features:
        {question_id: feature_dict}
    focus_group:
        Group to draw examples from (default: ``revise_helps``).
    n:
        Maximum number of examples to return.
    """
    candidates = [
        qid for qid, grp in group_map.items()
        if grp == focus_group and per_query_features.get(qid, {}).get("question_text", "")
    ]
    if not candidates:
        return []

    def _tq_signal_count(qid: str) -> int:
        feats = per_query_features.get(qid, {})
        return sum(int(feats.get(f, 0)) for f in TARGET_QUANTITY_FEATURES)

    candidates_sorted = sorted(candidates, key=_tq_signal_count, reverse=True)

    examples = []
    for qid in candidates_sorted[:n]:
        feats = per_query_features[qid]
        fired = [f for f in TARGET_QUANTITY_FEATURES if feats.get(f, 0)]
        examples.append({
            "question_id": qid,
            "group": focus_group,
            "question_text": feats.get("question_text", ""),
            "target_quantity_features_fired": fired,
            "n_tq_features_fired": len(fired),
            "feature_snapshot": {
                f: int(feats.get(f, 0)) for f in TARGET_QUANTITY_FEATURES
            },
        })
    return examples


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


def _build_recommendation(
    group_sizes: dict[str, int],
    top_features: list[dict[str, Any]],
    diff_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive a data-grounded recommendation on whether to hand-craft more features.

    Returns
    -------
    dict with keys: recommendation, justification, strongest_signals.
    """
    n_revise = group_sizes.get(GROUP_REVISE_HELPS, 0)
    n_direct = group_sizes.get(GROUP_DIRECT_ALREADY_ENOUGH, 0)

    # Count how many target-quantity features are in the top-5 separating features
    tq_in_top = sum(
        1 for r in top_features if r["feature"] in TARGET_QUANTITY_FEATURES
    )
    max_abs_diff = top_features[0]["abs_difference"] if top_features else 0.0

    strongest = [r["feature"] for r in top_features[:3]]

    if n_revise == 0 or n_direct == 0:
        recommendation = "insufficient_data"
        justification = (
            "No real oracle output data available. Run oracle evaluation with "
            "real model outputs, then re-run this analysis to get a genuine "
            "recommendation. Current feature vectors are computed from the "
            "bundled 20-query GSM8K sample with synthetic group assignments."
        )
    elif max_abs_diff < 0.1:
        recommendation = "move_to_learned_routing"
        justification = (
            f"The largest feature difference is only {max_abs_diff:.3f}. "
            "No target-quantity feature shows meaningful separation between "
            "revise_helps and direct_already_enough. Manual feature engineering "
            "has reached diminishing returns. Move to learned routing."
        )
    elif tq_in_top >= 3 and max_abs_diff >= 0.2:
        recommendation = "one_more_hand_crafted_router"
        justification = (
            f"{tq_in_top}/5 top-separating features come from the new "
            "target-quantity family, with max |diff|="
            f"{max_abs_diff:.3f}. This is strong enough to justify one more "
            "hand-crafted router attempt before learned routing."
        )
    else:
        recommendation = "move_to_learned_routing"
        justification = (
            f"Top feature difference is {max_abs_diff:.3f}, and only "
            f"{tq_in_top}/5 top features come from the new target-quantity "
            "family. The signal is marginal. Proceed to learned routing using "
            "the full combined feature vector as input."
        )

    return {
        "recommendation": recommendation,
        "justification": justification,
        "strongest_separating_signals": strongest,
        "n_revise_helps": n_revise,
        "n_direct_already_enough": n_direct,
        "max_abs_feature_difference": max_abs_diff,
        "tq_features_in_top_5": tq_in_top,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_group_feature_rates(
    rate_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> str:
    """Write group_feature_rates.csv; return the written path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = base / "group_feature_rates.csv"
    if not rate_rows:
        out.write_text("group,n\n")
        return str(out)
    fieldnames = list(rate_rows[0].keys())
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rate_rows)
    return str(out)


def write_feature_differences(
    diff_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> str:
    """Write feature_differences.csv; return the written path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = base / "feature_differences.csv"
    if not diff_rows:
        out.write_text("feature,focus_group,baseline_group,focus_rate,baseline_rate,"
                       "difference,abs_difference\n")
        return str(out)
    fieldnames = list(diff_rows[0].keys())
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diff_rows)
    return str(out)


def write_query_feature_table(
    per_query_features: dict[str, dict[str, Any]],
    group_map: dict[str, str],
    output_dir: str | Path,
) -> str:
    """Write query_feature_table.csv; return the written path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = base / "query_feature_table.csv"

    rows = []
    for qid, feats in per_query_features.items():
        row = {"question_id": qid, "group": group_map.get(qid, "unknown")}
        row["question_text"] = feats.get("question_text", "")
        for feat in TARGET_QUANTITY_FEATURES + BASE_BOOL_FEATURES + BASE_NUMERIC_FEATURES:
            row[feat] = feats.get(feat, "")
        rows.append(row)

    if not rows:
        out.write_text("question_id,group\n")
        return str(out)

    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(out)


def write_example_cases(
    examples: list[dict[str, Any]],
    output_dir: str | Path,
) -> str:
    """Write example_cases.json; return the written path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = base / "example_cases.json"
    out.write_text(json.dumps(examples, indent=2))
    return str(out)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_revise_help_feature_analysis(
    oracle_assignments_path: str | Path,
    per_query_matrix_path: str | Path,
    case_table_path: str | Path,
    bundled_gsm8k_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the full revise-help feature analysis pipeline.

    All input paths are optional — missing files are handled gracefully.
    When no oracle data is present, the analysis runs on any queries found
    in the bundled GSM8K sample (all assigned to a synthetic group for
    demonstration purposes).

    Parameters
    ----------
    oracle_assignments_path:
        Path to outputs/oracle_subset_eval/oracle_assignments.csv.
    per_query_matrix_path:
        Path to outputs/oracle_subset_eval/per_query_matrix.csv.
    case_table_path:
        Path to outputs/revise_case_analysis/case_table.csv.
    bundled_gsm8k_path:
        Path to src/datasets/bundled/gsm8k_test_sample.json.
    output_dir:
        Directory where output files are written.

    Returns
    -------
    dict with keys:
        group_sizes, feature_rates, feature_differences, top_separating_features,
        example_cases, recommendation, output_paths.
    """
    # ---- Load inputs -------------------------------------------------------
    oracle_assignments = load_oracle_assignments(oracle_assignments_path)
    per_query_matrix = load_per_query_matrix(per_query_matrix_path)
    case_table = load_case_table(case_table_path)
    bundled_gsm8k = load_bundled_gsm8k(bundled_gsm8k_path)

    # ---- Case table category overrides -------------------------------------
    case_overrides: dict[str, str] = {}
    for row in case_table:
        qid = row.get("question_id", "")
        cat = row.get("category", row.get("group", ""))
        if qid and cat:
            case_overrides[qid] = cat

    # ---- Build group map ---------------------------------------------------
    group_map = build_extended_group_map(
        oracle_assignments, per_query_matrix, case_overrides
    )

    # ---- Build question text map -------------------------------------------
    qtext_map = build_question_text_map(per_query_matrix, oracle_assignments, bundled_gsm8k)

    # If no oracle data, include all bundled queries with a placeholder group
    # so that feature vectors are always computed and output files have rows.
    if not group_map and bundled_gsm8k:
        for item in bundled_gsm8k:
            qid = item.get("id", "")
            if qid:
                group_map[qid] = GROUP_REASONING_ENOUGH

    # ---- Compute per-query features ----------------------------------------
    all_qids = set(group_map.keys()) | set(qtext_map.keys())
    per_query_features: dict[str, dict[str, Any]] = {}
    for qid in all_qids:
        text = qtext_map.get(qid, "")
        per_query_features[qid] = extract_combined_features(qid, text)
        # Ensure group is set for all queries
        if qid not in group_map:
            group_map[qid] = GROUP_REASONING_ENOUGH

    # ---- Build group feature lists -----------------------------------------
    group_features: dict[str, list[dict[str, Any]]] = {g: [] for g in _ALL_GROUPS}
    for qid, grp in group_map.items():
        if grp in group_features:
            group_features[grp].append(per_query_features.get(qid, {}))
        else:
            group_features[GROUP_REASONING_ENOUGH].append(
                per_query_features.get(qid, {})
            )

    # ---- Compute feature rates and differences -----------------------------
    rate_rows = compute_group_feature_rates(group_features)
    diff_rows = compute_feature_differences(rate_rows)
    top_features = find_top_separating_features(diff_rows, n=5)

    # ---- Select example cases ----------------------------------------------
    example_cases = build_example_cases(group_map, per_query_features)

    # ---- Build recommendation ----------------------------------------------
    group_sizes = {g: len(fl) for g, fl in group_features.items()}
    recommendation = _build_recommendation(group_sizes, top_features, diff_rows)

    # ---- Write outputs -----------------------------------------------------
    out_dir = Path(output_dir)
    rates_path = write_group_feature_rates(rate_rows, out_dir)
    diffs_path = write_feature_differences(diff_rows, out_dir)
    table_path = write_query_feature_table(per_query_features, group_map, out_dir)
    examples_path = write_example_cases(example_cases, out_dir)

    return {
        "group_sizes": group_sizes,
        "feature_rates": rate_rows,
        "feature_differences": diff_rows,
        "top_separating_features": top_features,
        "example_cases": example_cases,
        "recommendation": recommendation,
        "output_paths": {
            "group_feature_rates_csv": rates_path,
            "feature_differences_csv": diffs_path,
            "query_feature_table_csv": table_path,
            "example_cases_json": examples_path,
        },
    }
