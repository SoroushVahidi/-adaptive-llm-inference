"""Unit tests for src/analysis/revise_help_feature_analysis.py.

All tests are fully offline — no API calls, no oracle output files required
unless synthesised in a temporary directory inside the test.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.analysis.revise_help_feature_analysis import (
    ALL_BOOL_FEATURES,
    GROUP_DIRECT_ALREADY_ENOUGH,
    GROUP_REASONING_ENOUGH,
    GROUP_REVISE_HELPS,
    GROUP_REVISE_NOT_ENOUGH,
    GROUP_UNIQUE_OTHER_STRATEGY,
    TARGET_QUANTITY_FEATURES,
    _build_recommendation,
    assign_extended_group,
    build_example_cases,
    build_extended_group_map,
    build_question_text_map,
    compute_feature_differences,
    compute_group_feature_rates,
    extract_combined_features,
    find_top_separating_features,
    load_bundled_gsm8k,
    load_case_table,
    load_oracle_assignments,
    load_per_query_matrix,
    run_revise_help_feature_analysis,
    write_example_cases,
    write_feature_differences,
    write_group_feature_rates,
    write_query_feature_table,
)

# ---------------------------------------------------------------------------
# CSV / JSON helpers
# ---------------------------------------------------------------------------


def _write_oracle_assignments(tmpdir: Path, rows: list[dict]) -> Path:
    path = tmpdir / "oracle_assignments.csv"
    if not rows:
        path.write_text(
            "question_id,cheapest_correct_strategy,direct_greedy_correct,"
            "direct_already_optimal\n"
        )
        return path
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_per_query_matrix(tmpdir: Path, rows: list[dict]) -> Path:
    path = tmpdir / "per_query_matrix.csv"
    if not rows:
        path.write_text("question_id,strategy,correct\n")
        return path
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_case_table(tmpdir: Path, rows: list[dict]) -> Path:
    path = tmpdir / "case_table.csv"
    if not rows:
        path.write_text("question_id,group\n")
        return path
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_bundled_gsm8k(tmpdir: Path, items: list[dict]) -> Path:
    path = tmpdir / "gsm8k_test_sample.json"
    path.write_text(json.dumps(items))
    return path


# ---------------------------------------------------------------------------
# Constants / structure tests
# ---------------------------------------------------------------------------


def test_group_label_constants() -> None:
    assert GROUP_REVISE_HELPS == "revise_helps"
    assert GROUP_DIRECT_ALREADY_ENOUGH == "direct_already_enough"
    assert GROUP_UNIQUE_OTHER_STRATEGY == "unique_other_strategy_case"
    assert GROUP_REVISE_NOT_ENOUGH == "revise_not_enough"
    assert GROUP_REASONING_ENOUGH == "reasoning_enough"


def test_target_quantity_features_list_length() -> None:
    assert len(TARGET_QUANTITY_FEATURES) == 11


def test_all_bool_features_contains_target_quantity() -> None:
    for feat in TARGET_QUANTITY_FEATURES:
        assert feat in ALL_BOOL_FEATURES


# ---------------------------------------------------------------------------
# assign_extended_group
# ---------------------------------------------------------------------------


def test_assign_group_direct_correct() -> None:
    result = assign_extended_group(
        direct_greedy_correct=1,
        revise_correct=None,
        oracle_any_correct=1,
        direct_already_optimal=1,
        non_revise_strategy_correct=0,
    )
    assert result == GROUP_DIRECT_ALREADY_ENOUGH


def test_assign_group_revise_helps() -> None:
    result = assign_extended_group(
        direct_greedy_correct=0,
        revise_correct=1,
        oracle_any_correct=1,
        direct_already_optimal=0,
        non_revise_strategy_correct=0,
    )
    assert result == GROUP_REVISE_HELPS


def test_assign_group_unique_other_strategy() -> None:
    result = assign_extended_group(
        direct_greedy_correct=0,
        revise_correct=0,
        oracle_any_correct=1,
        direct_already_optimal=0,
        non_revise_strategy_correct=1,
    )
    assert result == GROUP_UNIQUE_OTHER_STRATEGY


def test_assign_group_revise_not_enough_no_oracle() -> None:
    result = assign_extended_group(
        direct_greedy_correct=0,
        revise_correct=None,
        oracle_any_correct=0,
        direct_already_optimal=0,
        non_revise_strategy_correct=0,
    )
    assert result == GROUP_REVISE_NOT_ENOUGH


def test_assign_group_revise_not_enough_revise_wrong() -> None:
    result = assign_extended_group(
        direct_greedy_correct=0,
        revise_correct=0,
        oracle_any_correct=0,
        direct_already_optimal=0,
        non_revise_strategy_correct=0,
    )
    assert result == GROUP_REVISE_NOT_ENOUGH


def test_assign_group_reasoning_enough_fallback() -> None:
    # direct wrong, revise unknown, some strategy correct but not non-revise-specific
    result = assign_extended_group(
        direct_greedy_correct=0,
        revise_correct=None,
        oracle_any_correct=1,
        direct_already_optimal=0,
        non_revise_strategy_correct=0,
    )
    assert result == GROUP_REASONING_ENOUGH


# ---------------------------------------------------------------------------
# build_extended_group_map
# ---------------------------------------------------------------------------


def test_build_extended_group_map_revise_helps(tmp_path: Path) -> None:
    oracle = [
        {
            "question_id": "q1",
            "direct_greedy_correct": "0",
            "direct_already_optimal": "0",
            "cheapest_correct_strategy": "direct_plus_revise",
        }
    ]
    matrix = [
        {"question_id": "q1", "strategy": "direct_plus_revise", "correct": "1"},
        {"question_id": "q1", "strategy": "direct_greedy", "correct": "0"},
    ]
    result = build_extended_group_map(oracle, matrix)
    assert result["q1"] == GROUP_REVISE_HELPS


def test_build_extended_group_map_direct_already_enough(tmp_path: Path) -> None:
    oracle = [
        {
            "question_id": "q2",
            "direct_greedy_correct": "1",
            "direct_already_optimal": "1",
            "cheapest_correct_strategy": "direct_greedy",
        }
    ]
    result = build_extended_group_map(oracle, [])
    assert result["q2"] == GROUP_DIRECT_ALREADY_ENOUGH


def test_build_extended_group_map_case_table_override(tmp_path: Path) -> None:
    oracle = [
        {
            "question_id": "q3",
            "direct_greedy_correct": "1",
            "direct_already_optimal": "0",
            "cheapest_correct_strategy": "",
        }
    ]
    overrides = {"q3": GROUP_REVISE_HELPS}
    result = build_extended_group_map(oracle, [], overrides)
    assert result["q3"] == GROUP_REVISE_HELPS


def test_build_extended_group_map_unique_other_strategy(tmp_path: Path) -> None:
    oracle = [
        {
            "question_id": "q4",
            "direct_greedy_correct": "0",
            "direct_already_optimal": "0",
            "cheapest_correct_strategy": "best_of_n",
        }
    ]
    matrix = [
        {"question_id": "q4", "strategy": "direct_greedy", "correct": "0"},
        {"question_id": "q4", "strategy": "direct_plus_revise", "correct": "0"},
        {"question_id": "q4", "strategy": "best_of_n", "correct": "1"},
    ]
    result = build_extended_group_map(oracle, matrix)
    assert result["q4"] == GROUP_UNIQUE_OTHER_STRATEGY


# ---------------------------------------------------------------------------
# build_question_text_map
# ---------------------------------------------------------------------------


def test_build_question_text_map_prefers_matrix_over_bundled() -> None:
    matrix = [{"question_id": "q1", "question_text": "matrix text"}]
    oracle = [{"question_id": "q1", "question_text": "oracle text"}]
    bundled = [{"id": "q1", "question": "bundled text"}]
    qmap = build_question_text_map(matrix, oracle, bundled)
    assert qmap["q1"] == "matrix text"


def test_build_question_text_map_falls_back_to_bundled() -> None:
    qmap = build_question_text_map([], [], [{"id": "q99", "question": "bundled only"}])
    assert qmap["q99"] == "bundled only"


def test_build_question_text_map_empty_inputs() -> None:
    qmap = build_question_text_map([], [], [])
    assert qmap == {}


# ---------------------------------------------------------------------------
# extract_combined_features
# ---------------------------------------------------------------------------


def test_extract_combined_features_structure() -> None:
    feats = extract_combined_features("q1", "She spent $5 and has $3 remaining.")
    assert feats["question_id"] == "q1"
    assert feats["question_text"] == "She spent $5 and has $3 remaining."
    # Target-quantity features
    for feat in TARGET_QUANTITY_FEATURES:
        assert feat in feats, f"Missing target-quantity feature: {feat}"
        assert isinstance(feats[feat], int)
    # Base features
    assert "num_numeric_mentions" in feats
    assert "has_multi_step_cue" in feats


def test_extract_combined_features_empty_string() -> None:
    feats = extract_combined_features("q0", "")
    for feat in TARGET_QUANTITY_FEATURES:
        assert feats[feat] == 0


def test_extract_combined_features_no_key_collision() -> None:
    # Check known target-quantity key names don't collide with base key names
    base_keys = {
        "question_length_chars", "num_numeric_mentions", "has_multi_step_cue",
        "has_currency_symbol", "has_fraction_pattern",
    }
    tq_keys = set(TARGET_QUANTITY_FEATURES)
    assert base_keys.isdisjoint(tq_keys)


def test_extract_combined_features_remaining_detected() -> None:
    feats = extract_combined_features("q1", "How many apples are remaining?")
    assert feats["asks_remaining_or_left"] == 1


def test_extract_combined_features_subtraction_trap_detected() -> None:
    feats = extract_combined_features("q1", "She spent $20 on clothes.")
    assert feats["has_subtraction_trap_verb"] == 1


def test_extract_combined_features_money_detected() -> None:
    feats = extract_combined_features("q1", "He earned $50 per day for 5 days.")
    assert feats["asks_money"] == 1
    assert feats["asks_rate_or_unit"] == 1


# ---------------------------------------------------------------------------
# compute_group_feature_rates
# ---------------------------------------------------------------------------


def test_compute_group_feature_rates_returns_all_groups() -> None:
    group_features = {
        GROUP_REVISE_HELPS: [extract_combined_features("q1", "She left 3 apples.")],
        GROUP_DIRECT_ALREADY_ENOUGH: [extract_combined_features("q2", "2 + 2.")],
        GROUP_UNIQUE_OTHER_STRATEGY: [],
        GROUP_REVISE_NOT_ENOUGH: [],
        GROUP_REASONING_ENOUGH: [],
    }
    rows = compute_group_feature_rates(group_features)
    groups = [r["group"] for r in rows]
    assert GROUP_REVISE_HELPS in groups
    assert GROUP_DIRECT_ALREADY_ENOUGH in groups
    assert len(rows) == 5


def test_compute_group_feature_rates_correct_rate() -> None:
    q_remaining = extract_combined_features("q1", "How many are remaining?")
    q_other = extract_combined_features("q2", "What is 2+2?")
    group_features = {
        GROUP_REVISE_HELPS: [q_remaining, q_other],
        GROUP_DIRECT_ALREADY_ENOUGH: [],
        GROUP_UNIQUE_OTHER_STRATEGY: [],
        GROUP_REVISE_NOT_ENOUGH: [],
        GROUP_REASONING_ENOUGH: [],
    }
    rows = compute_group_feature_rates(group_features)
    rh_row = next(r for r in rows if r["group"] == GROUP_REVISE_HELPS)
    # 1/2 queries have asks_remaining_or_left
    assert rh_row["asks_remaining_or_left_rate"] == pytest.approx(0.5)


def test_compute_group_feature_rates_empty_group() -> None:
    group_features = {g: [] for g in [
        GROUP_REVISE_HELPS, GROUP_DIRECT_ALREADY_ENOUGH,
        GROUP_UNIQUE_OTHER_STRATEGY, GROUP_REVISE_NOT_ENOUGH, GROUP_REASONING_ENOUGH,
    ]}
    rows = compute_group_feature_rates(group_features)
    for row in rows:
        assert row["n"] == 0
        assert row["asks_remaining_or_left_rate"] == 0.0


# ---------------------------------------------------------------------------
# compute_feature_differences
# ---------------------------------------------------------------------------


def test_compute_feature_differences_structure() -> None:
    group_features = {
        GROUP_REVISE_HELPS: [
            extract_combined_features("q1", "How many apples are remaining?"),
        ],
        GROUP_DIRECT_ALREADY_ENOUGH: [
            extract_combined_features("q2", "What is 2 + 2?"),
        ],
        GROUP_UNIQUE_OTHER_STRATEGY: [],
        GROUP_REVISE_NOT_ENOUGH: [],
        GROUP_REASONING_ENOUGH: [],
    }
    rates = compute_group_feature_rates(group_features)
    diffs = compute_feature_differences(rates)
    assert len(diffs) == len(ALL_BOOL_FEATURES)
    for row in diffs:
        assert "feature" in row
        assert "focus_rate" in row
        assert "baseline_rate" in row
        assert "difference" in row
        assert "abs_difference" in row


def test_compute_feature_differences_sorted_by_abs_diff() -> None:
    group_features = {
        GROUP_REVISE_HELPS: [
            extract_combined_features("q1", "She spent $5 and has $3 remaining."),
        ],
        GROUP_DIRECT_ALREADY_ENOUGH: [
            extract_combined_features("q2", "The answer is 4."),
        ],
        GROUP_UNIQUE_OTHER_STRATEGY: [],
        GROUP_REVISE_NOT_ENOUGH: [],
        GROUP_REASONING_ENOUGH: [],
    }
    rates = compute_group_feature_rates(group_features)
    diffs = compute_feature_differences(rates)
    abs_diffs = [r["abs_difference"] for r in diffs]
    assert abs_diffs == sorted(abs_diffs, reverse=True)


def test_compute_feature_differences_custom_groups() -> None:
    group_features = {
        GROUP_REVISE_HELPS: [
            extract_combined_features("q1", "How many remain?"),
        ],
        GROUP_REASONING_ENOUGH: [
            extract_combined_features("q2", "What is the total?"),
        ],
        GROUP_DIRECT_ALREADY_ENOUGH: [],
        GROUP_UNIQUE_OTHER_STRATEGY: [],
        GROUP_REVISE_NOT_ENOUGH: [],
    }
    rates = compute_group_feature_rates(group_features)
    diffs = compute_feature_differences(rates, focus_group=GROUP_REVISE_HELPS,
                                        baseline_group=GROUP_REASONING_ENOUGH)
    assert all(r["focus_group"] == GROUP_REVISE_HELPS for r in diffs)
    assert all(r["baseline_group"] == GROUP_REASONING_ENOUGH for r in diffs)


# ---------------------------------------------------------------------------
# find_top_separating_features
# ---------------------------------------------------------------------------


def test_find_top_separating_features_returns_n() -> None:
    group_features = {
        GROUP_REVISE_HELPS: [
            extract_combined_features("q1", "She spent $5 and has $3 remaining."),
        ],
        GROUP_DIRECT_ALREADY_ENOUGH: [
            extract_combined_features("q2", "What is 2 + 2?"),
        ],
        GROUP_UNIQUE_OTHER_STRATEGY: [],
        GROUP_REVISE_NOT_ENOUGH: [],
        GROUP_REASONING_ENOUGH: [],
    }
    rates = compute_group_feature_rates(group_features)
    diffs = compute_feature_differences(rates)
    top = find_top_separating_features(diffs, n=3)
    assert len(top) <= 3


def test_find_top_separating_features_min_abs_diff_filter() -> None:
    group_features = {g: [] for g in [
        GROUP_REVISE_HELPS, GROUP_DIRECT_ALREADY_ENOUGH,
        GROUP_UNIQUE_OTHER_STRATEGY, GROUP_REVISE_NOT_ENOUGH, GROUP_REASONING_ENOUGH,
    ]}
    rates = compute_group_feature_rates(group_features)
    diffs = compute_feature_differences(rates)
    # All differences will be 0 when groups are empty; min_abs_diff=0.5 should filter all
    top = find_top_separating_features(diffs, n=5, min_abs_diff=0.5)
    assert top == []


def test_find_top_separating_features_empty_input() -> None:
    top = find_top_separating_features([], n=5)
    assert top == []


# ---------------------------------------------------------------------------
# build_example_cases
# ---------------------------------------------------------------------------


def test_build_example_cases_selects_from_revise_helps() -> None:
    group_map = {
        "q1": GROUP_REVISE_HELPS,
        "q2": GROUP_DIRECT_ALREADY_ENOUGH,
        "q3": GROUP_REVISE_HELPS,
    }
    per_query_features = {
        "q1": extract_combined_features("q1", "She spent $10 and has $5 left."),
        "q2": extract_combined_features("q2", "What is 2 + 2?"),
        "q3": extract_combined_features("q3", "How many apples are remaining?"),
    }
    examples = build_example_cases(group_map, per_query_features, n=3)
    assert all(e["group"] == GROUP_REVISE_HELPS for e in examples)
    assert len(examples) <= 2  # only 2 revise_helps queries exist


def test_build_example_cases_structure() -> None:
    group_map = {"q1": GROUP_REVISE_HELPS}
    per_query_features = {
        "q1": extract_combined_features(
            "q1", "Janet spent $15 per day for 3 days. How much is left?"
        ),
    }
    examples = build_example_cases(group_map, per_query_features)
    assert len(examples) == 1
    ex = examples[0]
    assert "question_id" in ex
    assert "group" in ex
    assert "question_text" in ex
    assert "target_quantity_features_fired" in ex
    assert "n_tq_features_fired" in ex
    assert "feature_snapshot" in ex


def test_build_example_cases_prefers_high_signal_density() -> None:
    """Queries with more TQ features firing should appear first."""
    group_map = {
        "q_rich": GROUP_REVISE_HELPS,
        "q_sparse": GROUP_REVISE_HELPS,
    }
    per_query_features = {
        "q_rich": extract_combined_features(
            "q_rich",
            "She spent $5 per hour for 3 hours. How much money does she have left?",
        ),
        "q_sparse": extract_combined_features(
            "q_sparse",
            "What is 2 + 3?",
        ),
    }
    examples = build_example_cases(group_map, per_query_features, n=2)
    assert examples[0]["question_id"] == "q_rich"


def test_build_example_cases_no_revise_helps() -> None:
    group_map = {"q1": GROUP_DIRECT_ALREADY_ENOUGH}
    per_query_features = {"q1": extract_combined_features("q1", "What is 2 + 2?")}
    examples = build_example_cases(group_map, per_query_features)
    assert examples == []


def test_build_example_cases_empty_question_text_excluded() -> None:
    group_map = {"q1": GROUP_REVISE_HELPS}
    per_query_features = {"q1": extract_combined_features("q1", "")}
    examples = build_example_cases(group_map, per_query_features)
    assert examples == []


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def test_load_oracle_assignments_missing_file(tmp_path: Path) -> None:
    result = load_oracle_assignments(tmp_path / "nonexistent.csv")
    assert result == []


def test_load_per_query_matrix_missing_file(tmp_path: Path) -> None:
    result = load_per_query_matrix(tmp_path / "nonexistent.csv")
    assert result == []


def test_load_case_table_missing_file(tmp_path: Path) -> None:
    result = load_case_table(tmp_path / "nonexistent.csv")
    assert result == []


def test_load_bundled_gsm8k_missing_file(tmp_path: Path) -> None:
    result = load_bundled_gsm8k(tmp_path / "nonexistent.json")
    assert result == []


def test_load_bundled_gsm8k_valid_file(tmp_path: Path) -> None:
    data = [{"id": "gsm8k_test_0", "question": "What is 2+2?", "answer": "#### 4"}]
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(data))
    result = load_bundled_gsm8k(p)
    assert len(result) == 1
    assert result[0]["id"] == "gsm8k_test_0"


def test_load_bundled_gsm8k_malformed_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    result = load_bundled_gsm8k(p)
    assert result == []


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def test_write_group_feature_rates_creates_file(tmp_path: Path) -> None:
    rates = compute_group_feature_rates({g: [] for g in [
        GROUP_REVISE_HELPS, GROUP_DIRECT_ALREADY_ENOUGH,
        GROUP_UNIQUE_OTHER_STRATEGY, GROUP_REVISE_NOT_ENOUGH, GROUP_REASONING_ENOUGH,
    ]})
    path = write_group_feature_rates(rates, tmp_path)
    assert Path(path).exists()
    rows = list(csv.DictReader(Path(path).open()))
    assert len(rows) == 5


def test_write_feature_differences_creates_file(tmp_path: Path) -> None:
    diffs = [
        {
            "feature": "asks_remaining_or_left",
            "focus_group": GROUP_REVISE_HELPS,
            "baseline_group": GROUP_DIRECT_ALREADY_ENOUGH,
            "focus_rate": 0.5,
            "baseline_rate": 0.1,
            "difference": 0.4,
            "abs_difference": 0.4,
        }
    ]
    path = write_feature_differences(diffs, tmp_path)
    assert Path(path).exists()
    rows = list(csv.DictReader(Path(path).open()))
    assert rows[0]["feature"] == "asks_remaining_or_left"


def test_write_query_feature_table_creates_file(tmp_path: Path) -> None:
    feats = {"q1": extract_combined_features("q1", "She spent $5 and has $3 left.")}
    group_map = {"q1": GROUP_REVISE_HELPS}
    path = write_query_feature_table(feats, group_map, tmp_path)
    assert Path(path).exists()
    rows = list(csv.DictReader(Path(path).open()))
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["group"] == GROUP_REVISE_HELPS


def test_write_example_cases_creates_file(tmp_path: Path) -> None:
    examples = [{"question_id": "q1", "group": GROUP_REVISE_HELPS}]
    path = write_example_cases(examples, tmp_path)
    assert Path(path).exists()
    loaded = json.loads(Path(path).read_text())
    assert loaded[0]["question_id"] == "q1"


def test_write_example_cases_empty_list(tmp_path: Path) -> None:
    path = write_example_cases([], tmp_path)
    assert Path(path).exists()
    assert json.loads(Path(path).read_text()) == []


# ---------------------------------------------------------------------------
# _build_recommendation
# ---------------------------------------------------------------------------


def test_build_recommendation_insufficient_data() -> None:
    rec = _build_recommendation(
        {GROUP_REVISE_HELPS: 0, GROUP_DIRECT_ALREADY_ENOUGH: 0},
        [],
        [],
    )
    assert rec["recommendation"] == "insufficient_data"


def test_build_recommendation_low_diff_moves_to_learned_routing() -> None:
    # All features have 0 difference
    diff_rows = [
        {
            "feature": "asks_remaining_or_left",
            "focus_group": GROUP_REVISE_HELPS,
            "baseline_group": GROUP_DIRECT_ALREADY_ENOUGH,
            "focus_rate": 0.1,
            "baseline_rate": 0.1,
            "difference": 0.0,
            "abs_difference": 0.0,
        }
    ]
    top = find_top_separating_features(diff_rows, n=5)
    rec = _build_recommendation(
        {GROUP_REVISE_HELPS: 10, GROUP_DIRECT_ALREADY_ENOUGH: 10},
        top,
        diff_rows,
    )
    assert rec["recommendation"] == "move_to_learned_routing"


def test_build_recommendation_strong_tq_signal() -> None:
    diff_rows = [
        {
            "feature": feat,
            "focus_group": GROUP_REVISE_HELPS,
            "baseline_group": GROUP_DIRECT_ALREADY_ENOUGH,
            "focus_rate": 0.9,
            "baseline_rate": 0.1,
            "difference": 0.8,
            "abs_difference": 0.8,
        }
        for feat in TARGET_QUANTITY_FEATURES[:5]
    ]
    top = find_top_separating_features(diff_rows, n=5)
    rec = _build_recommendation(
        {GROUP_REVISE_HELPS: 10, GROUP_DIRECT_ALREADY_ENOUGH: 10},
        top,
        diff_rows,
    )
    assert rec["recommendation"] == "one_more_hand_crafted_router"


# ---------------------------------------------------------------------------
# run_revise_help_feature_analysis — end-to-end with synthetic oracle data
# ---------------------------------------------------------------------------


def _make_synthetic_oracle_data(tmpdir: Path) -> tuple[Path, Path, Path, Path]:
    """Create minimal synthetic CSVs so the full pipeline can run."""
    oracle_rows = [
        {
            "question_id": "q1",
            "direct_greedy_correct": "0",
            "direct_already_optimal": "0",
            "cheapest_correct_strategy": "direct_plus_revise",
            "question_text": "Janet spent $5. How many dollars does she have left?",
        },
        {
            "question_id": "q2",
            "direct_greedy_correct": "1",
            "direct_already_optimal": "1",
            "cheapest_correct_strategy": "direct_greedy",
            "question_text": "What is 2 + 2?",
        },
        {
            "question_id": "q3",
            "direct_greedy_correct": "0",
            "direct_already_optimal": "0",
            "cheapest_correct_strategy": "",
            "question_text": "The answer to this problem is unknowable.",
        },
    ]
    matrix_rows = [
        {"question_id": "q1", "strategy": "direct_greedy", "correct": "0",
         "question_text": "Janet spent $5. How many dollars does she have left?"},
        {"question_id": "q1", "strategy": "direct_plus_revise", "correct": "1",
         "question_text": "Janet spent $5. How many dollars does she have left?"},
        {"question_id": "q2", "strategy": "direct_greedy", "correct": "1",
         "question_text": "What is 2 + 2?"},
        {"question_id": "q3", "strategy": "direct_greedy", "correct": "0",
         "question_text": ""},
    ]
    oracle_path = _write_oracle_assignments(tmpdir, oracle_rows)
    matrix_path = _write_per_query_matrix(tmpdir, matrix_rows)
    case_path = _write_case_table(tmpdir, [])
    bundled = [
        {"id": "q1", "question": "Janet spent $5.", "answer": "#### 5"},
    ]
    bundled_path = _write_bundled_gsm8k(tmpdir, bundled)
    return oracle_path, matrix_path, case_path, bundled_path


def test_run_revise_help_feature_analysis_end_to_end(tmp_path: Path) -> None:
    oracle_p, matrix_p, case_p, bundled_p = _make_synthetic_oracle_data(tmp_path)
    out_dir = tmp_path / "output"

    result = run_revise_help_feature_analysis(
        oracle_assignments_path=oracle_p,
        per_query_matrix_path=matrix_p,
        case_table_path=case_p,
        bundled_gsm8k_path=bundled_p,
        output_dir=out_dir,
    )

    # Group sizes are correct
    assert result["group_sizes"][GROUP_REVISE_HELPS] == 1
    assert result["group_sizes"][GROUP_DIRECT_ALREADY_ENOUGH] == 1

    # All 4 output files were created
    for key, path in result["output_paths"].items():
        assert Path(path).exists(), f"Output file not created: {key}"

    # Feature rates have correct structure
    assert len(result["feature_rates"]) == 5

    # Feature differences have correct structure
    assert len(result["feature_differences"]) == len(ALL_BOOL_FEATURES)

    # Recommendation is present
    assert "recommendation" in result["recommendation"]
    assert "justification" in result["recommendation"]


def test_run_revise_help_feature_analysis_all_missing(tmp_path: Path) -> None:
    """When all input files are missing, the pipeline should not crash."""
    out_dir = tmp_path / "output"
    result = run_revise_help_feature_analysis(
        oracle_assignments_path=tmp_path / "missing1.csv",
        per_query_matrix_path=tmp_path / "missing2.csv",
        case_table_path=tmp_path / "missing3.csv",
        bundled_gsm8k_path=tmp_path / "missing4.json",
        output_dir=out_dir,
    )
    assert isinstance(result["group_sizes"], dict)
    assert isinstance(result["feature_rates"], list)


def test_run_revise_help_feature_analysis_bundled_gsm8k_fallback(tmp_path: Path) -> None:
    """When no oracle data, bundled GSM8K sample provides queries."""
    bundled = [
        {"id": "gsm8k_0", "question": "She spent $5 remaining.", "answer": "#### 5"},
        {"id": "gsm8k_1", "question": "What is 2 + 2?", "answer": "#### 4"},
    ]
    bundled_path = _write_bundled_gsm8k(tmp_path, bundled)
    out_dir = tmp_path / "output"

    result = run_revise_help_feature_analysis(
        oracle_assignments_path=tmp_path / "missing1.csv",
        per_query_matrix_path=tmp_path / "missing2.csv",
        case_table_path=tmp_path / "missing3.csv",
        bundled_gsm8k_path=bundled_path,
        output_dir=out_dir,
    )
    # Bundled queries should appear in reasoning_enough (fallback group)
    assert result["group_sizes"].get(GROUP_REASONING_ENOUGH, 0) == 2
    # Query feature table should have 2 rows
    table_path = Path(result["output_paths"]["query_feature_table_csv"])
    rows = list(csv.DictReader(table_path.open()))
    assert len(rows) == 2


def test_output_files_all_created(tmp_path: Path) -> None:
    """All 4 output files are created even with no oracle data."""
    out_dir = tmp_path / "output"
    result = run_revise_help_feature_analysis(
        oracle_assignments_path=tmp_path / "m1.csv",
        per_query_matrix_path=tmp_path / "m2.csv",
        case_table_path=tmp_path / "m3.csv",
        bundled_gsm8k_path=tmp_path / "m4.json",
        output_dir=out_dir,
    )
    expected_keys = {
        "group_feature_rates_csv",
        "feature_differences_csv",
        "query_feature_table_csv",
        "example_cases_json",
    }
    assert set(result["output_paths"].keys()) == expected_keys
    for path in result["output_paths"].values():
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Integration: query_feature_table contains all feature columns
# ---------------------------------------------------------------------------


def test_query_feature_table_has_all_tq_feature_columns(tmp_path: Path) -> None:
    oracle_p, matrix_p, case_p, bundled_p = _make_synthetic_oracle_data(tmp_path)
    out_dir = tmp_path / "output"
    result = run_revise_help_feature_analysis(
        oracle_assignments_path=oracle_p,
        per_query_matrix_path=matrix_p,
        case_table_path=case_p,
        bundled_gsm8k_path=bundled_p,
        output_dir=out_dir,
    )
    table_path = Path(result["output_paths"]["query_feature_table_csv"])
    reader = csv.DictReader(table_path.open())
    headers = reader.fieldnames or []
    for feat in TARGET_QUANTITY_FEATURES:
        assert feat in headers, f"Column '{feat}' missing from query_feature_table.csv"
