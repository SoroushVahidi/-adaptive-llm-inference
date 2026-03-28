"""Unit tests for src/analysis/feature_gap_analysis.py.

All tests are fully offline — no API calls, no oracle output files required
unless explicitly synthesised in a temporary directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.analysis.feature_gap_analysis import (
    GROUP_REASONING_ENOUGH,
    GROUP_REVISE_HELPS,
    GROUP_REVISE_NOT_ENOUGH,
    assign_group,
    build_group_map,
    build_per_query_features,
    build_policy_revise_set,
    compute_group_feature_summary,
    extract_qualitative_patterns,
    find_missed_revise_cases,
    get_question_text_map,
    load_oracle_assignments,
    load_per_query_matrix,
    load_policy_results,
    run_feature_gap_analysis,
    write_group_feature_summary,
    write_missed_revise_cases,
    write_pattern_notes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_oracle_assignments(tmpdir: Path, rows: list[dict]) -> Path:
    path = tmpdir / "oracle_assignments.csv"
    if not rows:
        path.write_text("question_id,cheapest_correct_strategy,direct_greedy_correct,"
                        "direct_already_optimal\n")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_per_query_matrix(tmpdir: Path, rows: list[dict]) -> Path:
    path = tmpdir / "per_query_matrix.csv"
    if not rows:
        path.write_text("question_id,strategy,correct\n")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_policy_results(tmpdir: Path, filename: str, rows: list[dict]) -> Path:
    path = tmpdir / filename
    if not rows:
        path.write_text("question_id,strategy\n")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# assign_group — unit tests
# ---------------------------------------------------------------------------


def test_assign_group_reasoning_enough_when_direct_correct() -> None:
    assert assign_group("q0", direct_greedy_correct=1,
                        revise_correct=None, oracle_any_correct=1) == GROUP_REASONING_ENOUGH


def test_assign_group_revise_helps_when_direct_wrong_revise_correct() -> None:
    assert assign_group("q0", direct_greedy_correct=0,
                        revise_correct=1, oracle_any_correct=1) == GROUP_REVISE_HELPS


def test_assign_group_revise_not_enough_when_oracle_fails() -> None:
    assert assign_group("q0", direct_greedy_correct=0,
                        revise_correct=None, oracle_any_correct=0) == GROUP_REVISE_NOT_ENOUGH


def test_assign_group_revise_not_enough_when_revise_wrong() -> None:
    assert assign_group("q0", direct_greedy_correct=0,
                        revise_correct=0, oracle_any_correct=1) == GROUP_REVISE_NOT_ENOUGH


def test_assign_group_reasoning_enough_when_other_strategy_corrects() -> None:
    # direct wrong, revise unknown, but something else was correct
    assert assign_group("q0", direct_greedy_correct=0,
                        revise_correct=None, oracle_any_correct=1) == GROUP_REASONING_ENOUGH


# ---------------------------------------------------------------------------
# load_oracle_assignments / load_per_query_matrix / load_policy_results
# ---------------------------------------------------------------------------


def test_load_oracle_assignments_missing_file_returns_empty() -> None:
    rows = load_oracle_assignments("/nonexistent/path/oracle_assignments.csv")
    assert rows == []


def test_load_per_query_matrix_missing_file_returns_empty() -> None:
    rows = load_per_query_matrix("/nonexistent/path/matrix.csv")
    assert rows == []


def test_load_policy_results_missing_file_returns_empty() -> None:
    rows = load_policy_results("/nonexistent/path/policy.csv")
    assert rows == []


def test_load_oracle_assignments_reads_csv(tmp_path: Path) -> None:
    path = _write_oracle_assignments(tmp_path, [
        {"question_id": "q0", "cheapest_correct_strategy": "direct_greedy",
         "direct_greedy_correct": "1", "direct_already_optimal": "1"},
    ])
    rows = load_oracle_assignments(path)
    assert len(rows) == 1
    assert rows[0]["question_id"] == "q0"


# ---------------------------------------------------------------------------
# build_group_map
# ---------------------------------------------------------------------------


def test_build_group_map_basic_assignment() -> None:
    oracle_rows = [
        {"question_id": "q0", "cheapest_correct_strategy": "direct_greedy",
         "direct_greedy_correct": "1", "direct_already_optimal": "1"},
        {"question_id": "q1", "cheapest_correct_strategy": "direct_plus_revise",
         "direct_greedy_correct": "0", "direct_already_optimal": "0"},
        {"question_id": "q2", "cheapest_correct_strategy": "",
         "direct_greedy_correct": "0", "direct_already_optimal": "0"},
    ]
    matrix_rows = [
        {"question_id": "q1", "strategy": "direct_plus_revise", "correct": "1"},
        {"question_id": "q1", "strategy": "direct_greedy", "correct": "0"},
        {"question_id": "q2", "strategy": "direct_greedy", "correct": "0"},
    ]
    group_map = build_group_map(oracle_rows, matrix_rows)
    assert group_map["q0"] == GROUP_REASONING_ENOUGH
    assert group_map["q1"] == GROUP_REVISE_HELPS
    assert group_map["q2"] == GROUP_REVISE_NOT_ENOUGH


def test_build_group_map_empty_inputs() -> None:
    group_map = build_group_map([], [])
    assert group_map == {}


def test_build_group_map_skips_rows_without_question_id() -> None:
    oracle_rows = [{"question_id": "", "cheapest_correct_strategy": "",
                    "direct_greedy_correct": "0", "direct_already_optimal": "0"}]
    group_map = build_group_map(oracle_rows, [])
    assert group_map == {}


# ---------------------------------------------------------------------------
# build_per_query_features
# ---------------------------------------------------------------------------


def test_build_per_query_features_includes_question_id() -> None:
    feats = build_per_query_features("q0", "How many apples are left?")
    assert feats["question_id"] == "q0"


def test_build_per_query_features_includes_wording_trap_signals() -> None:
    feats = build_per_query_features("q0", "How many apples are left after she gave away 3?")
    assert "has_remaining_left_cue" in feats
    assert "has_subtraction_trap_verb" in feats
    assert "has_total_earned_cue" in feats
    assert "has_unit_per_cue" in feats
    assert "has_intermediate_quantity_ask" in feats


def test_build_per_query_features_remaining_cue_detected() -> None:
    feats = build_per_query_features("q0", "How many are remaining after the sale?")
    assert feats["has_remaining_left_cue"] == 1


def test_build_per_query_features_subtraction_verb_detected() -> None:
    feats = build_per_query_features("q0", "She spent $5 on apples.")
    assert feats["has_subtraction_trap_verb"] == 1


def test_build_per_query_features_per_cue_detected() -> None:
    feats = build_per_query_features("q0", "He earns 10 per day.")
    assert feats["has_unit_per_cue"] == 1


def test_build_per_query_features_no_false_positives_on_plain_question() -> None:
    feats = build_per_query_features("q0", "What is 3 plus 4?")
    assert feats["has_remaining_left_cue"] == 0
    assert feats["has_subtraction_trap_verb"] == 0
    assert feats["has_unit_per_cue"] == 0


# ---------------------------------------------------------------------------
# get_question_text_map
# ---------------------------------------------------------------------------


def test_get_question_text_map_builds_correct_map() -> None:
    rows = [
        {"question_id": "q0", "strategy": "direct_greedy",
         "question_text": "What is 2+2?", "correct": "1"},
        {"question_id": "q0", "strategy": "direct_plus_revise",
         "question_text": "What is 2+2?", "correct": "1"},
        {"question_id": "q1", "strategy": "direct_greedy",
         "question_text": "How many remain?", "correct": "0"},
    ]
    qmap = get_question_text_map(rows)
    assert qmap["q0"] == "What is 2+2?"
    assert qmap["q1"] == "How many remain?"
    assert len(qmap) == 2


# ---------------------------------------------------------------------------
# build_policy_revise_set
# ---------------------------------------------------------------------------


def test_build_policy_revise_set_detects_revise_strategy() -> None:
    rows = [
        {"question_id": "q0", "strategy": "direct_plus_revise"},
        {"question_id": "q1", "strategy": "direct_greedy"},
        {"question_id": "q2", "strategy": "direct_plus_revise"},
    ]
    revise_set = build_policy_revise_set(rows)
    assert "q0" in revise_set
    assert "q1" not in revise_set
    assert "q2" in revise_set


def test_build_policy_revise_set_empty_when_no_revise() -> None:
    rows = [
        {"question_id": "q0", "strategy": "direct_greedy"},
        {"question_id": "q1", "strategy": "reasoning_best_of_3"},
    ]
    assert build_policy_revise_set(rows) == set()


def test_build_policy_revise_set_empty_on_empty_input() -> None:
    assert build_policy_revise_set([]) == set()


# ---------------------------------------------------------------------------
# compute_group_feature_summary
# ---------------------------------------------------------------------------


def test_compute_group_feature_summary_returns_one_row_per_group() -> None:
    feats_a = [build_per_query_features(f"q{i}", "left over") for i in range(3)]
    feats_b = [build_per_query_features(f"q{i}", "What is 2+2?") for i in range(2)]
    summary = compute_group_feature_summary({
        GROUP_REVISE_HELPS: feats_a,
        GROUP_REASONING_ENOUGH: feats_b,
    })
    groups_in_summary = [r["group"] for r in summary]
    assert GROUP_REVISE_HELPS in groups_in_summary
    assert GROUP_REASONING_ENOUGH in groups_in_summary


def test_compute_group_feature_summary_n_counts_are_correct() -> None:
    feats = [build_per_query_features(f"q{i}", "test") for i in range(5)]
    summary = compute_group_feature_summary({GROUP_REVISE_HELPS: feats})
    row = summary[0]
    assert row["n"] == 5


def test_compute_group_feature_summary_empty_group() -> None:
    summary = compute_group_feature_summary({GROUP_REVISE_HELPS: []})
    assert summary[0]["n"] == 0


# ---------------------------------------------------------------------------
# find_missed_revise_cases
# ---------------------------------------------------------------------------


def test_find_missed_revise_cases_identifies_uncaught_cases() -> None:
    group_map = {
        "q0": GROUP_REVISE_HELPS,
        "q1": GROUP_REVISE_HELPS,
        "q2": GROUP_REASONING_ENOUGH,
    }
    feats = {
        "q0": build_per_query_features("q0", "How many are left?"),
        "q1": build_per_query_features("q1", "What remains after buying?"),
        "q2": build_per_query_features("q2", "What is 2+2?"),
    }
    v3 = {"q0"}  # v3 caught q0
    v4: set[str] = set()
    missed = find_missed_revise_cases(group_map, feats, v3, v4)
    assert len(missed) == 1
    assert missed[0]["question_id"] == "q1"


def test_find_missed_revise_cases_returns_empty_when_all_caught() -> None:
    group_map = {"q0": GROUP_REVISE_HELPS}
    feats = {"q0": build_per_query_features("q0", "left over")}
    missed = find_missed_revise_cases(group_map, feats, {"q0"}, set())
    assert missed == []


def test_find_missed_revise_cases_ignores_non_revise_helps_groups() -> None:
    group_map = {
        "q0": GROUP_REASONING_ENOUGH,
        "q1": GROUP_REVISE_NOT_ENOUGH,
    }
    feats = {
        "q0": build_per_query_features("q0", "What is 2+2?"),
        "q1": build_per_query_features("q1", "Hard question."),
    }
    missed = find_missed_revise_cases(group_map, feats, set(), set())
    assert missed == []


def test_find_missed_revise_cases_includes_absent_signals_field() -> None:
    group_map = {"q0": GROUP_REVISE_HELPS}
    feats = {"q0": build_per_query_features("q0", "Simple question.")}
    missed = find_missed_revise_cases(group_map, feats, set(), set())
    assert "absent_cheap_signals" in missed[0]


# ---------------------------------------------------------------------------
# extract_qualitative_patterns
# ---------------------------------------------------------------------------


def test_extract_qualitative_patterns_returns_required_keys() -> None:
    feats_rh = [build_per_query_features(f"q{i}", "How many left?") for i in range(3)]
    feats_re = [build_per_query_features(f"q{i}", "What is 2+2?") for i in range(3)]
    missed: list[dict] = []
    patterns = extract_qualitative_patterns(missed, feats_rh, feats_re)
    assert "data_summary" in patterns
    assert "wording_trap_feature_gaps" in patterns
    assert "qualitative_patterns" in patterns
    assert "current_feature_failures" in patterns
    assert "candidate_next_signals" in patterns
    assert "suggested_direction" in patterns


def test_extract_qualitative_patterns_data_summary_counts_correct() -> None:
    feats_rh = [build_per_query_features(f"q{i}", "left over") for i in range(4)]
    feats_re = [build_per_query_features(f"q{i}", "What is 2+2?") for i in range(6)]
    missed = [{"question_id": "q0", "absent_cheap_signals": "has_remaining_left_cue"}]
    patterns = extract_qualitative_patterns(missed, feats_rh, feats_re)
    ds = patterns["data_summary"]
    assert ds["n_revise_helps"] == 4
    assert ds["n_reasoning_enough"] == 6
    assert ds["n_missed_revise_cases"] == 1


def test_extract_qualitative_patterns_candidate_signals_has_five() -> None:
    patterns = extract_qualitative_patterns([], [], [])
    assert len(patterns["candidate_next_signals"]) == 5


def test_extract_qualitative_patterns_no_data_returns_safely() -> None:
    """Should not raise when all inputs are empty."""
    patterns = extract_qualitative_patterns([], [], [])
    assert patterns["data_summary"]["n_revise_helps"] == 0


# ---------------------------------------------------------------------------
# write_* helpers
# ---------------------------------------------------------------------------


def test_write_group_feature_summary_creates_file(tmp_path: Path) -> None:
    feats = [build_per_query_features(f"q{i}", "left") for i in range(2)]
    summary = compute_group_feature_summary({GROUP_REVISE_HELPS: feats})
    path = write_group_feature_summary(summary, tmp_path)
    assert Path(path).exists()
    rows = list(csv.DictReader(Path(path).open()))
    assert len(rows) == 1
    assert rows[0]["group"] == GROUP_REVISE_HELPS


def test_write_group_feature_summary_empty_rows(tmp_path: Path) -> None:
    path = write_group_feature_summary([], tmp_path)
    assert Path(path).exists()


def test_write_missed_revise_cases_creates_file(tmp_path: Path) -> None:
    cases = [
        {
            "question_id": "q0",
            "group": GROUP_REVISE_HELPS,
            "v3_triggered_revise": 0,
            "v4_triggered_revise": 0,
            "absent_cheap_signals": "has_remaining_left_cue",
        }
    ]
    path = write_missed_revise_cases(cases, tmp_path)
    assert Path(path).exists()
    rows = list(csv.DictReader(Path(path).open()))
    assert rows[0]["question_id"] == "q0"


def test_write_missed_revise_cases_empty(tmp_path: Path) -> None:
    path = write_missed_revise_cases([], tmp_path)
    assert Path(path).exists()


def test_write_pattern_notes_creates_valid_json(tmp_path: Path) -> None:
    notes = {"data_summary": {"n_revise_helps": 3}}
    path = write_pattern_notes(notes, tmp_path)
    assert Path(path).exists()
    loaded = json.loads(Path(path).read_text())
    assert loaded["data_summary"]["n_revise_helps"] == 3


# ---------------------------------------------------------------------------
# run_feature_gap_analysis — integration test
# ---------------------------------------------------------------------------


def _make_full_fixtures(tmpdir: Path) -> dict[str, Path]:
    """Write a minimal but complete set of input fixtures."""
    # oracle_assignments.csv
    oa_rows = [
        {"question_id": "q0", "cheapest_correct_strategy": "direct_greedy",
         "direct_greedy_correct": "1", "direct_already_optimal": "1"},
        {"question_id": "q1", "cheapest_correct_strategy": "direct_plus_revise",
         "direct_greedy_correct": "0", "direct_already_optimal": "0"},
        {"question_id": "q2", "cheapest_correct_strategy": "",
         "direct_greedy_correct": "0", "direct_already_optimal": "0"},
        {"question_id": "q3", "cheapest_correct_strategy": "direct_plus_revise",
         "direct_greedy_correct": "0", "direct_already_optimal": "0"},
    ]
    oa_path = _write_oracle_assignments(tmpdir, oa_rows)

    # per_query_matrix.csv
    matrix_rows = [
        {"question_id": "q0", "strategy": "direct_greedy",
         "question_text": "What is 2+2?", "correct": "1"},
        {"question_id": "q1", "strategy": "direct_greedy",
         "question_text": "How many apples are remaining after she sold 3?", "correct": "0"},
        {"question_id": "q1", "strategy": "direct_plus_revise",
         "question_text": "How many apples are remaining after she sold 3?", "correct": "1"},
        {"question_id": "q2", "strategy": "direct_greedy",
         "question_text": "Very hard question.", "correct": "0"},
        {"question_id": "q3", "strategy": "direct_greedy",
         "question_text": "She spent $5 per day. How much left?", "correct": "0"},
        {"question_id": "q3", "strategy": "direct_plus_revise",
         "question_text": "She spent $5 per day. How much left?", "correct": "1"},
    ]
    mx_path = _write_per_query_matrix(tmpdir, matrix_rows)

    # v3 policy: only caught q1
    v3_rows = [
        {"question_id": "q1", "strategy": "direct_plus_revise"},
        {"question_id": "q0", "strategy": "direct_greedy"},
    ]
    v3_path = _write_policy_results(tmpdir, "v3_results.csv", v3_rows)

    # v4 policy: caught neither q1 nor q3
    v4_rows = [
        {"question_id": "q0", "strategy": "direct_greedy"},
        {"question_id": "q2", "strategy": "direct_greedy"},
    ]
    v4_path = _write_policy_results(tmpdir, "v4_results.csv", v4_rows)

    out_dir = tmpdir / "out"
    return {
        "oa": oa_path,
        "mx": mx_path,
        "v3": v3_path,
        "v4": v4_path,
        "out": out_dir,
    }


def test_run_feature_gap_analysis_returns_required_keys(tmp_path: Path) -> None:
    paths = _make_full_fixtures(tmp_path)
    result = run_feature_gap_analysis(
        oracle_assignments_path=paths["oa"],
        per_query_matrix_path=paths["mx"],
        case_table_path="/nonexistent",
        category_summary_path="/nonexistent",
        v3_results_path=paths["v3"],
        v4_results_path=paths["v4"],
        output_dir=paths["out"],
    )
    for key in ("group_sizes", "n_missed_revise_cases", "output_paths",
                "pattern_notes", "summary_rows", "missed_cases"):
        assert key in result, f"Missing key: {key}"


def test_run_feature_gap_analysis_group_sizes(tmp_path: Path) -> None:
    paths = _make_full_fixtures(tmp_path)
    result = run_feature_gap_analysis(
        oracle_assignments_path=paths["oa"],
        per_query_matrix_path=paths["mx"],
        case_table_path="/nonexistent",
        category_summary_path="/nonexistent",
        v3_results_path=paths["v3"],
        v4_results_path=paths["v4"],
        output_dir=paths["out"],
    )
    gs = result["group_sizes"]
    # q0 → reasoning_enough, q1 → revise_helps, q2 → revise_not_enough, q3 → revise_helps
    assert gs[GROUP_REASONING_ENOUGH] == 1
    assert gs[GROUP_REVISE_HELPS] == 2
    assert gs[GROUP_REVISE_NOT_ENOUGH] == 1


def test_run_feature_gap_analysis_missed_cases(tmp_path: Path) -> None:
    """q3 is revise_helps but neither v3 nor v4 triggered revise for it."""
    paths = _make_full_fixtures(tmp_path)
    result = run_feature_gap_analysis(
        oracle_assignments_path=paths["oa"],
        per_query_matrix_path=paths["mx"],
        case_table_path="/nonexistent",
        category_summary_path="/nonexistent",
        v3_results_path=paths["v3"],
        v4_results_path=paths["v4"],
        output_dir=paths["out"],
    )
    assert result["n_missed_revise_cases"] == 1
    assert result["missed_cases"][0]["question_id"] == "q3"


def test_run_feature_gap_analysis_all_outputs_exist(tmp_path: Path) -> None:
    paths = _make_full_fixtures(tmp_path)
    result = run_feature_gap_analysis(
        oracle_assignments_path=paths["oa"],
        per_query_matrix_path=paths["mx"],
        case_table_path="/nonexistent",
        category_summary_path="/nonexistent",
        v3_results_path=paths["v3"],
        v4_results_path=paths["v4"],
        output_dir=paths["out"],
    )
    for key, path in result["output_paths"].items():
        assert Path(path).exists(), f"Output not created: {key} → {path}"


def test_run_feature_gap_analysis_all_missing_inputs_runs_without_error(tmp_path: Path) -> None:
    """Should complete gracefully when no input files are present."""
    result = run_feature_gap_analysis(
        oracle_assignments_path="/nonexistent/a.csv",
        per_query_matrix_path="/nonexistent/b.csv",
        case_table_path="/nonexistent/c.csv",
        category_summary_path="/nonexistent/d.csv",
        v3_results_path="/nonexistent/e.csv",
        v4_results_path="/nonexistent/f.csv",
        output_dir=tmp_path / "out",
    )
    assert result["group_sizes"][GROUP_REVISE_HELPS] == 0
    assert result["n_missed_revise_cases"] == 0


def test_run_feature_gap_analysis_output_csvs_are_valid(tmp_path: Path) -> None:
    paths = _make_full_fixtures(tmp_path)
    result = run_feature_gap_analysis(
        oracle_assignments_path=paths["oa"],
        per_query_matrix_path=paths["mx"],
        case_table_path="/nonexistent",
        category_summary_path="/nonexistent",
        v3_results_path=paths["v3"],
        v4_results_path=paths["v4"],
        output_dir=paths["out"],
    )
    # group_feature_summary.csv should have one row per group
    summary_path = result["output_paths"]["group_feature_summary_csv"]
    rows = list(csv.DictReader(Path(summary_path).open()))
    groups_found = {r["group"] for r in rows}
    assert GROUP_REVISE_HELPS in groups_found
    assert GROUP_REASONING_ENOUGH in groups_found

    # pattern_notes.json should be valid JSON with required keys
    notes_path = result["output_paths"]["pattern_notes_json"]
    notes = json.loads(Path(notes_path).read_text())
    assert "candidate_next_signals" in notes
    assert len(notes["candidate_next_signals"]) == 5
