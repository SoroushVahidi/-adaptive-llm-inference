"""Unit tests for src/datasets/routing_dataset.py.

All tests are fully offline — no API calls, no file-system oracle outputs
required.  Oracle files are synthesised in temporary directories when needed.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from src.datasets.routing_dataset import (
    FIRST_PASS_COLUMNS,
    ORACLE_LABEL_COLUMNS,
    OracleData,
    assemble_routing_dataset,
    load_oracle_files,
    save_routing_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeQuery:
    """Mimics src.datasets.gsm8k.Query."""

    def __init__(self, qid: str, question: str, answer: str = "42") -> None:
        self.id = qid
        self.question = question
        self.answer = answer


def _make_queries(n: int = 3) -> list[_FakeQuery]:
    return [_FakeQuery(f"q{i}", f"What is {i} + 1?") for i in range(n)]


def _write_oracle_csvs(
    tmpdir: str | Path,
    assignments: list[dict],
    matrix_rows: list[dict] | None = None,
) -> None:
    """Write minimal oracle CSV fixtures to *tmpdir*."""
    base = Path(tmpdir)

    # oracle_assignments.csv
    assign_path = base / "oracle_assignments.csv"
    if assignments:
        fieldnames = list(assignments[0].keys())
        with assign_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(assignments)

    # per_query_matrix.csv
    if matrix_rows:
        matrix_path = base / "per_query_matrix.csv"
        fieldnames_m = list(matrix_rows[0].keys())
        with matrix_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames_m)
            w.writeheader()
            w.writerows(matrix_rows)


_SAMPLE_ASSIGNMENTS = [
    {
        "question_id": "q0",
        "any_correct": "1",
        "cheapest_correct_strategy": "direct_greedy",
        "direct_already_optimal": "1",
        "best_accuracy_strategies": "direct_greedy|reasoning_best_of_3",
    },
    {
        "question_id": "q1",
        "any_correct": "1",
        "cheapest_correct_strategy": "reasoning_best_of_3",
        "direct_already_optimal": "0",
        "best_accuracy_strategies": "reasoning_best_of_3",
    },
    {
        "question_id": "q2",
        "any_correct": "0",
        "cheapest_correct_strategy": "",
        "direct_already_optimal": "0",
        "best_accuracy_strategies": "",
    },
]

_SAMPLE_MATRIX = [
    {"question_id": "q0", "strategy": "direct_greedy", "correct": "1"},
    {"question_id": "q0", "strategy": "reasoning_best_of_3", "correct": "1"},
    {"question_id": "q1", "strategy": "direct_greedy", "correct": "0"},
    {"question_id": "q1", "strategy": "reasoning_best_of_3", "correct": "1"},
    {"question_id": "q2", "strategy": "direct_greedy", "correct": "0"},
    {"question_id": "q2", "strategy": "reasoning_best_of_3", "correct": "0"},
]


# ---------------------------------------------------------------------------
# load_oracle_files
# ---------------------------------------------------------------------------


def test_load_oracle_files_missing_dir_returns_empty_oracle_data() -> None:
    missing = Path(tempfile.gettempdir()) / "routing_dataset_test_does_not_exist_xyz"
    data = load_oracle_files(missing)
    assert not data.available
    assert len(data.missing_files) == 2  # both CSVs missing


def test_load_oracle_files_with_valid_csvs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_oracle_csvs(tmpdir, _SAMPLE_ASSIGNMENTS, _SAMPLE_MATRIX)
        data = load_oracle_files(tmpdir)

    assert data.available
    assert "q0" in data.assignments
    assert "q1" in data.assignments
    assert len(data.source_files) == 2


def test_load_oracle_files_missing_matrix_still_loads_assignments() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_oracle_csvs(tmpdir, _SAMPLE_ASSIGNMENTS)  # no matrix
        data = load_oracle_files(tmpdir)

    assert data.available  # assignments loaded
    assert any("per_query_matrix" in f for f in data.missing_files)


def test_load_oracle_strategy_correct_counts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_oracle_csvs(tmpdir, _SAMPLE_ASSIGNMENTS, _SAMPLE_MATRIX)
        data = load_oracle_files(tmpdir)

    assert data.strategy_correct_counts["q0"] == 2
    assert data.strategy_correct_counts["q1"] == 1
    assert data.strategy_correct_counts["q2"] == 0


# ---------------------------------------------------------------------------
# assemble_routing_dataset — schema
# ---------------------------------------------------------------------------


def test_assemble_returns_one_row_per_query() -> None:
    queries = _make_queries(5)
    rows = assemble_routing_dataset(queries)
    assert len(rows) == 5


def test_assemble_no_duplicate_question_ids() -> None:
    queries = _make_queries(4)
    rows = assemble_routing_dataset(queries)
    ids = [r["question_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_assemble_contains_required_columns() -> None:
    rows = assemble_routing_dataset(_make_queries(1))
    row = rows[0]
    required = [
        "question_id",
        "question_text",
        "question_length_chars",
        "question_length_tokens_approx",
        "num_numeric_mentions",
        "num_sentences_approx",
        "has_multi_step_cue",
        "has_equation_like_pattern",
        "has_percent_symbol",
        "has_fraction_pattern",
        "has_currency_symbol",
        "max_numeric_value_approx",
        "min_numeric_value_approx",
        "numeric_range_approx",
        "repeated_number_flag",
        "oracle_label_available",
    ]
    for col in required:
        assert col in row, f"Missing column: {col}"


def test_assemble_oracle_label_columns_present() -> None:
    rows = assemble_routing_dataset(_make_queries(1))
    row = rows[0]
    for col in ORACLE_LABEL_COLUMNS:
        assert col in row, f"Missing oracle column: {col}"


def test_assemble_first_pass_columns_present() -> None:
    rows = assemble_routing_dataset(_make_queries(1))
    row = rows[0]
    for col in FIRST_PASS_COLUMNS:
        assert col in row, f"Missing first-pass column: {col}"


# ---------------------------------------------------------------------------
# assemble_routing_dataset — schema-only mode
# ---------------------------------------------------------------------------


def test_assemble_schema_only_when_no_oracle() -> None:
    rows = assemble_routing_dataset(_make_queries(3), oracle_data=None)
    for row in rows:
        assert row["oracle_label_available"] is False


def test_assemble_schema_only_oracle_columns_empty() -> None:
    rows = assemble_routing_dataset(_make_queries(2), oracle_data=None)
    for row in rows:
        for col in ORACLE_LABEL_COLUMNS:
            assert row[col] == "", f"Expected empty string for {col} in schema-only mode"


def test_assemble_schema_only_with_empty_oracle_data() -> None:
    empty = OracleData()
    rows = assemble_routing_dataset(_make_queries(2), oracle_data=empty)
    for row in rows:
        assert row["oracle_label_available"] is False


# ---------------------------------------------------------------------------
# assemble_routing_dataset — full mode with oracle labels
# ---------------------------------------------------------------------------


def test_assemble_full_mode_oracle_label_available_true() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_oracle_csvs(tmpdir, _SAMPLE_ASSIGNMENTS, _SAMPLE_MATRIX)
        oracle_data = load_oracle_files(tmpdir)

    queries = _make_queries(3)
    rows = assemble_routing_dataset(queries, oracle_data=oracle_data)
    for row in rows:
        assert row["oracle_label_available"] is True


def test_assemble_full_mode_cheapest_correct_strategy() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_oracle_csvs(tmpdir, _SAMPLE_ASSIGNMENTS, _SAMPLE_MATRIX)
        oracle_data = load_oracle_files(tmpdir)

    rows = assemble_routing_dataset(_make_queries(3), oracle_data=oracle_data)
    by_id = {r["question_id"]: r for r in rows}
    assert by_id["q0"]["cheapest_correct_strategy"] == "direct_greedy"
    assert by_id["q1"]["cheapest_correct_strategy"] == "reasoning_best_of_3"
    assert by_id["q2"]["cheapest_correct_strategy"] == ""


def test_assemble_full_mode_best_accuracy_strategy_first_token() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_oracle_csvs(tmpdir, _SAMPLE_ASSIGNMENTS, _SAMPLE_MATRIX)
        oracle_data = load_oracle_files(tmpdir)

    rows = assemble_routing_dataset(_make_queries(3), oracle_data=oracle_data)
    by_id = {r["question_id"]: r for r in rows}
    assert by_id["q0"]["best_accuracy_strategy"] == "direct_greedy"


def test_assemble_full_mode_direct_already_optimal() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_oracle_csvs(tmpdir, _SAMPLE_ASSIGNMENTS, _SAMPLE_MATRIX)
        oracle_data = load_oracle_files(tmpdir)

    rows = assemble_routing_dataset(_make_queries(3), oracle_data=oracle_data)
    by_id = {r["question_id"]: r for r in rows}
    assert by_id["q0"]["direct_already_optimal"] == 1
    assert by_id["q1"]["direct_already_optimal"] == 0


def test_assemble_full_mode_num_strategies_correct() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_oracle_csvs(tmpdir, _SAMPLE_ASSIGNMENTS, _SAMPLE_MATRIX)
        oracle_data = load_oracle_files(tmpdir)

    rows = assemble_routing_dataset(_make_queries(3), oracle_data=oracle_data)
    by_id = {r["question_id"]: r for r in rows}
    assert by_id["q0"]["num_strategies_correct"] == 2
    assert by_id["q1"]["num_strategies_correct"] == 1
    assert by_id["q2"]["num_strategies_correct"] == 0


# ---------------------------------------------------------------------------
# assemble_routing_dataset — first-pass features
# ---------------------------------------------------------------------------


def test_assemble_first_pass_features_merged() -> None:
    queries = [_FakeQuery("q0", "How much?")]
    fp = {
        "q0": {
            "first_pass_parse_success": True,
            "first_pass_output_length": 20,
            "first_pass_has_final_answer_cue": True,
            "first_pass_has_uncertainty_phrase": False,
            "first_pass_num_numeric_mentions": 1,
            "first_pass_empty_or_malformed_flag": False,
        }
    }
    rows = assemble_routing_dataset(queries, first_pass_rows=fp)
    assert rows[0]["first_pass_parse_success"] is True
    assert rows[0]["first_pass_output_length"] == 20


def test_assemble_first_pass_absent_columns_empty() -> None:
    rows = assemble_routing_dataset(_make_queries(1))
    for col in FIRST_PASS_COLUMNS:
        assert rows[0][col] == ""


# ---------------------------------------------------------------------------
# assemble_routing_dataset — dict query input
# ---------------------------------------------------------------------------


def test_assemble_accepts_dict_queries() -> None:
    queries = [
        {"question_id": "a", "question_text": "What is 2 + 2?"},
        {"question_id": "b", "question_text": "Find x if x + 3 = 7."},
    ]
    rows = assemble_routing_dataset(queries)
    assert len(rows) == 2
    assert rows[0]["question_id"] == "a"


# ---------------------------------------------------------------------------
# save_routing_dataset
# ---------------------------------------------------------------------------


def test_save_creates_csv_and_summary() -> None:
    rows = assemble_routing_dataset(_make_queries(3))
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_routing_dataset(rows, tmpdir)
        assert Path(paths["csv_path"]).exists()
        assert Path(paths["summary_path"]).exists()


def test_save_csv_has_expected_columns() -> None:
    rows = assemble_routing_dataset(_make_queries(2))
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_routing_dataset(rows, tmpdir)
        with open(paths["csv_path"]) as fh:
            header = (csv.DictReader(fh).fieldnames or [])
    assert "question_id" in header
    assert "question_text" in header
    assert "oracle_label_available" in header
    for col in ORACLE_LABEL_COLUMNS:
        assert col in header


def test_save_no_duplicate_question_ids_in_csv() -> None:
    rows = assemble_routing_dataset(_make_queries(4))
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_routing_dataset(rows, tmpdir)
        with open(paths["csv_path"]) as fh:
            read_rows = list(csv.DictReader(fh))
    ids = [r["question_id"] for r in read_rows]
    assert len(ids) == len(set(ids))


def test_save_summary_json_keys() -> None:
    rows = assemble_routing_dataset(_make_queries(3))
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_routing_dataset(rows, tmpdir)
        summary = json.loads(Path(paths["summary_path"]).read_text())

    required_keys = [
        "num_queries",
        "oracle_labels_available",
        "num_feature_columns",
        "num_label_columns",
        "source_files",
        "missing_optional_inputs",
    ]
    for key in required_keys:
        assert key in summary, f"Missing key in summary: {key}"


def test_save_summary_num_queries_correct() -> None:
    n = 5
    rows = assemble_routing_dataset(_make_queries(n))
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_routing_dataset(rows, tmpdir)
        summary = json.loads(Path(paths["summary_path"]).read_text())
    assert summary["num_queries"] == n


def test_save_summary_oracle_labels_available_false_when_no_oracle() -> None:
    rows = assemble_routing_dataset(_make_queries(3))
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_routing_dataset(rows, tmpdir, oracle_data=None)
        summary = json.loads(Path(paths["summary_path"]).read_text())
    assert summary["oracle_labels_available"] is False


def test_save_summary_oracle_labels_available_true_when_oracle_loaded() -> None:
    with tempfile.TemporaryDirectory() as oracledir:
        _write_oracle_csvs(oracledir, _SAMPLE_ASSIGNMENTS, _SAMPLE_MATRIX)
        oracle_data = load_oracle_files(oracledir)

    rows = assemble_routing_dataset(_make_queries(3), oracle_data=oracle_data)
    with tempfile.TemporaryDirectory() as outdir:
        paths = save_routing_dataset(rows, outdir, oracle_data=oracle_data)
        summary = json.loads(Path(paths["summary_path"]).read_text())
    assert summary["oracle_labels_available"] is True


def test_save_empty_rows() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_routing_dataset([], tmpdir)
        assert Path(paths["csv_path"]).exists()
        assert Path(paths["summary_path"]).exists()
        summary = json.loads(Path(paths["summary_path"]).read_text())
        assert summary["num_queries"] == 0
