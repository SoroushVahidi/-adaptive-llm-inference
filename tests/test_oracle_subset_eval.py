"""Unit tests for oracle_subset_eval.py.

All tests use lightweight fake models so they run completely offline without
any API key or network access.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.oracle_subset_eval import (
    CORE_ORACLE_STRATEGIES,
    STRATEGY_COST_PROXY,
    compute_oracle_summaries,
    compute_pairwise_win_matrix,
    format_oracle_summary,
    run_oracle_subset_eval,
    write_oracle_outputs,
)

# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------


class _FakeModelFixed:
    """Always returns the same answer."""

    def __init__(self, answer: str = "42") -> None:
        self.answer = answer

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return f"Final answer: {self.answer}"

    def generate_n(self, prompt: str, n: int) -> list[str]:  # noqa: ARG002
        return [f"Final answer: {self.answer}"] * n


class _FakeModelSequenced:
    """Returns responses from a fixed sequence, cycling when exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self._idx = 0

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        r = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return r

    def generate_n(self, prompt: str, n: int) -> list[str]:  # noqa: ARG002
        out = []
        for _ in range(n):
            out.append(self.responses[self._idx % len(self.responses)])
            self._idx += 1
        return out


class _FakeQuery:
    def __init__(self, qid: str, question: str, answer: str) -> None:
        self.id = qid
        self.question = question
        self.answer = answer


def _make_queries(n: int = 3, base_answer: str = "5") -> list[_FakeQuery]:
    return [_FakeQuery(f"q{i}", f"question {i}", base_answer) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests: run_oracle_subset_eval — basic structure
# ---------------------------------------------------------------------------


def test_eval_returns_required_keys() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2, "5")
    result = run_oracle_subset_eval(model, queries, strategies=["direct_greedy"])

    assert "per_query_rows" in result
    assert "strategies_run" in result
    assert "query_ids" in result


def test_eval_per_query_row_has_required_fields() -> None:
    model = _FakeModelFixed("3")
    queries = [_FakeQuery("q0", "What is 1+2?", "3")]
    result = run_oracle_subset_eval(model, queries, strategies=["direct_greedy"])

    row = result["per_query_rows"][0]
    for field in (
        "question_id",
        "strategy",
        "predicted_answer",
        "gold_answer",
        "correct",
        "samples_used",
        "cost_proxy",
    ):
        assert field in row, f"Missing field: {field}"


def test_eval_no_duplicate_strategy_rows_per_query() -> None:
    model = _FakeModelFixed("7")
    queries = _make_queries(3, "7")
    strategies = ["direct_greedy", "reasoning_best_of_3"]
    result = run_oracle_subset_eval(model, queries, strategies=strategies)

    seen: set[tuple[str, str]] = set()
    for row in result["per_query_rows"]:
        key = (row["question_id"], row["strategy"])
        assert key not in seen, f"Duplicate row for {key}"
        seen.add(key)


def test_eval_query_ids_match_input() -> None:
    model = _FakeModelFixed("1")
    queries = [_FakeQuery("q_a", "q?", "1"), _FakeQuery("q_b", "q?", "1")]
    result = run_oracle_subset_eval(model, queries, strategies=["direct_greedy"])
    assert result["query_ids"] == ["q_a", "q_b"]


def test_eval_unknown_strategy_raises() -> None:
    model = _FakeModelFixed("1")
    queries = _make_queries(1)
    with pytest.raises(ValueError, match="Unknown oracle strategies"):
        run_oracle_subset_eval(model, queries, strategies=["not_a_real_strategy"])


def test_eval_correct_field_is_int() -> None:
    model = _FakeModelFixed("5")
    queries = [_FakeQuery("q0", "q?", "5")]
    result = run_oracle_subset_eval(model, queries, strategies=["direct_greedy"])
    assert isinstance(result["per_query_rows"][0]["correct"], int)


def test_eval_strong_direct_excluded_when_no_strong_model() -> None:
    """strong_direct must be silently removed if no strong_model is passed."""
    model = _FakeModelFixed("5")
    queries = _make_queries(2, "5")
    result = run_oracle_subset_eval(
        model, queries,
        strategies=["direct_greedy", "strong_direct"],
        strong_model=None,
    )
    assert "strong_direct" not in result["strategies_run"]
    assert "direct_greedy" in result["strategies_run"]


def test_eval_strong_direct_included_when_strong_model_provided() -> None:
    primary = _FakeModelFixed("5")
    strong = _FakeModelFixed("5")
    queries = _make_queries(2, "5")
    result = run_oracle_subset_eval(
        primary, queries,
        strategies=["direct_greedy", "strong_direct"],
        strong_model=strong,
    )
    assert "strong_direct" in result["strategies_run"]


# ---------------------------------------------------------------------------
# Tests: compute_oracle_summaries — oracle assignment logic
# ---------------------------------------------------------------------------


def _build_rows(
    queries: list[_FakeQuery],
    strategy_results: dict[str, str],
) -> list[dict]:
    """Helper: build per_query_rows with explicit strategy→answer mapping."""
    rows = []
    for q in queries:
        for strategy, answer in strategy_results.items():
            rows.append({
                "question_id": q.id,
                "strategy": strategy,
                "predicted_answer": answer,
                "gold_answer": q.answer,
                "correct": int(answer == q.answer),
                "samples_used": STRATEGY_COST_PROXY.get(strategy, 1),
                "cost_proxy": STRATEGY_COST_PROXY.get(strategy, 1),
            })
    return rows


def test_oracle_accuracy_all_strategies_wrong() -> None:
    queries = [_FakeQuery("q0", "q?", "5")]
    rows = _build_rows(queries, {"direct_greedy": "99", "reasoning_best_of_3": "88"})
    summaries = compute_oracle_summaries(rows, ["direct_greedy", "reasoning_best_of_3"])
    assert summaries["oracle_accuracy"] == 0.0
    assert summaries["oracle_correct"] == 0


def test_oracle_accuracy_at_least_one_correct() -> None:
    queries = [_FakeQuery("q0", "q?", "5")]
    rows = _build_rows(queries, {"direct_greedy": "99", "reasoning_best_of_3": "5"})
    summaries = compute_oracle_summaries(rows, ["direct_greedy", "reasoning_best_of_3"])
    assert summaries["oracle_accuracy"] == 1.0
    assert summaries["oracle_correct"] == 1


def test_oracle_minus_direct_gap() -> None:
    # direct wrong, reasoning correct → gap should be positive
    queries = [_FakeQuery("q0", "q?", "5")]
    rows = _build_rows(queries, {"direct_greedy": "1", "reasoning_best_of_3": "5"})
    summaries = compute_oracle_summaries(rows, ["direct_greedy", "reasoning_best_of_3"])
    assert summaries["oracle_minus_direct_gap"] == pytest.approx(1.0)


def test_cheapest_correct_strategy_prefers_lower_cost() -> None:
    """direct_greedy (cost=1) should be chosen over reasoning_best_of_3 (cost=3)."""
    queries = [_FakeQuery("q0", "q?", "5")]
    rows = _build_rows(queries, {"direct_greedy": "5", "reasoning_best_of_3": "5"})
    summaries = compute_oracle_summaries(rows, ["direct_greedy", "reasoning_best_of_3"])
    oracle_rec = summaries["per_query_oracle"][0]
    assert oracle_rec["cheapest_correct_strategy"] == "direct_greedy"


def test_cheapest_correct_strategy_none_when_all_wrong() -> None:
    queries = [_FakeQuery("q0", "q?", "5")]
    rows = _build_rows(queries, {"direct_greedy": "1"})
    summaries = compute_oracle_summaries(rows, ["direct_greedy"])
    oracle_rec = summaries["per_query_oracle"][0]
    assert oracle_rec["cheapest_correct_strategy"] == ""


def test_direct_already_optimal_true_when_direct_cheapest_correct() -> None:
    queries = [_FakeQuery("q0", "q?", "5")]
    rows = _build_rows(queries, {"direct_greedy": "5", "reasoning_best_of_3": "5"})
    summaries = compute_oracle_summaries(rows, ["direct_greedy", "reasoning_best_of_3"])
    oracle_rec = summaries["per_query_oracle"][0]
    assert oracle_rec["direct_already_optimal"] == 1


def test_direct_already_optimal_false_when_direct_wrong() -> None:
    queries = [_FakeQuery("q0", "q?", "5")]
    rows = _build_rows(queries, {"direct_greedy": "1", "reasoning_best_of_3": "5"})
    summaries = compute_oracle_summaries(rows, ["direct_greedy", "reasoning_best_of_3"])
    oracle_rec = summaries["per_query_oracle"][0]
    assert oracle_rec["direct_already_optimal"] == 0


def test_fixes_direct_greedy_counts_only_when_direct_wrong_and_strategy_correct() -> None:
    """reasoning_best_of_3 fixes 1 query where direct_greedy was wrong and it was correct."""
    rows = [
        # q0: direct wrong, reasoning correct
        {"question_id": "q0", "strategy": "direct_greedy",
         "predicted_answer": "1", "gold_answer": "5",
         "correct": 0, "samples_used": 1, "cost_proxy": 1},
        {"question_id": "q0", "strategy": "reasoning_best_of_3",
         "predicted_answer": "5", "gold_answer": "5",
         "correct": 1, "samples_used": 3, "cost_proxy": 3},
        # q1: both correct
        {"question_id": "q1", "strategy": "direct_greedy",
         "predicted_answer": "10", "gold_answer": "10",
         "correct": 1, "samples_used": 1, "cost_proxy": 1},
        {"question_id": "q1", "strategy": "reasoning_best_of_3",
         "predicted_answer": "10", "gold_answer": "10",
         "correct": 1, "samples_used": 3, "cost_proxy": 3},
    ]
    summaries = compute_oracle_summaries(rows, ["direct_greedy", "reasoning_best_of_3"])
    assert summaries["fixes_direct_greedy"].get("reasoning_best_of_3", 0) == 1
    assert summaries["fixes_direct_greedy"].get("direct_greedy", 0) == 0


def test_strategy_accuracy_computed_correctly() -> None:
    queries = [_FakeQuery(f"q{i}", "q?", "5") for i in range(4)]
    rows = []
    # direct_greedy correct on 3 of 4
    answers_dg = ["5", "5", "5", "99"]
    for q, ans in zip(queries, answers_dg):
        rows.append({
            "question_id": q.id,
            "strategy": "direct_greedy",
            "predicted_answer": ans,
            "gold_answer": "5",
            "correct": int(ans == "5"),
            "samples_used": 1,
            "cost_proxy": 1,
        })
    summaries = compute_oracle_summaries(rows, ["direct_greedy"])
    assert summaries["strategy_accuracy"]["direct_greedy"]["accuracy"] == pytest.approx(0.75)
    assert summaries["strategy_accuracy"]["direct_greedy"]["correct"] == 3


# ---------------------------------------------------------------------------
# Tests: compute_pairwise_win_matrix
# ---------------------------------------------------------------------------


def test_pairwise_win_matrix_shape() -> None:
    queries = [_FakeQuery("q0", "q?", "5")]
    strategies = ["direct_greedy", "reasoning_best_of_3"]
    rows = _build_rows(queries, {"direct_greedy": "5", "reasoning_best_of_3": "1"})
    pw = compute_pairwise_win_matrix(rows, strategies)
    assert set(pw["matrix"].keys()) == set(strategies)
    for s in strategies:
        assert set(pw["matrix"][s].keys()) == set(strategies)


def test_pairwise_win_matrix_values() -> None:
    """direct_greedy correct, reasoning wrong → direct beats reasoning once."""
    queries = [_FakeQuery("q0", "q?", "5")]
    rows = _build_rows(queries, {"direct_greedy": "5", "reasoning_best_of_3": "1"})
    pw = compute_pairwise_win_matrix(rows, ["direct_greedy", "reasoning_best_of_3"])
    assert pw["matrix"]["direct_greedy"]["reasoning_best_of_3"] == 1
    assert pw["matrix"]["reasoning_best_of_3"]["direct_greedy"] == 0


def test_pairwise_diagonal_is_zero() -> None:
    queries = [_FakeQuery("q0", "q?", "5")]
    rows = _build_rows(queries, {"direct_greedy": "5"})
    pw = compute_pairwise_win_matrix(rows, ["direct_greedy"])
    assert pw["matrix"]["direct_greedy"]["direct_greedy"] == 0


# ---------------------------------------------------------------------------
# Tests: write_oracle_outputs — file creation
# ---------------------------------------------------------------------------


def test_write_outputs_creates_all_five_files() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2, "5")
    strategies = ["direct_greedy", "direct_plus_verify"]
    eval_result = run_oracle_subset_eval(model, queries, strategies=strategies)
    oracle_summaries = compute_oracle_summaries(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )
    pairwise = compute_pairwise_win_matrix(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = write_oracle_outputs(eval_result, oracle_summaries, pairwise, tmpdir)

        for key in (
            "per_query_matrix_csv",
            "summary_json",
            "summary_csv",
            "oracle_assignments_csv",
            "pairwise_win_matrix_csv",
        ):
            assert key in paths, f"Missing key: {key}"
            assert Path(paths[key]).exists(), f"File not found: {paths[key]}"


def test_summary_json_has_required_keys() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2, "5")
    eval_result = run_oracle_subset_eval(model, queries, strategies=["direct_greedy"])
    oracle_summaries = compute_oracle_summaries(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )
    pairwise = compute_pairwise_win_matrix(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = write_oracle_outputs(eval_result, oracle_summaries, pairwise, tmpdir)
        payload = json.loads(Path(paths["summary_json"]).read_text())

    for key in (
        "total_queries",
        "strategies_run",
        "strategy_accuracy",
        "oracle_accuracy",
        "oracle_minus_direct_gap",
    ):
        assert key in payload, f"Missing key in summary.json: {key}"


def test_oracle_assignments_csv_has_cheapest_column() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(3, "5")
    eval_result = run_oracle_subset_eval(model, queries, strategies=["direct_greedy"])
    oracle_summaries = compute_oracle_summaries(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )
    pairwise = compute_pairwise_win_matrix(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = write_oracle_outputs(eval_result, oracle_summaries, pairwise, tmpdir)
        with open(paths["oracle_assignments_csv"]) as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []

    assert "cheapest_correct_strategy" in header
    assert "direct_greedy_correct" in header
    assert "direct_already_optimal" in header


def test_per_query_matrix_csv_has_cost_proxy_column() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2, "5")
    eval_result = run_oracle_subset_eval(model, queries, strategies=["direct_greedy"])
    oracle_summaries = compute_oracle_summaries(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )
    pairwise = compute_pairwise_win_matrix(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = write_oracle_outputs(eval_result, oracle_summaries, pairwise, tmpdir)
        with open(paths["per_query_matrix_csv"]) as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []

    assert "cost_proxy" in header


# ---------------------------------------------------------------------------
# Tests: format_oracle_summary
# ---------------------------------------------------------------------------


def test_format_oracle_summary_contains_strategy_names() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2, "5")
    strategies = ["direct_greedy", "direct_plus_verify"]
    eval_result = run_oracle_subset_eval(model, queries, strategies=strategies)
    oracle_summaries = compute_oracle_summaries(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )
    text = format_oracle_summary(oracle_summaries, {})
    for s in strategies:
        assert s in text


def test_format_oracle_summary_contains_gap() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2, "5")
    eval_result = run_oracle_subset_eval(model, queries, strategies=["direct_greedy"])
    oracle_summaries = compute_oracle_summaries(
        eval_result["per_query_rows"], eval_result["strategies_run"]
    )
    text = format_oracle_summary(oracle_summaries, {})
    assert "gap" in text.lower()


# ---------------------------------------------------------------------------
# Tests: CORE_ORACLE_STRATEGIES and STRATEGY_COST_PROXY completeness
# ---------------------------------------------------------------------------


def test_core_strategies_all_in_cost_proxy() -> None:
    for s in CORE_ORACLE_STRATEGIES:
        assert s in STRATEGY_COST_PROXY, f"Missing cost proxy for '{s}'"


def test_core_strategies_no_duplicates() -> None:
    assert len(CORE_ORACLE_STRATEGIES) == len(set(CORE_ORACLE_STRATEGIES))


def test_eval_runs_all_core_strategies() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(1, "5")
    eval_result = run_oracle_subset_eval(model, queries)  # default strategies
    assert set(eval_result["strategies_run"]) == set(CORE_ORACLE_STRATEGIES)
