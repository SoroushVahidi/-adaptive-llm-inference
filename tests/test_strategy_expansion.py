"""Unit tests for strategy_expansion_eval.py.

All tests use lightweight fake models so they run offline without any API key.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.strategy_expansion_eval import (
    ALL_STRATEGIES,
    format_strategy_expansion_summary,
    run_direct_greedy,
    run_direct_plus_revise,
    run_direct_plus_verify,
    run_reasoning_best_of_3,
    run_strategy,
    run_strategy_expansion_eval,
    run_structured_sampling_3,
    write_strategy_expansion_outputs,
)

# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------

class _FakeModelFixed:
    """Returns the same answer every call."""

    def __init__(self, answer: str = "42") -> None:
        self.answer = answer
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return f"Final answer: {self.answer}"

    def generate_n(self, prompt: str, n: int) -> list[str]:
        self.calls.append(prompt)
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


# ---------------------------------------------------------------------------
# Tests for individual strategy runners
# ---------------------------------------------------------------------------

def test_run_direct_greedy_returns_correct_structure() -> None:
    model = _FakeModelFixed("7")
    result = run_direct_greedy(model, "What is 3+4?")

    assert result["samples_used"] == 1
    assert result["predicted_answer"] == "7"
    assert len(result["raw_outputs"]) == 1


def test_run_reasoning_best_of_3_returns_correct_structure() -> None:
    model = _FakeModelFixed("10")
    result = run_reasoning_best_of_3(model, "What is 5+5?")

    assert result["samples_used"] == 3
    assert result["predicted_answer"] == "10"
    assert len(result["raw_outputs"]) == 3


def test_run_structured_sampling_3_returns_correct_structure() -> None:
    model = _FakeModelFixed("15")
    result = run_structured_sampling_3(model, "What is 3×5?")

    assert result["samples_used"] == 3
    assert result["predicted_answer"] == "15"
    assert len(result["raw_outputs"]) == 3


def test_run_structured_sampling_3_uses_majority_vote() -> None:
    # Two say 6, one says 7 → majority should be 6
    model = _FakeModelSequenced([
        "Final answer: 6",
        "Final answer: 7",
        "Final answer: 6",
    ])
    result = run_structured_sampling_3(model, "What is 2×3?")

    assert result["predicted_answer"] == "6"


def test_run_direct_plus_verify_keeps_original_when_correct() -> None:
    # Verifier says CORRECT → keep first answer
    model = _FakeModelSequenced([
        "Final answer: 8",   # direct answer
        "CORRECT",           # verifier response
    ])
    result = run_direct_plus_verify(model, "What is 4+4?")

    assert result["samples_used"] == 2
    assert result["first_answer"] == "8"
    assert result["predicted_answer"] == "8"
    assert "first_answer" in result
    assert "revised_answer" in result


def test_run_direct_plus_verify_corrects_when_wrong() -> None:
    # Verifier says WRONG and proposes 9
    model = _FakeModelSequenced([
        "Final answer: 8",                     # direct answer (wrong)
        "WRONG. Correct answer: 9",            # verifier rejects and corrects
    ])
    result = run_direct_plus_verify(model, "What is 4+5?")

    assert result["samples_used"] == 2
    assert result["first_answer"] == "8"
    assert result["predicted_answer"] == "9"


def test_run_direct_plus_revise_extracts_revised_answer() -> None:
    model = _FakeModelSequenced([
        "Final answer: 5",               # first direct answer
        "Let me recheck. Final answer: 6",  # revision answer
    ])
    result = run_direct_plus_revise(model, "What is 2+4?")

    assert result["samples_used"] == 2
    assert result["first_answer"] == "5"
    assert result["predicted_answer"] == "6"
    assert result["revised_answer"] == "6"


def test_run_direct_plus_revise_fallback_to_first_when_revision_empty() -> None:
    # Revision produces no extractable number → fall back to first_answer
    model = _FakeModelSequenced([
        "Final answer: 5",
        "I cannot determine the answer.",  # no number
    ])
    result = run_direct_plus_revise(model, "What is 2+3?")

    assert result["predicted_answer"] == "5"


# ---------------------------------------------------------------------------
# Tests for run_strategy dispatcher
# ---------------------------------------------------------------------------

def test_run_strategy_dispatches_to_correct_runner() -> None:
    model = _FakeModelFixed("3")
    for strategy in ALL_STRATEGIES:
        result = run_strategy(strategy, model, "dummy question")
        assert "predicted_answer" in result
        assert "samples_used" in result


def test_run_strategy_raises_for_unknown_strategy() -> None:
    model = _FakeModelFixed("1")
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_strategy("nonexistent_strategy", model, "q")


# ---------------------------------------------------------------------------
# Tests for run_strategy_expansion_eval
# ---------------------------------------------------------------------------

def _make_queries(n: int = 3) -> list[_FakeQuery]:
    return [_FakeQuery(f"q{i}", f"question {i}", str(i + 1)) for i in range(n)]


def test_eval_returns_all_required_keys() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2)
    result = run_strategy_expansion_eval(model, queries, strategies=["direct_greedy"])

    assert "per_query_rows" in result
    assert "strategy_summaries" in result
    assert "pairwise_comparisons" in result
    assert "total_queries" in result
    assert "strategies_run" in result


def test_eval_per_query_row_structure() -> None:
    model = _FakeModelFixed("3")
    queries = [_FakeQuery("q0", "What is 1+2?", "3")]
    result = run_strategy_expansion_eval(model, queries, strategies=["direct_greedy"])

    rows = result["per_query_rows"]
    assert len(rows) == 1
    row = rows[0]
    assert "question_id" in row
    assert "strategy" in row
    assert "predicted_answer" in row
    assert "gold_answer" in row
    assert "correct" in row
    assert "samples_used" in row


def test_eval_accuracy_calculation() -> None:
    # Model always answers "3", gold is also "3" for all queries
    model = _FakeModelFixed("3")
    queries = [
        _FakeQuery("q0", "q0", "3"),
        _FakeQuery("q1", "q1", "3"),
    ]
    result = run_strategy_expansion_eval(model, queries, strategies=["direct_greedy"])
    summary = result["strategy_summaries"]["direct_greedy"]

    assert summary["accuracy"] == 1.0
    assert summary["correct"] == 2


def test_eval_pairwise_comparisons_include_direct_greedy_as_baseline() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2)
    result = run_strategy_expansion_eval(
        model, queries, strategies=["direct_greedy", "direct_plus_verify"]
    )
    baselines_in_pairwise = {pw["baseline"] for pw in result["pairwise_comparisons"]}
    assert "direct_greedy" in baselines_in_pairwise


def test_eval_two_step_strategies_expose_first_and_revised_answer() -> None:
    model = _FakeModelFixed("7")
    queries = [_FakeQuery("q0", "q?", "7")]
    result = run_strategy_expansion_eval(
        model, queries, strategies=["direct_plus_verify", "direct_plus_revise"]
    )
    for row in result["per_query_rows"]:
        assert "first_answer" in row
        assert "revised_answer" in row


# ---------------------------------------------------------------------------
# Tests for output writing
# ---------------------------------------------------------------------------

def test_write_outputs_creates_expected_files() -> None:
    model = _FakeModelFixed("4")
    queries = _make_queries(2)
    result = run_strategy_expansion_eval(model, queries, strategies=["direct_greedy"])

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = write_strategy_expansion_outputs(result, tmpdir)

        assert Path(paths["summary_json"]).exists()
        assert Path(paths["summary_csv"]).exists()
        assert Path(paths["per_query_csv"]).exists()

        # summary.json should be valid JSON
        payload = json.loads(Path(paths["summary_json"]).read_text())
        assert "strategy_summaries" in payload
        assert "pairwise_comparisons" in payload


def test_format_summary_contains_strategy_names() -> None:
    model = _FakeModelFixed("1")
    queries = _make_queries(1)
    result = run_strategy_expansion_eval(model, queries, strategies=["direct_greedy"])
    paths = {
        "summary_json": "/tmp/s.json",
        "summary_csv": "/tmp/s.csv",
        "per_query_csv": "/tmp/p.csv",
    }
    summary_text = format_strategy_expansion_summary(result, paths)

    assert "direct_greedy" in summary_text
    assert "accuracy" in summary_text.lower()
