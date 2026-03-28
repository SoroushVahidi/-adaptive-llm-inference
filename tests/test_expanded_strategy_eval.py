"""Unit tests for expanded_strategy_eval.py.

All tests use lightweight fake models so they run offline without any API key.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.expanded_strategy_eval import (
    ALL_EXPANDED_STRATEGIES,
    format_expanded_strategy_summary,
    run_direct_plus_critique_plus_final,
    run_expanded_strategy,
    run_expanded_strategy_eval,
    run_first_pass_then_hint_guided_reason,
    write_expanded_strategy_outputs,
)

# ---------------------------------------------------------------------------
# Fake models (same pattern as test_strategy_expansion.py)
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
# Tests: direct_plus_critique_plus_final
# ---------------------------------------------------------------------------


def test_critique_plus_final_structure() -> None:
    model = _FakeModelFixed("9")
    result = run_direct_plus_critique_plus_final(model, "What is 3×3?")

    assert result["samples_used"] == 3
    assert len(result["raw_outputs"]) == 3
    assert "predicted_answer" in result
    assert "first_answer" in result
    assert "critique_text" in result
    assert "revised_answer" in result


def test_critique_plus_final_uses_final_stage_answer() -> None:
    # stage1 says 8, stage2 (critique) mentions 10, stage3 says Final answer: 9
    model = _FakeModelSequenced([
        "Final answer: 8",                        # stage 1 direct
        "The answer should be 10 because ...",    # stage 2 critique
        "Final answer: 9",                        # stage 3 final
    ])
    result = run_direct_plus_critique_plus_final(model, "What is 3+6?")

    assert result["first_answer"] == "8"
    assert result["predicted_answer"] == "9"
    assert result["samples_used"] == 3


def test_critique_plus_final_fallback_to_first_when_final_extraction_fails() -> None:
    # stage3 has no extractable number → should fall back to first answer
    model = _FakeModelSequenced([
        "Final answer: 5",
        "The reasoning looks correct.",        # critique with no number
        "The answer is still correct.",        # final with no number
    ])
    result = run_direct_plus_critique_plus_final(model, "dummy question")

    assert result["predicted_answer"] == "5"


def test_critique_plus_final_fallback_to_critique_when_stage3_empty() -> None:
    # stage3 has no number, but critique has "Corrected answer: 7"
    model = _FakeModelSequenced([
        "Final answer: 5",
        "Corrected answer: 7",         # critique has a number
        "No number here at all.",      # final has no number
    ])
    result = run_direct_plus_critique_plus_final(model, "dummy question")

    # Should use the number extracted from the critique
    assert result["predicted_answer"] == "7"


# ---------------------------------------------------------------------------
# Tests: first_pass_then_hint_guided_reason
# ---------------------------------------------------------------------------


def test_hint_guided_reason_structure() -> None:
    model = _FakeModelFixed("12")
    result = run_first_pass_then_hint_guided_reason(model, "What is 4×3?")

    assert result["samples_used"] == 2
    assert len(result["raw_outputs"]) == 2
    assert "predicted_answer" in result
    assert "first_answer" in result
    assert "revised_answer" in result


def test_hint_guided_reason_uses_stage2_answer() -> None:
    model = _FakeModelSequenced([
        "Final answer: 11",   # stage 1 – hint (slightly off)
        "Final answer: 12",   # stage 2 – corrected by hint-guided reasoning
    ])
    result = run_first_pass_then_hint_guided_reason(model, "What is 4×3?")

    assert result["first_answer"] == "11"
    assert result["predicted_answer"] == "12"


def test_hint_guided_reason_fallback_to_first_when_stage2_empty() -> None:
    model = _FakeModelSequenced([
        "Final answer: 7",
        "I cannot compute this.",  # no number in stage 2
    ])
    result = run_first_pass_then_hint_guided_reason(model, "dummy")

    assert result["predicted_answer"] == "7"


# ---------------------------------------------------------------------------
# Tests: dispatcher run_expanded_strategy
# ---------------------------------------------------------------------------


def test_run_expanded_strategy_dispatches_all_known() -> None:
    model = _FakeModelFixed("3")
    for strategy in ALL_EXPANDED_STRATEGIES:
        result = run_expanded_strategy(strategy, model, "dummy question")
        assert "predicted_answer" in result
        assert "samples_used" in result


def test_run_expanded_strategy_raises_for_unknown() -> None:
    model = _FakeModelFixed("1")
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_expanded_strategy("not_a_real_strategy", model, "q")


# ---------------------------------------------------------------------------
# Tests: run_expanded_strategy_eval
# ---------------------------------------------------------------------------


def _make_queries(n: int = 3) -> list[_FakeQuery]:
    return [_FakeQuery(f"q{i}", f"question {i}", str(i + 1)) for i in range(n)]


def test_eval_returns_all_required_keys() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2)
    result = run_expanded_strategy_eval(
        model, queries, strategies=["direct_greedy", "direct_plus_critique_plus_final"]
    )

    assert "per_query_rows" in result
    assert "strategy_summaries" in result
    assert "pairwise_comparisons" in result
    assert "total_queries" in result
    assert "strategies_run" in result


def test_eval_per_query_row_has_required_fields() -> None:
    model = _FakeModelFixed("3")
    queries = [_FakeQuery("q0", "What is 1+2?", "3")]
    result = run_expanded_strategy_eval(
        model, queries, strategies=["direct_greedy"]
    )

    row = result["per_query_rows"][0]
    required_fields = (
        "question_id", "strategy", "predicted_answer", "gold_answer", "correct", "samples_used"
    )
    for field in required_fields:
        assert field in row, f"Missing field: {field}"


def test_eval_new_strategies_expose_first_and_revised_answer() -> None:
    model = _FakeModelFixed("7")
    queries = [_FakeQuery("q0", "q?", "7")]
    result = run_expanded_strategy_eval(
        model,
        queries,
        strategies=["direct_plus_critique_plus_final", "first_pass_then_hint_guided_reason"],
    )
    for row in result["per_query_rows"]:
        assert "first_answer" in row
        assert "revised_answer" in row


def test_eval_accuracy_when_model_always_correct() -> None:
    model = _FakeModelFixed("3")
    queries = [_FakeQuery("q0", "q0", "3"), _FakeQuery("q1", "q1", "3")]
    result = run_expanded_strategy_eval(model, queries, strategies=["direct_greedy"])
    summary = result["strategy_summaries"]["direct_greedy"]

    assert summary["accuracy"] == 1.0
    assert summary["correct"] == 2


def test_eval_pairwise_has_direct_greedy_as_baseline() -> None:
    model = _FakeModelFixed("5")
    queries = _make_queries(2)
    result = run_expanded_strategy_eval(
        model,
        queries,
        strategies=["direct_greedy", "direct_plus_critique_plus_final"],
    )
    baselines = {pw["baseline"] for pw in result["pairwise_comparisons"]}
    assert "direct_greedy" in baselines


# ---------------------------------------------------------------------------
# Tests: output writing
# ---------------------------------------------------------------------------


def test_write_outputs_creates_three_files() -> None:
    model = _FakeModelFixed("4")
    queries = _make_queries(2)
    result = run_expanded_strategy_eval(model, queries, strategies=["direct_greedy"])

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = write_expanded_strategy_outputs(result, tmpdir)

        assert Path(paths["summary_json"]).exists()
        assert Path(paths["summary_csv"]).exists()
        assert Path(paths["per_query_csv"]).exists()

        payload = json.loads(Path(paths["summary_json"]).read_text())
        assert "strategy_summaries" in payload
        assert "pairwise_comparisons" in payload


def test_format_summary_contains_all_strategy_names() -> None:
    model = _FakeModelFixed("1")
    queries = _make_queries(1)
    strategies = [
        "direct_greedy",
        "direct_plus_critique_plus_final",
        "first_pass_then_hint_guided_reason",
    ]
    result = run_expanded_strategy_eval(model, queries, strategies=strategies)
    paths = {
        "summary_json": "/tmp/s.json",
        "summary_csv": "/tmp/s.csv",
        "per_query_csv": "/tmp/p.csv",
    }
    text = format_expanded_strategy_summary(result, paths)

    for strategy in strategies:
        assert strategy in text
    assert "accuracy" in text.lower()
