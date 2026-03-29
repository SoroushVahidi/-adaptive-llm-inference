"""Unit tests for oracle strategy evaluation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.oracle_strategy_eval import (
    BASE_REQUIRED_STRATEGIES,
    run_oracle_strategy_eval,
    write_oracle_strategy_outputs,
)


class _FakeQuery:
    def __init__(self, qid: str, question: str, answer: str) -> None:
        self.id = qid
        self.question = question
        self.answer = answer


class _PromptAwareFakeModel:
    """A tiny deterministic fake model keyed by prompt/query markers."""

    def generate(self, prompt: str) -> str:
        qtag = "Q1" if "Q1" in prompt else "Q2" if "Q2" in prompt else "Q3"

        if "Based on the critique" in prompt:
            return "Final answer: 2" if qtag == "Q2" else "Final answer: 9"
        if "A student gave this answer" in prompt:
            return "Corrected answer: 2" if qtag == "Q2" else "Looks fine."
        if "Please review your work" in prompt:
            return "Final answer: 2" if qtag == "Q2" else "Final answer: 0"
        if "Is this answer correct?" in prompt:
            return "WRONG. Correct answer: 2" if qtag == "Q2" else "CORRECT"
        if "Hint:" in prompt:
            return "Final answer: 2" if qtag == "Q2" else "Final answer: 0"
        if "double-check your work" in prompt and qtag == "Q3":
            return "Final answer: 3"
        if "step by step" in prompt and qtag == "Q3":
            return "Final answer: 3"

        if qtag == "Q1":
            return "Final answer: 1"
        if qtag == "Q2":
            return "Final answer: 5"
        return "Final answer: 8"

    def generate_n(self, prompt: str, n: int) -> list[str]:
        qtag = "Q1" if "Q1" in prompt else "Q2" if "Q2" in prompt else "Q3"
        if qtag == "Q3":
            return ["Final answer: 3"] * n
        if qtag == "Q2":
            return ["Final answer: 5"] * n
        return ["Final answer: 1"] * n


def _queries() -> list[_FakeQuery]:
    return [
        _FakeQuery("q1", "Q1: one plus zero?", "1"),
        _FakeQuery("q2", "Q2: one plus one?", "2"),
        _FakeQuery("q3", "Q3: one plus two?", "3"),
    ]


def test_oracle_eval_computes_direct_vs_oracle_gap() -> None:
    model = _PromptAwareFakeModel()
    result = run_oracle_strategy_eval(
        model=model,
        queries=_queries(),
        strategies=list(BASE_REQUIRED_STRATEGIES),
        lambda_penalty=0.1,
    )

    summary = result["summary"]
    assert summary["total_queries"] == 3
    assert summary["direct_accuracy"] == pytest.approx(1 / 3)
    assert summary["oracle_accuracy"] == pytest.approx(1.0)
    assert summary["oracle_direct_gap"] == pytest.approx(2 / 3)
    assert summary["fraction_direct_optimal"] == pytest.approx(1 / 3)


def test_oracle_eval_tracks_family_help_fractions() -> None:
    model = _PromptAwareFakeModel()
    result = run_oracle_strategy_eval(
        model=model,
        queries=_queries(),
        strategies=list(BASE_REQUIRED_STRATEGIES),
    )

    summary = result["summary"]
    assert summary["fraction_critique_helps"] == pytest.approx(1 / 3)
    assert summary["fraction_hint_guided_helps"] == pytest.approx(1 / 3)
    assert summary["fraction_multi_sample_helps"] == pytest.approx(1 / 3)


def test_oracle_eval_outputs_contain_expected_sections() -> None:
    model = _PromptAwareFakeModel()
    result = run_oracle_strategy_eval(
        model=model,
        queries=_queries(),
        strategies=list(BASE_REQUIRED_STRATEGIES),
    )

    assert "per_query_matrix" in result
    assert "oracle_assignments" in result
    assert "pairwise_wins" in result
    assert result["oracle_assignments"]
    assert result["pairwise_wins"]


def test_write_oracle_outputs_creates_expected_files() -> None:
    model = _PromptAwareFakeModel()
    result = run_oracle_strategy_eval(
        model=model,
        queries=_queries(),
        strategies=list(BASE_REQUIRED_STRATEGIES),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = write_oracle_strategy_outputs(result, tmpdir)

        assert Path(paths["summary_json"]).exists()
        assert Path(paths["summary_csv"]).exists()
        assert Path(paths["per_query_csv"]).exists()
        assert Path(paths["oracle_assignments_csv"]).exists()

        payload = json.loads(Path(paths["summary_json"]).read_text())
        assert "oracle_accuracy" in payload
        assert "direct_accuracy" in payload
        assert "pairwise_wins" in payload
