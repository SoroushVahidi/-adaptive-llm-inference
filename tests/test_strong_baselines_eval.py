"""Offline tests for strong_baselines_eval and self-consistency voting."""

from __future__ import annotations

from src.baselines.self_consistency import majority_vote_self_consistency
from src.evaluation.strategy_expansion_eval import (
    run_reasoning_greedy,
    run_reasoning_then_revise,
)
from src.evaluation.strong_baselines_eval import (
    evaluate_compute_ladder,
    run_static_method,
)


class _FakeModel:
    def __init__(self, answer: str = "7") -> None:
        self._a = answer

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return f"Final answer: {self._a}"

    def generate_n(self, prompt: str, n: int) -> list[str]:  # noqa: ARG002
        return [f"Final answer: {self._a}"] * n


class _FakeQuery:
    def __init__(self, qid: str, question: str, answer: str) -> None:
        self.id = qid
        self.question = question
        self.answer = answer


def test_majority_vote_three_way_tie() -> None:
    maj, amb, tie = majority_vote_self_consistency(
        [
            "Final answer: 1",
            "Final answer: 2",
            "Final answer: 3",
        ],
        use_math_extraction=False,
    )
    assert tie
    assert maj == "1"
    assert not amb


def test_majority_vote_ambiguous_all_empty() -> None:
    maj, amb, tie = majority_vote_self_consistency(
        ["no numbers", "still none"],
        use_math_extraction=False,
    )
    assert amb
    assert maj == ""
    assert not tie


def test_reasoning_then_revise_two_calls() -> None:
    m = _FakeModel("9")
    r = run_reasoning_then_revise(m, "What is 3*3?")
    assert r["samples_used"] == 2
    assert len(r["raw_outputs"]) == 2
    assert r["predicted_answer"] == "9"


def test_reasoning_greedy_one_call() -> None:
    m = _FakeModel("4")
    r = run_reasoning_greedy(m, "Q")
    assert r["samples_used"] == 1


def test_compute_ladder_smoke() -> None:
    queries = [_FakeQuery("q1", "What is 2+2?", "4")]
    ladder = evaluate_compute_ladder(
        _FakeModel("4"), queries, dataset_key="test", task_type="numeric"
    )
    assert "reasoning_greedy" in ladder["methods"]
    assert ladder["methods"]["reasoning_greedy"]["accuracy"] == 1.0


def test_self_consistency_3_uses_math_when_flagged() -> None:
    class _M:
        def generate(self, p: str) -> str:  # noqa: ARG002
            return ""

        def generate_n(self, p: str, n: int) -> list[str]:  # noqa: ARG002
            return [r"Final answer: \boxed{\frac{1}{2}}"] * n

    from src.datasets.gsm8k import Query

    q = Query(id="1", question="q", answer=r"\frac{1}{2}")
    out = run_static_method(
        _M(),
        "self_consistency_3",
        q,
        eval_opts={"use_math_extraction": True, "use_mcq": False},
    )
    assert out["samples_used"] == 3
