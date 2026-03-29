"""Offline tests for multi-action routing helpers and new strategy runners."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from src.evaluation.multi_action_routing import (
    LAMBDA_VALUES,
    best_accuracy_action,
    build_multi_action_rows,
    compute_oracle_summary_json,
    write_multi_action_csv,
)
from src.evaluation.oracle_subset_eval import MULTI_ACTION_ORACLE_STRATEGIES, run_oracle_subset_eval
from src.evaluation.strategy_expansion_eval import run_reasoning_then_revise, run_self_consistency_3


class _FakeFixed:
    def __init__(self, answer: str = "5") -> None:
        self.answer = answer

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return f"Final answer: {self.answer}"

    def generate_n(self, prompt: str, n: int) -> list[str]:  # noqa: ARG002
        return [f"Final answer: {self.answer}"] * n


class _FakeVote:
    """Three different answers for self_consistency."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return "Final answer: 1"

    def generate_n(self, prompt: str, n: int) -> list[str]:  # noqa: ARG002
        return ["Final answer: 1", "Final answer: 1", "Final answer: 2"]


class _Q:
    def __init__(self, qid: str, q: str, a: str) -> None:
        self.id = qid
        self.question = q
        self.answer = a


def test_reasoning_then_revise_two_calls() -> None:
    m = _FakeFixed("7")
    r = run_reasoning_then_revise(m, "Q?")
    assert r["samples_used"] == 2
    assert len(r["raw_outputs"]) == 2
    assert r["predicted_answer"] == "7"


def test_self_consistency_3_majority() -> None:
    m = _FakeVote()
    r = run_self_consistency_3(m, "Q?")
    assert r["predicted_answer"] == "1"
    assert r["samples_used"] == 3
    assert r.get("self_consistency_ambiguous") is False


def test_multi_action_oracle_eval_smoke() -> None:
    model = _FakeFixed("3")
    queries = [_Q("q0", "x", "3")]
    res = run_oracle_subset_eval(model, queries, strategies=list(MULTI_ACTION_ORACLE_STRATEGIES))
    assert len(res["per_query_rows"]) == 4
    assert {r["strategy"] for r in res["per_query_rows"]} == set(MULTI_ACTION_ORACLE_STRATEGIES)


def test_best_accuracy_prefers_correct_cheaper() -> None:
    corr = {"a": 0, "b": 1, "c": 1}
    cost = {"a": 1, "b": 3, "c": 2}
    assert best_accuracy_action(corr, cost) == "c"


def test_build_rows_and_summary() -> None:
    model = _FakeFixed("9")
    queries = [_Q("q0", "How many?", "9")]
    res = run_oracle_subset_eval(model, queries, strategies=list(MULTI_ACTION_ORACLE_STRATEGIES))
    rows = build_multi_action_rows(
        queries,
        res["per_query_rows"],
        list(MULTI_ACTION_ORACLE_STRATEGIES),
        "testds",
    )
    assert len(rows) == 1
    assert rows[0]["best_accuracy_action"] in MULTI_ACTION_ORACLE_STRATEGIES
    for lam in LAMBDA_VALUES:
        suf = f"{lam:.2f}".replace(".", "_")
        assert f"best_utility_action_lambda_{suf}" in rows[0]
    summ = compute_oracle_summary_json(
        rows,
        list(MULTI_ACTION_ORACLE_STRATEGIES),
        model="fake",
        dataset_name="testds",
    )
    assert summ["num_queries"] == 1
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        write_multi_action_csv(rows, p)
        with p.open() as fh:
            r = list(csv.DictReader(fh))
        assert len(r) == 1


def test_gsm8k_tail_slice() -> None:
    from src.datasets.gsm8k import load_gsm8k

    payload = [
        {"question": "a", "answer": "#### 1"},
        {"question": "b", "answer": "#### 2"},
        {"question": "c", "answer": "#### 3"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(payload, tmp)
        path = Path(tmp.name)
    try:
        q = load_gsm8k(data_file=path, tail_max_samples=2)
        assert len(q) == 2
        assert q[0].answer == "2"
        assert q[1].answer == "3"
    finally:
        path.unlink()
