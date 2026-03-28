from __future__ import annotations

import json

from src.evaluation.oracle_subset_eval import (
    build_oracle_subset_artifacts,
    run_reasoning_greedy,
    write_oracle_subset_outputs,
)


class _FakeModelFixed:
    def __init__(self, answer: str = "42") -> None:
        self.answer = answer

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return f"Final answer: {self.answer}"

    def generate_n(self, prompt: str, n: int) -> list[str]:  # noqa: ARG002
        return [f"Final answer: {self.answer}"] * n


def test_run_reasoning_greedy_returns_single_sample_answer() -> None:
    model = _FakeModelFixed("9")
    result = run_reasoning_greedy(model, "What is 4 + 5?")

    assert result["predicted_answer"] == "9"
    assert result["samples_used"] == 1
    assert len(result["raw_outputs"]) == 1


def test_build_oracle_subset_artifacts_computes_oracle_and_fix_counts() -> None:
    rows = [
        {
            "question_id": "q1",
            "strategy": "direct_greedy",
            "predicted_answer": "1",
            "gold_answer": "2",
            "correct": False,
            "samples_used": 1,
        },
        {
            "question_id": "q1",
            "strategy": "reasoning_greedy",
            "predicted_answer": "2",
            "gold_answer": "2",
            "correct": True,
            "samples_used": 1,
        },
        {
            "question_id": "q1",
            "strategy": "structured_sampling_3",
            "predicted_answer": "2",
            "gold_answer": "2",
            "correct": True,
            "samples_used": 3,
        },
        {
            "question_id": "q2",
            "strategy": "direct_greedy",
            "predicted_answer": "5",
            "gold_answer": "5",
            "correct": True,
            "samples_used": 1,
        },
        {
            "question_id": "q2",
            "strategy": "reasoning_greedy",
            "predicted_answer": "0",
            "gold_answer": "5",
            "correct": False,
            "samples_used": 1,
        },
        {
            "question_id": "q2",
            "strategy": "structured_sampling_3",
            "predicted_answer": "0",
            "gold_answer": "5",
            "correct": False,
            "samples_used": 3,
        },
    ]

    artifacts = build_oracle_subset_artifacts(
        per_query_rows=rows,
        strategies=["direct_greedy", "reasoning_greedy", "structured_sampling_3"],
        total_queries=2,
    )

    summary = {row["strategy"]: row for row in artifacts["summary_rows"]}
    assert artifacts["global_metrics"]["oracle_accuracy"] == 1.0
    assert artifacts["global_metrics"]["oracle_minus_direct_gap"] == 0.5
    assert artifacts["global_metrics"]["oracle_minus_reasoning_greedy_gap"] == 0.5
    assert summary["reasoning_greedy"]["fixes_direct_failures"] == 1
    assert summary["direct_greedy"]["cheapest_correct_count"] == 1
    assert summary["reasoning_greedy"]["cheapest_correct_count"] == 1


def test_write_oracle_subset_outputs_creates_requested_files(tmp_path) -> None:
    result = {
        "run_status": "COMPLETED",
        "dataset": {"name": "gsm8k", "num_queries": 1},
        "models": {"current": "gpt-4o-mini", "strong": None, "strong_model_status": None},
        "access_checks": [],
        "strategies_run": ["direct_greedy"],
        "per_query_rows": [],
        "summary_rows": [
            {
                "strategy": "direct_greedy",
                "accuracy": 1.0,
                "correct": 1,
                "total_queries": 1,
                "wins": 1,
                "unique_wins": 1,
                "fixes_direct_failures": 0,
                "fixes_reasoning_failures": "",
                "cheapest_correct_count": 1,
                "cheapest_correct_fraction": 1.0,
                "avg_cost": 1.0,
            }
        ],
        "per_query_matrix_rows": [
            {
                "question_id": "q1",
                "gold_answer": "5",
                "direct_greedy_correct": True,
            }
        ],
        "oracle_assignments": [
            {
                "question_id": "q1",
                "oracle_strategy": "direct_greedy",
                "oracle_correct": True,
            }
        ],
        "pairwise_win_matrix_rows": [
            {"strategy": "direct_greedy", "direct_greedy": 0}
        ],
        "global_metrics": {
            "oracle_accuracy": 1.0,
            "oracle_minus_direct_gap": 0.0,
            "oracle_minus_reasoning_greedy_gap": None,
            "average_oracle_cost_on_success": 1.0,
        },
    }

    paths = write_oracle_subset_outputs(result, tmp_path)

    summary_payload = json.loads((tmp_path / "summary.json").read_text())
    assert summary_payload["global_metrics"]["oracle_accuracy"] == 1.0
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "per_query_matrix.csv").exists()
    assert (tmp_path / "oracle_assignments.csv").exists()
    assert (tmp_path / "pairwise_win_matrix.csv").exists()
    assert paths["pairwise_win_matrix_csv"].endswith("pairwise_win_matrix.csv")
