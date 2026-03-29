from __future__ import annotations

import csv

import pytest

from src.evaluation.real_routing_model_eval import run_real_routing_model_eval
from src.models.revise_helpful_classifier import detect_sklearn_support


@pytest.mark.skipif(not detect_sklearn_support().available, reason="sklearn unavailable")
def test_real_routing_model_eval_runs_on_small_dataset(tmp_path) -> None:
    dataset = tmp_path / "real.csv"
    rows = [
        {
            "question_id": "q1",
            "question": "",
            "gold_answer": "",
            "reasoning_answer": "",
            "revise_answer": "",
            "reasoning_correct": 0,
            "revise_correct": 1,
            "revise_helpful": 1,
            "reasoning_cost": 1,
            "revise_cost": 2,
            "calibrated_strong_escalation_candidate": 1,
            "unified_error_score": 0.9,
            "question_length_chars": 20,
        },
        {
            "question_id": "q2",
            "question": "",
            "gold_answer": "",
            "reasoning_answer": "",
            "revise_answer": "",
            "reasoning_correct": 1,
            "revise_correct": 1,
            "revise_helpful": 0,
            "reasoning_cost": 1,
            "revise_cost": 2,
            "calibrated_strong_escalation_candidate": 0,
            "unified_error_score": 0.1,
            "question_length_chars": 10,
        },
        {
            "question_id": "q3",
            "question": "",
            "gold_answer": "",
            "reasoning_answer": "",
            "revise_answer": "",
            "reasoning_correct": 0,
            "revise_correct": 1,
            "revise_helpful": 1,
            "reasoning_cost": 1,
            "revise_cost": 2,
            "calibrated_strong_escalation_candidate": 1,
            "unified_error_score": 0.8,
            "question_length_chars": 30,
        },
        {
            "question_id": "q4",
            "question": "",
            "gold_answer": "",
            "reasoning_answer": "",
            "revise_answer": "",
            "reasoning_correct": 1,
            "revise_correct": 1,
            "revise_helpful": 0,
            "reasoning_cost": 1,
            "revise_cost": 2,
            "calibrated_strong_escalation_candidate": 0,
            "unified_error_score": 0.2,
            "question_length_chars": 12,
        },
    ]

    with dataset.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = run_real_routing_model_eval(dataset_csv=dataset, output_dir=tmp_path / "outputs")
    assert result["summary"]["run_status"] == "OK"
    assert (tmp_path / "outputs" / "model_metrics.csv").exists()
    assert (tmp_path / "outputs" / "routing_simulation.csv").exists()


def test_real_routing_model_eval_blocks_when_dataset_missing(tmp_path) -> None:
    result = run_real_routing_model_eval(
        dataset_csv=tmp_path / "missing.csv",
        output_dir=tmp_path / "outputs",
    )
    assert result["summary"]["run_status"] == "BLOCKED"
