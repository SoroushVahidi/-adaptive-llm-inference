from __future__ import annotations

import json

from src.evaluation.strategy_diagnostic import (
    MAX_QUERY_LIMIT,
    PROMPT_DIRECT,
    PROMPT_REASONING,
    STRATEGY_SPECS,
    build_comparison_summary,
    classify_access_error,
    summarize_rows,
    write_strategy_diagnostic_outputs,
)


def test_prompt_constants_and_strategy_set_are_present() -> None:
    assert PROMPT_DIRECT.strip()
    assert PROMPT_REASONING.strip()
    assert [spec.name for spec in STRATEGY_SPECS] == [
        "direct_greedy",
        "reasoning_greedy",
        "reasoning_best_of_3",
        "structured_sampling_3",
    ]
    assert MAX_QUERY_LIMIT == 30


def test_classify_access_error_distinguishes_common_failure_modes() -> None:
    assert classify_access_error("Missing OPENAI_API_KEY") == "config"
    assert classify_access_error("HTTP 429: insufficient_quota") == "quota"
    assert classify_access_error("HTTP 429: Rate limit reached") == "rate_limit"
    assert classify_access_error("The model does not exist or you do not have access") == (
        "model_access"
    )
    assert classify_access_error("OpenAI API request failed: timed out") == "network"


def test_summarize_rows_aggregates_accuracy_and_sample_counts() -> None:
    rows = [
        {
            "question_id": "q1",
            "model": "gpt-4o-mini",
            "strategy": "reasoning_greedy",
            "correct": True,
            "samples_used": 1,
        },
        {
            "question_id": "q2",
            "model": "gpt-4o-mini",
            "strategy": "reasoning_greedy",
            "correct": False,
            "samples_used": 1,
        },
        {
            "question_id": "q1",
            "model": "gpt-4o-mini",
            "strategy": "reasoning_best_of_3",
            "correct": True,
            "samples_used": 3,
        },
        {
            "question_id": "q2",
            "model": "gpt-4o-mini",
            "strategy": "reasoning_best_of_3",
            "correct": True,
            "samples_used": 3,
        },
    ]

    summary_rows = summarize_rows(rows)

    assert summary_rows == [
        {
            "model": "gpt-4o-mini",
            "strategy": "reasoning_greedy",
            "accuracy": 0.5,
            "correct": 1,
            "total_queries": 2,
            "total_samples": 2,
            "avg_samples_per_query": 1.0,
        },
        {
            "model": "gpt-4o-mini",
            "strategy": "reasoning_best_of_3",
            "accuracy": 1.0,
            "correct": 2,
            "total_queries": 2,
            "total_samples": 6,
            "avg_samples_per_query": 3.0,
        },
    ]


def test_build_comparison_summary_computes_model_and_strategy_deltas() -> None:
    summary_rows = [
        {"model": "gpt-4o-mini", "strategy": "reasoning_greedy", "accuracy": 0.4},
        {"model": "gpt-4o-mini", "strategy": "reasoning_best_of_3", "accuracy": 0.5},
        {"model": "gpt-4o-mini", "strategy": "structured_sampling_3", "accuracy": 0.6},
        {"model": "gpt-4o", "strategy": "reasoning_greedy", "accuracy": 0.5},
        {"model": "gpt-4o", "strategy": "reasoning_best_of_3", "accuracy": 0.7},
        {"model": "gpt-4o", "strategy": "structured_sampling_3", "accuracy": 0.75},
    ]

    comparison = build_comparison_summary(summary_rows, "gpt-4o-mini", "gpt-4o")

    assert comparison["per_model_improvements"] == [
        {
            "model": "gpt-4o",
            "reasoning_best_of_3_minus_reasoning_greedy": 0.2,
            "structured_sampling_3_minus_reasoning_best_of_3": 0.05,
            "structured_sampling_3_beats_reasoning_best_of_3": True,
        },
        {
            "model": "gpt-4o-mini",
            "reasoning_best_of_3_minus_reasoning_greedy": 0.1,
            "structured_sampling_3_minus_reasoning_best_of_3": 0.1,
            "structured_sampling_3_beats_reasoning_best_of_3": True,
        },
    ]
    assert comparison["stronger_model_effect"]["available"] is True
    assert comparison["stronger_model_effect"]["reasoning_best_of_3_minus_reasoning_greedy"][
        "difference"
    ] == 0.1


def test_write_outputs_excludes_per_query_rows_from_summary_json(tmp_path) -> None:
    result = {
        "scientific_intent": "diagnostic",
        "dataset": {"name": "math500", "num_queries": 1},
        "models": {"current": "gpt-4o-mini", "stronger": "gpt-4o", "tested": ["gpt-4o-mini"]},
        "access_checks": [],
        "stronger_model_status": {
            "model": "gpt-4o",
            "status": "failed",
            "error_type": "model_access",
            "error": "no access",
        },
        "summary_rows": [
            {
                "model": "gpt-4o-mini",
                "strategy": "direct_greedy",
                "accuracy": 1.0,
                "correct": 1,
                "total_queries": 1,
                "total_samples": 1,
                "avg_samples_per_query": 1.0,
            }
        ],
        "comparisons": {
            "per_model_improvements": [],
            "stronger_model_effect": {"available": False},
        },
        "per_query_results": [
            {
                "question_id": "test/algebra/1.json",
                "model": "gpt-4o-mini",
                "strategy": "direct_greedy",
                "predicted_answer": "\\frac{1}{2}",
                "gold_answer": "\\frac{1}{2}",
                "parsed_candidate_answers": json.dumps(["\\frac{1}{2}"]),
            }
        ],
    }

    paths = write_strategy_diagnostic_outputs(result, tmp_path)

    summary_payload = json.loads((tmp_path / "summary.json").read_text())
    assert "per_query_results" not in summary_payload
    assert (tmp_path / "summary.csv").read_text().startswith("model,strategy,accuracy")
    assert (tmp_path / "per_query_results.csv").read_text().startswith(
        "question_id,model,strategy,predicted_answer,gold_answer"
    )
    assert paths["summary_json"].endswith("summary.json")
