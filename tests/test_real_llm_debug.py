from src.evaluation.real_llm_debug import (
    _selective_debug,
    run_real_llm_debug,
)


def test_run_real_llm_debug_summary_counts_diversity_and_parse_collapse() -> None:
    result = {
        "raw_samples": [
            {
                "question_id": "q1",
                "prompt_style": "direct",
                "raw_samples": ["Answer: 14", "The final answer is 14.", "14"],
                "parsed_answers": ["14", "14", "14"],
                "raw_outputs_identical": False,
                "parsed_answers_identical": True,
                "extraction_failures": [False, False, False],
            },
            {
                "question_id": "q2",
                "prompt_style": "direct",
                "raw_samples": ["Answer: 5", "Answer: 6", "Answer: 6"],
                "parsed_answers": ["5", "6", "6"],
                "raw_outputs_identical": False,
                "parsed_answers_identical": False,
                "extraction_failures": [False, False, False],
            },
        ],
        "parsed_samples": [],
        "selective_debug": [],
        "summary": {
            "model_name": "gpt-test",
            "greedy_temperature": 0.0,
            "sample_temperature": 0.7,
            "multiple_samples_requested_in_single_call": True,
            "n_samples_per_debug_call": 3,
            "total_query_prompt_pairs": 2,
            "raw_outputs_identical_count": 0,
            "parsed_answers_identical_count": 1,
            "parsed_collapse_count": 1,
            "strict_escalation_candidate_count": 0,
            "looser_escalation_candidate_count": 0,
        },
    }

    assert result["summary"]["total_query_prompt_pairs"] == 2
    assert result["summary"]["parsed_answers_identical_count"] == 1
    assert result["summary"]["parsed_collapse_count"] == 1


def test_selective_debug_counts_threshold_candidates() -> None:
    direct_rows = [
        {
            "question_id": "q1",
            "raw_samples": ["Answer: 3", "Answer: 3", "Answer: 3"],
            "parsed_answers": ["3", "3", "3"],
        },
        {
            "question_id": "q2",
            "raw_samples": ["", "Answer: 12", "Answer: 12"],
            "parsed_answers": ["", "12", "12"],
        },
    ]
    rows = _selective_debug(
        direct_rows=direct_rows,
        method_cfg={
            "total_budget": 10,
            "extra_samples_per_escalated_query": 2,
            "use_second_sample_for_disagreement": True,
            "parse_failure_weight": 2.0,
            "disagreement_weight": 1.5,
            "malformed_output_weight": 1.0,
            "missing_numeric_weight": 1.0,
            "min_score_to_escalate": 1.5,
        },
    )
    assert len(rows) == 2
    assert rows[0]["strict_threshold_would_escalate"] is False
    assert rows[1]["strict_threshold_would_escalate"] is True
