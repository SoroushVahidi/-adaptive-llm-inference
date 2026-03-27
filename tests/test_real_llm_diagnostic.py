from src.evaluation.real_llm_diagnostic import (
    DIRECT_PROMPT,
    REASONING_PROMPT,
    build_overlap_summary,
    summarize_method_results,
)


def test_pairwise_overlap_summary_counts_changes() -> None:
    per_query_rows = [
        {
            "question_id": "q1",
            "gold_answer": "12",
            "greedy_direct_correct": True,
            "greedy_reasoning_correct": True,
            "best_of_3_reasoning_correct": False,
            "self_consistency_3_reasoning_correct": True,
        },
        {
            "question_id": "q2",
            "gold_answer": "9",
            "greedy_direct_correct": False,
            "greedy_reasoning_correct": False,
            "best_of_3_reasoning_correct": True,
            "self_consistency_3_reasoning_correct": True,
        },
    ]

    summary = build_overlap_summary(per_query_rows)

    assert summary["greedy_direct_only_vs_best_of_3"] == 1
    assert summary["self_consistency_only_vs_greedy_direct"] == 1
    assert summary["queries_changed_correctness_best_of_3_vs_greedy_reasoning"] == 1
    assert summary["queries_changed_correctness_self_consistency_vs_greedy_reasoning"] == 1


def test_summarize_method_results_computes_accuracy_and_samples() -> None:
    method_results = {
        "greedy_direct": [
            type("Result", (), {"correct": True, "samples_used": 1})(),
            type("Result", (), {"correct": False, "samples_used": 1})(),
        ],
        "best_of_3_reasoning": [
            type("Result", (), {"correct": True, "samples_used": 3})(),
            type("Result", (), {"correct": False, "samples_used": 3})(),
        ],
    }

    method_specs = [
        {"name": "greedy_direct", "prompt_style": "direct", "n_samples": 1},
        {
            "name": "best_of_3_reasoning",
            "prompt_style": "reasoning",
            "n_samples": 3,
        },
    ]

    summary = summarize_method_results(method_results, method_specs)

    assert summary[0]["method"] == "greedy_direct"
    assert summary[0]["accuracy"] == 0.5
    assert summary[0]["total_samples_used"] == 2
    assert summary[1]["method"] == "best_of_3_reasoning"
    assert summary[1]["avg_samples_per_query"] == 3.0


def test_prompt_constants_are_non_empty() -> None:
    assert DIRECT_PROMPT.strip()
    assert REASONING_PROMPT.strip()
