from src.evaluation.real_llm_diagnostic import (
    DIRECT_PROMPT,
    REASONING_PROMPT,
    build_overlap_summary,
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
    assert summary["queries_changed_correctness_best_of_3_vs_greedy_reasoning"] == 2
    assert summary["queries_changed_correctness_self_consistency_vs_greedy_reasoning"] == 1


def test_prompt_constants_are_non_empty() -> None:
    assert DIRECT_PROMPT.strip()
    assert REASONING_PROMPT.strip()
