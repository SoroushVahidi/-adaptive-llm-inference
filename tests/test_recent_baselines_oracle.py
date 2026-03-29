"""Unit tests for oracle aggregation (no API calls)."""

from __future__ import annotations

from src.evaluation.recent_baselines_eval import compute_oracle_summaries


def _row(qid: str, correct: bool, cost: float) -> dict:
    return {
        "query_id": qid,
        "correct": correct,
        "samples_used": int(cost),
        "metadata": {"cost_proxy": cost},
        "candidates": [],
        "question": "",
        "final_answer": "",
        "ground_truth": "",
    }


def test_binary_oracle_picks_cheaper_when_both_correct() -> None:
    per = {
        "reasoning_greedy": [_row("a", True, 1.0), _row("b", False, 1.0)],
        "direct_plus_revise": [_row("a", True, 2.0), _row("b", True, 2.0)],
        "reasoning_then_revise": [],
        "self_consistency_3": [],
        "self_consistency_5": [],
    }
    out = compute_oracle_summaries(per)
    bin_o = out["binary_oracle"]
    assert bin_o["oracle_accuracy"] == 1.0
    assert bin_o["action_frequencies"]["reasoning_greedy"] == 1
    assert bin_o["action_frequencies"]["direct_plus_revise"] == 1


def test_multi_oracle_prefers_correct_over_wrong() -> None:
    per = {
        "reasoning_greedy": [_row("a", False, 1.0)],
        "direct_plus_revise": [_row("a", True, 2.0)],
        "reasoning_then_revise": [_row("a", False, 2.0)],
        "self_consistency_3": [_row("a", True, 3.0)],
        "self_consistency_5": [_row("a", True, 5.0)],
    }
    out = compute_oracle_summaries(per)
    m = out["multi_action_oracle"]
    assert m["oracle_accuracy"] == 1.0
    # Cheapest correct action among ties on accuracy
    assert m["action_frequencies"]["direct_plus_revise"] == 1
