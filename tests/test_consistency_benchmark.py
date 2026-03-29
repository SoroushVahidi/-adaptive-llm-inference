from __future__ import annotations

from src.analysis.consistency_benchmark import (
    FAILURE_TYPES,
    evaluate_benchmark,
    evaluate_candidate,
    evaluate_candidate_with_calibrated_role_features,
    evaluate_candidate_with_role_features,
    evaluate_candidate_with_unified_error,
    flatten_candidates,
    load_benchmark,
)


def test_benchmark_file_loads_expected_size() -> None:
    rows = load_benchmark("data/consistency_benchmark.json")
    assert len(rows) == 20


def test_flatten_candidates_has_correct_and_wrong_records() -> None:
    rows = load_benchmark("data/consistency_benchmark.json")
    flat = flatten_candidates(rows)
    assert len(flat) == 60
    assert any(r.is_correct for r in flat)
    assert any(not r.is_correct for r in flat)


def test_failure_types_are_all_present_in_benchmark() -> None:
    rows = load_benchmark("data/consistency_benchmark.json")
    flat = flatten_candidates(rows)
    present = {r.failure_type for r in flat if r.failure_type}
    assert FAILURE_TYPES.issubset(present)


def test_evaluate_candidate_flags_floor_ceiling_conflict() -> None:
    question = (
        "A van holds at most 7 students each. There are 31 students. "
        "What is the minimum number of vans needed?"
    )
    out = evaluate_candidate(question, "4")
    assert out["flagged"] is True
    assert "floor_ceiling_conflict" in out["triggered_signals"]


def test_role_augmented_checker_adds_role_signals() -> None:
    question = "A bus has 40 seats. 26 are occupied. How many seats are left?"
    out = evaluate_candidate_with_role_features(question, "26")
    assert out["flagged"] is True
    assert out["role_features"]["missing_required_number_count"] >= 1


def test_calibrated_checker_exposes_decision() -> None:
    question = "A bus has 40 seats. 26 are occupied. How many seats are left?"
    out = evaluate_candidate_with_calibrated_role_features(question, "26")
    assert out["calibrated"]["calibrated_decision"] in {
        "no_escalation",
        "maybe_escalate",
        "strong_escalation_candidate",
    }


def test_evaluate_benchmark_metrics_have_valid_bounds() -> None:
    rows = load_benchmark("data/consistency_benchmark.json")
    ev = evaluate_benchmark(rows)
    for variant in (
        "old_checker",
        "raw_role_checker",
        "calibrated_role_checker",
        "unified_error_checker",
    ):
        assert 0.0 <= ev["variant_metrics"][variant]["wrong_recall"] <= 1.0
        assert 0.0 <= ev["variant_metrics"][variant]["false_positive_rate_on_correct"] <= 1.0
    assert len(ev["recall_by_failure_type"]) == len(FAILURE_TYPES)


def test_unified_checker_returns_scores() -> None:
    question = "A bus has 40 seats. 26 are occupied. How many seats are left?"
    out = evaluate_candidate_with_unified_error(question, "26")
    assert "unified" in out
    assert 0.0 <= out["unified"]["unified_error_score"] <= 1.0
