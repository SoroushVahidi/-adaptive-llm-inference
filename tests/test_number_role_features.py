from __future__ import annotations

from src.features.number_role_features import (
    assign_number_roles,
    compute_calibrated_role_decision,
    compute_role_coverage_features,
    extract_problem_numbers,
)


def test_extract_problem_numbers_handles_digits_and_number_words() -> None:
    q = "Tom had 12 apples and gave away two. Then he bought 3 more."
    rows = extract_problem_numbers(q)
    surfaces = {r["surface_text"] for r in rows}
    assert "12" in surfaces
    assert "two" in surfaces
    assert "3" in surfaces


def test_assign_number_roles_detects_subtractive_and_additive_roles() -> None:
    q = "There were 45 apples. 18 were sold and 7 were added. How many remain?"
    roles = assign_number_roles(q)
    by_val = {str(r["surface_text"]): r for r in roles}
    assert by_val["18"]["expected_role"] == "subtract"
    assert by_val["18"]["strongly_required_for_final_answer"] is True


def test_compute_role_coverage_flags_missing_required_number() -> None:
    q = "A bus has 40 seats. 26 are occupied. How many seats are left?"
    reasoning = "Final answer: 26 seats are occupied."
    feats = compute_role_coverage_features(q, reasoning, parsed_answer="26")
    assert feats["missing_required_number_count"] >= 1
    assert feats["required_subtractive_number_missing"] is True


def test_compute_role_coverage_all_required_covered_for_complete_reasoning() -> None:
    q = "A machine makes 9 parts per hour for 7 hours. How many in total?"
    reasoning = "9 per hour for 7 hours gives 63. Final answer: 63"
    feats = compute_role_coverage_features(q, reasoning, parsed_answer="63")
    assert feats["num_required_numbers"] >= 2
    assert feats["all_required_numbers_covered"] is True


def test_calibrated_decision_reduces_single_weak_missing_signal() -> None:
    q = "A machine makes 9 parts per hour for 7 hours. How many in total?"
    out = compute_calibrated_role_decision(q, "Final answer: 63", parsed_answer="63")
    assert out["calibrated_decision"] in {"no_escalation", "maybe_escalate"}
    assert out["role_warning_score"] >= 0
