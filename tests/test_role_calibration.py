from __future__ import annotations

from src.analysis.consistency_benchmark import load_benchmark
from src.analysis.role_calibration import analyze_false_positives, build_signal_tradeoff_summary


def test_false_positive_analysis_has_expected_sections() -> None:
    from src.analysis.consistency_benchmark import evaluate_benchmark

    ev = evaluate_benchmark(load_benchmark("data/consistency_benchmark.json"))
    fp = analyze_false_positives(ev)
    assert "false_positives_by_signal" in fp
    assert "false_positives_by_question_pattern" in fp
    assert "false_positive_likely_causes" in fp


def test_tradeoff_summary_lists_three_variants() -> None:
    from src.analysis.consistency_benchmark import evaluate_benchmark

    ev = evaluate_benchmark(load_benchmark("data/consistency_benchmark.json"))
    rows = build_signal_tradeoff_summary(ev)
    names = {r["variant"] for r in rows}
    assert names == {
        "old_checker",
        "raw_role_checker",
        "calibrated_role_checker",
        "unified_error_checker",
    }
