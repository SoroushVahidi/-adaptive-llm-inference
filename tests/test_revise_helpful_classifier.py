from __future__ import annotations

from src.evaluation.revise_helpful_classifier_eval import (
    build_training_rows_from_benchmark,
    run_revise_helpful_classifier_eval,
)
from src.models.revise_helpful_classifier import compute_binary_metrics, detect_sklearn_support


def test_build_training_rows_has_expected_shape() -> None:
    rows = build_training_rows_from_benchmark("data/consistency_benchmark.json")
    assert len(rows) == 60
    assert any(int(r["revise_helpful"]) == 1 for r in rows)
    assert any(int(r["revise_helpful"]) == 0 for r in rows)
    sample = rows[0]
    assert "unified_error_score" in sample
    assert "role_warning_score" in sample


def test_binary_metrics_smoke() -> None:
    metrics = compute_binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics.accuracy == 0.5
    assert metrics.recall == 0.5
    assert metrics.false_positive_rate == 0.5


def test_run_eval_writes_outputs_even_when_blocked(tmp_path) -> None:
    out_dir = tmp_path / "revise"
    result = run_revise_helpful_classifier_eval(output_dir=out_dir)
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "model_metrics.csv").exists()
    assert (out_dir / "per_query_predictions.csv").exists()
    assert (out_dir / "routing_simulation.csv").exists()
    assert (out_dir / "feature_importance.csv").exists()

    support = detect_sklearn_support()
    if support.available:
        assert result["summary"]["run_status"] == "OK"
        assert "best_model" in result["summary"]
    else:
        assert result["summary"]["run_status"] == "BLOCKED"
        assert "block_reason" in result["summary"]
