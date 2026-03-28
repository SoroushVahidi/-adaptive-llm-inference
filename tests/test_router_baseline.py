"""Unit tests for src/policies/router_baseline.py.

All tests are fully offline — no API calls, no oracle CSV files required
unless explicitly synthesised in a temporary directory.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from src.policies.router_baseline import (
    QUERY_FEATURE_COLS,
    EvalResult,
    MajorityBaseline,
    MinimalDecisionTree,
    _gini,
    _split,
    fit_and_evaluate,
    load_routing_csv,
    prepare_features,
    save_router_outputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_routing_csv(
    tmpdir: str | Path,
    n_rows: int = 10,
    oracle_available: bool = True,
) -> Path:
    """Write a minimal routing_dataset.csv fixture."""
    base = Path(tmpdir)
    p = base / "routing_dataset.csv"

    strategies = [
        "direct_greedy",
        "reasoning_best_of_3",
        "direct_plus_verify",
    ]
    fieldnames = (
        ["question_id", "question_text"]
        + QUERY_FEATURE_COLS
        + [
            "first_pass_parse_success",
            "oracle_label_available",
            "best_accuracy_strategy",
            "cheapest_correct_strategy",
            "direct_already_optimal",
            "oracle_any_correct",
            "num_strategies_correct",
        ]
    )

    with p.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_rows):
            row: dict = {
                "question_id": f"q{i}",
                "question_text": f"What is {i} + 1?",
                "question_length_chars": 15 + i,
                "question_length_tokens_approx": 5,
                "num_numeric_mentions": i % 3 + 1,
                "num_sentences_approx": 1,
                "has_multi_step_cue": i % 2,
                "has_equation_like_pattern": 0,
                "has_percent_symbol": 0,
                "has_fraction_pattern": 0,
                "has_currency_symbol": 0,
                "max_numeric_value_approx": float(i),
                "min_numeric_value_approx": 0.0,
                "numeric_range_approx": float(i),
                "repeated_number_flag": 0,
                "first_pass_parse_success": "",
                "oracle_label_available": oracle_available,
                "best_accuracy_strategy": strategies[i % len(strategies)],
                "cheapest_correct_strategy": "direct_greedy",
                "direct_already_optimal": i % 2,
                "oracle_any_correct": 1,
                "num_strategies_correct": i % 3,
            }
            writer.writerow(row)

    return p


# ---------------------------------------------------------------------------
# _gini
# ---------------------------------------------------------------------------


def test_gini_pure_class_is_zero() -> None:
    assert _gini(["a", "a", "a"]) == pytest.approx(0.0)


def test_gini_uniform_two_classes() -> None:
    assert _gini([0, 1, 0, 1]) == pytest.approx(0.5)


def test_gini_empty_is_zero() -> None:
    assert _gini([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _split
# ---------------------------------------------------------------------------


def test_split_tiny_dataset_uses_all() -> None:
    X = [[float(i)] for i in range(3)]
    y = list(range(3))
    X_tr, y_tr, X_te, y_te = _split(X, y)
    assert len(X_tr) == len(X_te) == 3


def test_split_normal_dataset() -> None:
    X = [[float(i)] for i in range(10)]
    y = list(range(10))
    X_tr, y_tr, X_te, y_te = _split(X, y)
    assert len(X_tr) + len(X_te) == 10
    assert len(X_te) >= 1


# ---------------------------------------------------------------------------
# MajorityBaseline
# ---------------------------------------------------------------------------


def test_majority_baseline_predicts_most_common() -> None:
    model = MajorityBaseline()
    model.fit([[0], [1], [2]], ["a", "a", "b"])
    preds = model.predict([[9], [10]])
    assert all(p == "a" for p in preds)


def test_majority_baseline_raises_on_empty() -> None:
    model = MajorityBaseline()
    with pytest.raises(ValueError):
        model.fit([], [])


def test_majority_baseline_predict_before_fit_raises() -> None:
    model = MajorityBaseline()
    with pytest.raises(RuntimeError):
        model.predict([[1]])


def test_majority_baseline_name() -> None:
    assert MajorityBaseline().name == "majority_baseline"


# ---------------------------------------------------------------------------
# MinimalDecisionTree
# ---------------------------------------------------------------------------


def test_minimal_dt_separable_data() -> None:
    X = [[0.0], [0.0], [1.0], [1.0]]
    y = ["a", "a", "b", "b"]
    model = MinimalDecisionTree(max_depth=3)
    model.fit(X, y)
    preds = model.predict([[0.0], [1.0]])
    assert preds[0] == "a"
    assert preds[1] == "b"


def test_minimal_dt_single_class() -> None:
    X = [[0.0], [1.0], [2.0]]
    y = ["x", "x", "x"]
    model = MinimalDecisionTree(max_depth=3)
    model.fit(X, y)
    preds = model.predict([[5.0]])
    assert preds[0] == "x"


def test_minimal_dt_feature_importances_keys() -> None:
    X = [[float(i), float(i * 2)] for i in range(6)]
    y = ["a", "b"] * 3
    model = MinimalDecisionTree(max_depth=3)
    model.fit(X, y, feature_names=["feat1", "feat2"])
    imps = model.feature_importances()
    assert all(isinstance(v, float) for v in imps.values())


def test_minimal_dt_predict_before_fit_raises() -> None:
    model = MinimalDecisionTree()
    with pytest.raises(RuntimeError):
        model.predict([[1.0]])


def test_minimal_dt_fit_empty_raises() -> None:
    model = MinimalDecisionTree()
    with pytest.raises(ValueError):
        model.fit([], [])


# ---------------------------------------------------------------------------
# load_routing_csv
# ---------------------------------------------------------------------------


def test_load_routing_csv_not_found_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_routing_csv("/tmp/does_not_exist_routing_xyz.csv")


def test_load_routing_csv_returns_rows_and_oracle_flag() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = _make_routing_csv(tmpdir, n_rows=5, oracle_available=True)
        rows, oracle_available = load_routing_csv(p)

    assert len(rows) == 5
    assert oracle_available is True


def test_load_routing_csv_oracle_unavailable_when_all_false() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = _make_routing_csv(tmpdir, n_rows=3, oracle_available=False)
        rows, oracle_available = load_routing_csv(p)

    assert oracle_available is False


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------


def test_prepare_features_shape() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = _make_routing_csv(tmpdir, n_rows=5)
        rows, _ = load_routing_csv(p)

    X, names = prepare_features(rows)
    assert len(X) == 5
    assert len(X[0]) == len(QUERY_FEATURE_COLS)
    assert names == QUERY_FEATURE_COLS


def test_prepare_features_all_float() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = _make_routing_csv(tmpdir, n_rows=4)
        rows, _ = load_routing_csv(p)

    X, _ = prepare_features(rows)
    for row in X:
        for val in row:
            assert isinstance(val, float)


def test_prepare_features_bool_coercion() -> None:
    rows = [{"has_multi_step_cue": "True", "has_currency_symbol": "False"}]
    rows_augmented = [dict(r, **{c: 0 for c in QUERY_FEATURE_COLS}) for r in rows]
    rows_augmented[0]["has_multi_step_cue"] = "True"
    rows_augmented[0]["has_currency_symbol"] = "False"
    X, names = prepare_features(rows_augmented)
    idx_multi = names.index("has_multi_step_cue")
    idx_curr = names.index("has_currency_symbol")
    assert X[0][idx_multi] == 1.0
    assert X[0][idx_curr] == 0.0


# ---------------------------------------------------------------------------
# fit_and_evaluate
# ---------------------------------------------------------------------------


def test_fit_and_evaluate_returns_results() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = _make_routing_csv(tmpdir, n_rows=10, oracle_available=True)
        rows, _ = load_routing_csv(p)

    labeled = [r for r in rows if str(r.get("oracle_label_available", "")).lower() == "true"]
    X, feature_names = prepare_features(labeled)
    y = [str(r["direct_already_optimal"]) for r in labeled]
    results = fit_and_evaluate(X, y, "binary", feature_names)
    assert len(results) >= 1
    assert all(isinstance(r, EvalResult) for r in results)


def test_fit_and_evaluate_accuracy_in_range() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = _make_routing_csv(tmpdir, n_rows=10, oracle_available=True)
        rows, _ = load_routing_csv(p)

    labeled = [r for r in rows if str(r.get("oracle_label_available", "")).lower() == "true"]
    X, feature_names = prepare_features(labeled)
    y = [str(r["direct_already_optimal"]) for r in labeled]
    results = fit_and_evaluate(X, y, "binary", feature_names)
    for r in results:
        assert 0.0 <= r.accuracy <= 1.0


def test_fit_and_evaluate_majority_baseline_present() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = _make_routing_csv(tmpdir, n_rows=10, oracle_available=True)
        rows, _ = load_routing_csv(p)

    X, feature_names = prepare_features(rows)
    y = ["a"] * 5 + ["b"] * 5
    results = fit_and_evaluate(X, y, "binary", feature_names)
    names = [r.model_name for r in results]
    assert any("majority" in n for n in names)


def test_fit_and_evaluate_empty_returns_empty() -> None:
    results = fit_and_evaluate([], [], "binary", [])
    assert results == []


def test_fit_and_evaluate_multiclass_task() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = _make_routing_csv(tmpdir, n_rows=12, oracle_available=True)
        rows, _ = load_routing_csv(p)

    labeled = [r for r in rows if str(r.get("oracle_label_available", "")).lower() == "true"]
    X, feature_names = prepare_features(labeled)
    y = [r.get("best_accuracy_strategy", "") for r in labeled]
    y = [v for v in y if v]
    X = X[: len(y)]
    results = fit_and_evaluate(X, y, "multiclass", feature_names)
    assert len(results) >= 1


def test_fit_and_evaluate_predictions_have_expected_keys() -> None:
    X = [[float(i), float(i % 2)] for i in range(8)]
    y = ["0"] * 4 + ["1"] * 4
    results = fit_and_evaluate(X, y, "binary", ["f1", "f2"])
    for r in results:
        if r.predictions:
            assert "true_label" in r.predictions[0]
            assert "predicted_label" in r.predictions[0]


# ---------------------------------------------------------------------------
# save_router_outputs
# ---------------------------------------------------------------------------


def test_save_router_outputs_creates_files() -> None:
    X = [[float(i), float(i % 2)] for i in range(8)]
    y = ["0"] * 4 + ["1"] * 4
    binary_results = fit_and_evaluate(X, y, "binary", ["f1", "f2"])
    y_m = ["s1", "s2", "s1", "s2"] * 2
    multi_results = fit_and_evaluate(X, y_m, "multiclass", ["f1", "f2"])

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_router_outputs(binary_results, multi_results, tmpdir)
        for key in ("summary", "binary_predictions", "multiclass_predictions"):
            assert key in paths
            assert Path(paths[key]).exists()


def test_save_router_outputs_summary_keys() -> None:
    binary_results: list[EvalResult] = []
    multi_results: list[EvalResult] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_router_outputs(binary_results, multi_results, tmpdir)
        summary = json.loads(Path(paths["summary"]).read_text())

    required_keys = ["binary_task", "multiclass_task", "sklearn_available"]
    for key in required_keys:
        assert key in summary, f"Missing key: {key}"


def test_save_router_outputs_summary_sklearn_available() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_router_outputs([], [], tmpdir)
        summary = json.loads(Path(paths["summary"]).read_text())
    assert isinstance(summary["sklearn_available"], bool)


def test_save_router_outputs_binary_csv_has_columns() -> None:
    X = [[float(i)] for i in range(6)]
    y = ["0", "1"] * 3
    results = fit_and_evaluate(X, y, "binary", ["feat"])

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_router_outputs(results, [], tmpdir)
        with open(paths["binary_predictions"]) as fh:
            header = csv.DictReader(fh).fieldnames or []

    assert "true_label" in header
    assert "predicted_label" in header
    assert "correct" in header


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


def test_eval_result_to_dict_has_required_keys() -> None:
    r = EvalResult(
        task="binary",
        model_name="test_model",
        n_train=8,
        n_test=2,
        accuracy=0.75,
    )
    d = r.to_dict()
    for key in ("task", "model_name", "n_train", "n_test", "accuracy",
                "class_distribution", "feature_importances", "note"):
        assert key in d, f"Missing key in to_dict(): {key}"


def test_eval_result_accuracy_field() -> None:
    r = EvalResult(task="binary", model_name="m", n_train=5, n_test=2, accuracy=0.5)
    assert r.accuracy == 0.5
