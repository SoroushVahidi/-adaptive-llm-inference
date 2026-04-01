"""Tests for src/routing/learned_router/*.

All tests are fully offline — no API calls required.  Routing datasets are
synthesised in temporary directories.

Test coverage:
- features.py: load_regime_df, build_feature_matrix, build_training_dataset
- models.py: LogisticRegressionRouter (fit, predict_proba, save, load, best_threshold)
- train.py: train() end-to-end with a synthetic dataset
- eval.py: evaluate() end-to-end with a saved model
- LearnedRouterPolicy: decide_batch, with_threshold, from_saved
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.routing.learned_router.features import (
    FEATURE_COLS,
    REGIME_FILES,
    build_feature_matrix,
    build_training_dataset,
    load_regime_df,
)
from src.routing.learned_router.models import (
    LogisticRegressionRouter,
    make_router,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_routing_df(n: int = 50, seed: int = 0, pos_rate: float = 0.3) -> pd.DataFrame:
    """Synthesise a minimal enriched routing DataFrame."""
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 1, n) for col in FEATURE_COLS}
    data["question_id"] = [f"q{i}" for i in range(n)]
    data["reasoning_correct"] = rng.integers(0, 2, n)
    data["revise_correct"] = rng.integers(0, 2, n)
    # Ensure some positives (escalate) and some negatives
    n_pos = max(2, int(n * pos_rate))
    labels = np.zeros(n, dtype=int)
    labels[:n_pos] = 1
    rng.shuffle(labels)
    data["reasoning_correct"] = 1 - labels  # RG wrong when escalate=1
    data["revise_correct"] = labels  # DPR correct when escalate=1
    return pd.DataFrame(data)


def _write_routing_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


# ===========================================================================
# features.py
# ===========================================================================


class TestLoadRegimeDf:
    def test_label_derived_correctly(self) -> None:
        df_in = _make_routing_df(n=20, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "regime.csv"
            df_in.to_csv(p, index=False)
            df_out = load_regime_df(p)

        expected = (
            (df_in["revise_correct"] == 1) & (df_in["reasoning_correct"] == 0)
        ).astype(int)
        assert (df_out["escalate_label"].values == expected.values).all()

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_regime_df("/tmp/nonexistent_routing_xyz.csv")

    def test_missing_required_columns_raises(self) -> None:
        df = pd.DataFrame({"question_id": ["q1"]})
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            df.to_csv(p, index=False)
            with pytest.raises(ValueError, match="Required columns"):
                load_regime_df(p)

    def test_returns_dataframe_with_label_col(self) -> None:
        df_in = _make_routing_df(n=10)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.csv"
            df_in.to_csv(p, index=False)
            df_out = load_regime_df(p)
        assert "escalate_label" in df_out.columns


class TestBuildFeatureMatrix:
    def test_shape(self) -> None:
        df = _make_routing_df(n=30)
        df["escalate_label"] = 0
        X, y, names = build_feature_matrix(df)
        assert X.shape == (30, len(FEATURE_COLS))
        assert y.shape == (30,)
        assert len(names) == len(FEATURE_COLS)

    def test_missing_cols_zero_filled(self) -> None:
        df = pd.DataFrame({"reasoning_correct": [0, 1], "revise_correct": [1, 0], "escalate_label": [1, 0]})
        X, y, names = build_feature_matrix(df)
        assert X.shape[1] == len(FEATURE_COLS)
        # Missing features should be 0
        assert np.all(X == 0.0)

    def test_dtype_float32(self) -> None:
        df = _make_routing_df(n=10)
        df["escalate_label"] = 0
        X, _, _ = build_feature_matrix(df)
        assert X.dtype == np.float32

    def test_custom_feature_cols(self) -> None:
        df = _make_routing_df(n=10)
        df["escalate_label"] = 0
        X, y, names = build_feature_matrix(df, feature_cols=["q_question_length_chars", "unified_confidence_score"])
        assert X.shape == (10, 2)
        assert names == ["q_question_length_chars", "unified_confidence_score"]


class TestBuildTrainingDataset:
    def test_leave_one_out_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {}
            for regime in ["a", "b", "c"]:
                p = Path(tmp) / f"{regime}.csv"
                _write_routing_csv(_make_routing_df(n=20, seed=hash(regime) % 100), p)
                regime_files[regime] = str(p)

            X_tr, y_tr, X_v, y_v, X_te, y_te, _ = build_training_dataset(
                regime_files=regime_files,
                test_regime="a",
                val_fraction=0.2,
                random_seed=0,
            )

        # Test set = regime "a" = 20 rows
        assert len(X_te) == 20
        # Train+val = regimes "b" + "c" = 40 rows (before split)
        assert len(X_tr) + len(X_v) == 40

    def test_no_test_regime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {}
            for regime in ["x", "y"]:
                p = Path(tmp) / f"{regime}.csv"
                _write_routing_csv(_make_routing_df(n=20), p)
                regime_files[regime] = str(p)

            X_tr, y_tr, X_v, y_v, X_te, y_te, _ = build_training_dataset(
                regime_files=regime_files,
                test_regime=None,
            )

        assert len(X_te) == 0
        assert len(X_tr) + len(X_v) == 40

    def test_returns_correct_feature_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {}
            for regime in ["p", "q"]:
                p = Path(tmp) / f"{regime}.csv"
                _write_routing_csv(_make_routing_df(n=10), p)
                regime_files[regime] = str(p)

            _, _, _, _, _, _, names = build_training_dataset(regime_files=regime_files)

        assert names == FEATURE_COLS


# ===========================================================================
# models.py
# ===========================================================================


class TestLogisticRegressionRouter:
    def _make_data(self, n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n, len(FEATURE_COLS))).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(int)
        return X, y

    def test_fit_and_predict_proba(self) -> None:
        X, y = self._make_data()
        router = LogisticRegressionRouter()
        router.fit(X, y)
        proba = router.predict_proba(X)
        assert proba.shape == (len(X),)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_binary_output(self) -> None:
        X, y = self._make_data()
        router = LogisticRegressionRouter()
        router.fit(X, y)
        preds = router.predict(X, threshold=0.5)
        assert set(np.unique(preds)) <= {0, 1}

    def test_fit_empty_raises(self) -> None:
        router = LogisticRegressionRouter()
        with pytest.raises((ValueError, ImportError)):
            router.fit(np.empty((0, 5), dtype=np.float32), np.array([], dtype=int))

    def test_predict_before_fit_raises(self) -> None:
        router = LogisticRegressionRouter()
        with pytest.raises(RuntimeError):
            router.predict_proba(np.zeros((1, 5), dtype=np.float32))

    def test_save_and_load(self) -> None:
        X, y = self._make_data()
        router = LogisticRegressionRouter()
        router.fit(X, y)
        original_proba = router.predict_proba(X[:5])

        with tempfile.TemporaryDirectory() as tmp:
            router.save(Path(tmp) / "model")
            loaded = LogisticRegressionRouter.load(Path(tmp) / "model")

        loaded_proba = loaded.predict_proba(X[:5])
        np.testing.assert_allclose(original_proba, loaded_proba, rtol=1e-5)

    def test_best_threshold_returns_float_in_range(self) -> None:
        X, y = self._make_data()
        router = LogisticRegressionRouter()
        router.fit(X, y)
        tau = router.best_threshold(X, y)
        assert 0.0 <= tau <= 1.0

    def test_make_router_factory(self) -> None:
        router = make_router("logistic_regression", C=0.5)
        assert isinstance(router, LogisticRegressionRouter)
        assert router.C == 0.5

    def test_make_router_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            make_router("nonexistent_model")

    def test_accuracy_better_than_random_on_separable(self) -> None:
        rng = np.random.default_rng(99)
        # Perfectly separable dataset (feature 0 threshold at 0.5)
        X = rng.uniform(0, 1, (100, len(FEATURE_COLS))).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(int)
        router = LogisticRegressionRouter()
        router.fit(X, y)
        preds = router.predict(X, threshold=0.5)
        accuracy = float(np.mean(preds == y))
        assert accuracy > 0.6, f"Expected >60% accuracy on separable data, got {accuracy:.2f}"

    def test_fit_with_val_set_runs(self) -> None:
        X, y = self._make_data()
        router = LogisticRegressionRouter()
        router.fit(X[:40], y[:40], X[40:], y[40:])
        proba = router.predict_proba(X[40:])
        assert proba.shape == (20,)


# ===========================================================================
# train.py
# ===========================================================================


class TestTrain:
    def test_train_end_to_end(self) -> None:
        from src.routing.learned_router.train import train

        with tempfile.TemporaryDirectory() as tmp:
            # Write synthetic regime files
            regime_files = {}
            for regime in ["a", "b", "c"]:
                p = Path(tmp) / f"{regime}.csv"
                _write_routing_csv(_make_routing_df(n=30, seed=hash(regime) % 100), p)
                regime_files[regime] = str(p)

            cfg = {
                "model_type": "logistic_regression",
                "test_regime": "a",
                "val_fraction": 0.2,
                "random_seed": 42,
                "output_dir": str(Path(tmp) / "out"),
                "logreg": {"C": 1.0, "max_iter": 100},
                "threshold_sweep": {"start": 0.1, "stop": 0.9, "step": 0.1},
                "cheap_cost": 1.0,
                "expensive_cost": 2.0,
                "feature_cols": None,
            }
            # Override REGIME_FILES for this test
            import src.routing.learned_router.features as feat_mod
            orig = feat_mod.REGIME_FILES
            feat_mod.REGIME_FILES = regime_files
            try:
                summary = train(cfg)
            finally:
                feat_mod.REGIME_FILES = orig

        assert "selected_threshold" in summary
        assert 0.0 <= summary["selected_threshold"] <= 1.0
        assert summary["model_type"] == "logistic_regression"
        assert summary["test_n"] == 30  # regime "a" has 30 rows

    def test_train_creates_output_files(self) -> None:
        from src.routing.learned_router.train import train

        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {}
            for regime in ["x", "y"]:
                p = Path(tmp) / f"{regime}.csv"
                _write_routing_csv(_make_routing_df(n=25), p)
                regime_files[regime] = str(p)

            out_dir = Path(tmp) / "output"
            cfg = {
                "model_type": "logistic_regression",
                "test_regime": None,
                "val_fraction": 0.25,
                "random_seed": 0,
                "output_dir": str(out_dir),
                "logreg": {"C": 1.0, "max_iter": 50},
                "threshold_sweep": {"start": 0.2, "stop": 0.8, "step": 0.2},
                "cheap_cost": 1.0,
                "expensive_cost": 2.0,
                "feature_cols": None,
            }
            import src.routing.learned_router.features as feat_mod
            orig = feat_mod.REGIME_FILES
            feat_mod.REGIME_FILES = regime_files
            try:
                train(cfg)
            finally:
                feat_mod.REGIME_FILES = orig

            assert (out_dir / "train_summary.json").exists()
            assert (out_dir / "model" / "model.pkl").exists()
            assert (out_dir / "model" / "meta.json").exists()
            assert (out_dir / "threshold_sweep.csv").exists()


# ===========================================================================
# eval.py
# ===========================================================================


class TestEvaluate:
    def _setup_trained_model(self, tmp: str, regime_files: dict) -> dict:
        """Train a model and return the config."""
        from src.routing.learned_router.train import train
        import src.routing.learned_router.features as feat_mod

        out_dir = Path(tmp) / "model_output"
        cfg = {
            "model_type": "logistic_regression",
            "test_regime": None,
            "val_fraction": 0.2,
            "random_seed": 0,
            "output_dir": str(out_dir),
            "logreg": {"C": 1.0, "max_iter": 50},
            "threshold_sweep": {"start": 0.2, "stop": 0.8, "step": 0.2},
            "cheap_cost": 1.0,
            "expensive_cost": 2.0,
            "feature_cols": None,
        }
        orig = feat_mod.REGIME_FILES
        feat_mod.REGIME_FILES = regime_files
        try:
            train(cfg)
        finally:
            feat_mod.REGIME_FILES = orig
        return cfg

    def test_evaluate_end_to_end(self) -> None:
        from src.routing.learned_router.eval import evaluate
        import src.routing.learned_router.features as feat_mod

        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {}
            for regime in ["p", "q", "r"]:
                path = Path(tmp) / f"{regime}.csv"
                _write_routing_csv(_make_routing_df(n=20, seed=hash(regime) % 100), path)
                regime_files[regime] = str(path)

            cfg = self._setup_trained_model(tmp, regime_files)

            orig = feat_mod.REGIME_FILES
            feat_mod.REGIME_FILES = regime_files
            try:
                results = evaluate(cfg)
            finally:
                feat_mod.REGIME_FILES = orig

        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.accuracy <= 1.0
            assert 1.0 <= r.avg_cost <= 2.0
            assert 0.0 <= r.revise_rate <= 1.0

    def test_evaluate_creates_output_files(self) -> None:
        from src.routing.learned_router.eval import evaluate
        import src.routing.learned_router.features as feat_mod

        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {"alpha": str(Path(tmp) / "alpha.csv")}
            _write_routing_csv(_make_routing_df(n=20), Path(regime_files["alpha"]))

            cfg = self._setup_trained_model(tmp, regime_files)

            orig = feat_mod.REGIME_FILES
            feat_mod.REGIME_FILES = regime_files
            try:
                evaluate(cfg)
            finally:
                feat_mod.REGIME_FILES = orig

            eval_dir = Path(cfg["output_dir"]) / "eval"
            assert (eval_dir / "learned_router_eval_summary.json").exists()
            assert (eval_dir / "learned_router_eval_summary.csv").exists()
            assert (eval_dir / "alpha" / "metrics.json").exists()
            assert (eval_dir / "alpha" / "budget_curve.csv").exists()
            assert (eval_dir / "alpha" / "per_query_decisions.csv").exists()


# ===========================================================================
# LearnedRouterPolicy
# ===========================================================================


class TestLearnedRouterPolicy:
    def _trained_policy(self, tmp: str, regime_files: dict) -> "LearnedRouterPolicy":
        from src.routing.learned_router.train import train
        import src.routing.learned_router.features as feat_mod
        from src.policies.learned_router_policy import LearnedRouterPolicy

        out_dir = Path(tmp) / "policy_model"
        cfg = {
            "model_type": "logistic_regression",
            "test_regime": None,
            "val_fraction": 0.2,
            "random_seed": 0,
            "output_dir": str(out_dir),
            "logreg": {"C": 1.0, "max_iter": 50},
            "threshold_sweep": {"start": 0.2, "stop": 0.8, "step": 0.2},
            "cheap_cost": 1.0,
            "expensive_cost": 2.0,
            "feature_cols": None,
        }
        orig = feat_mod.REGIME_FILES
        feat_mod.REGIME_FILES = regime_files
        try:
            train(cfg)
        finally:
            feat_mod.REGIME_FILES = orig

        return LearnedRouterPolicy.from_train_output(out_dir)

    def test_from_train_output(self) -> None:
        from src.policies.learned_router_policy import LearnedRouterPolicy

        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {}
            for regime in ["a", "b"]:
                p = Path(tmp) / f"{regime}.csv"
                _write_routing_csv(_make_routing_df(n=20), p)
                regime_files[regime] = str(p)

            policy = self._trained_policy(tmp, regime_files)

        assert isinstance(policy, LearnedRouterPolicy)
        assert 0.0 <= policy.threshold <= 1.0

    def test_decide_batch_returns_valid_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {}
            for regime in ["a", "b"]:
                p = Path(tmp) / f"{regime}.csv"
                _write_routing_csv(_make_routing_df(n=20), p)
                regime_files[regime] = str(p)

            policy = self._trained_policy(tmp, regime_files)

        df = _make_routing_df(n=10)
        decisions = policy.decide_batch(df)
        assert len(decisions) == 10
        valid = {"reasoning_greedy", "direct_plus_revise"}
        assert set(decisions.unique()) <= valid

    def test_with_threshold_does_not_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {"x": str(Path(tmp) / "x.csv")}
            _write_routing_csv(_make_routing_df(n=20), Path(regime_files["x"]))
            policy = self._trained_policy(tmp, regime_files)

        original_tau = policy.threshold
        new_policy = policy.with_threshold(0.9)
        assert policy.threshold == original_tau
        assert new_policy.threshold == 0.9

    def test_predict_proba_batch_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {"y": str(Path(tmp) / "y.csv")}
            _write_routing_csv(_make_routing_df(n=20), Path(regime_files["y"]))
            policy = self._trained_policy(tmp, regime_files)

        df = _make_routing_df(n=15)
        proba = policy.predict_proba_batch(df)
        assert proba.shape == (15,)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_from_saved_loads_correct_model(self) -> None:
        from src.policies.learned_router_policy import LearnedRouterPolicy

        with tempfile.TemporaryDirectory() as tmp:
            regime_files = {}
            for regime in ["a", "b"]:
                p = Path(tmp) / f"{regime}.csv"
                _write_routing_csv(_make_routing_df(n=20), p)
                regime_files[regime] = str(p)

            policy_orig = self._trained_policy(tmp, regime_files)
            model_dir = Path(tmp) / "policy_model" / "model"

            policy_loaded = LearnedRouterPolicy.from_saved(
                model_dir, model_type="logistic_regression", threshold=0.6
            )

        assert policy_loaded.threshold == 0.6
        df = _make_routing_df(n=5)
        proba_orig = policy_orig.predict_proba_batch(df)
        proba_loaded = policy_loaded.predict_proba_batch(df)
        np.testing.assert_allclose(proba_orig, proba_loaded, rtol=1e-5)

    def test_policy_registered_in_init(self) -> None:
        from src.policies import LearnedRouterPolicy as LRP
        assert LRP is not None

    def test_policy_name(self) -> None:
        from src.policies.learned_router_policy import LearnedRouterPolicy
        assert LearnedRouterPolicy.name == "learned_router"


# ===========================================================================
# Real regime files (online tests — skipped when files missing)
# ===========================================================================


@pytest.mark.skipif(
    not all((_REPO_ROOT / v).exists() for v in REGIME_FILES.values()),
    reason="Real enriched routing CSVs not available",
)
class TestRealRegimeFiles:
    def test_load_all_regimes(self) -> None:
        for regime, path in REGIME_FILES.items():
            df = load_regime_df(_REPO_ROOT / path)
            assert len(df) == 100, f"Expected 100 rows for {regime}, got {len(df)}"
            assert "escalate_label" in df.columns

    def test_build_training_dataset_real(self) -> None:
        regime_files = {k: str(_REPO_ROOT / v) for k, v in REGIME_FILES.items()}
        X_tr, y_tr, X_v, y_v, X_te, y_te, names = build_training_dataset(
            regime_files=regime_files,
            test_regime="hard_gsm8k_100",
            val_fraction=0.2,
        )
        # Test set = hard_gsm8k_100 = 100 rows
        assert len(X_te) == 100
        # Training pool = remaining 3 regimes = 300 rows (before val split)
        assert len(X_tr) + len(X_v) == 300
        assert len(names) == len(FEATURE_COLS)

    def test_logistic_regression_on_real_data(self) -> None:
        regime_files = {k: str(_REPO_ROOT / v) for k, v in REGIME_FILES.items()}
        X_tr, y_tr, X_v, y_v, X_te, y_te, names = build_training_dataset(
            regime_files=regime_files,
            test_regime="hard_gsm8k_100",
        )
        router = LogisticRegressionRouter()
        router.fit(X_tr, y_tr, X_v, y_v)
        proba = router.predict_proba(X_te)
        assert proba.shape == (100,)
        assert np.all(proba >= 0) and np.all(proba <= 1)
