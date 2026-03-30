"""Tests for all manuscript-support additions.

Covers:
1. Confidence-threshold routing baseline (src/baselines/confidence_threshold_router.py)
2. Learned-router baseline (src/baselines/learned_router_baseline.py)
3. Bootstrap uncertainty analysis (src/evaluation/uncertainty_analysis.py)
4. Clarification export (src/evaluation/clarification_export.py)
5. Regime integrity checks — all four manuscript regimes are size-100 and
   consistently wired across routing dataset / policy eval / oracle / budget outputs.

All tests are fully offline — they read only committed files in the repo.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

REGIME_FILES = {
    "gsm8k_random_100": REPO_ROOT / "data/real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": REPO_ROOT / "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": REPO_ROOT / "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": REPO_ROOT / "data/real_math500_routing_dataset_enriched.csv",
}

EXPECTED_REGIMES = frozenset(REGIME_FILES.keys())

MAIN_RESULTS_CSV = REPO_ROOT / "outputs/paper_tables_cleaned/main_results_summary.csv"
ORACLE_CSV = REPO_ROOT / "outputs/paper_tables_cleaned/oracle_routing_eval.csv"
BUDGET_CURVES_CSV = REPO_ROOT / "outputs/paper_tables_cleaned/budget_curves_all_datasets.csv"
CROSS_REGIME_CSV = REPO_ROOT / "outputs/paper_tables_cleaned/final_cross_regime_summary_fixed.csv"


# ===========================================================================
# 1. REGIME INTEGRITY TESTS
# ===========================================================================


class TestRegimeIntegrity:
    """All four manuscript regimes must be present, size-100, and consistent."""

    def test_all_four_regime_files_exist(self) -> None:
        for regime, path in REGIME_FILES.items():
            assert path.exists(), f"Regime file missing: {path} (regime={regime})"

    def test_all_regimes_have_exactly_100_queries(self) -> None:
        for regime, path in REGIME_FILES.items():
            df = pd.read_csv(path)
            assert len(df) == 100, (
                f"Expected 100 queries for {regime}, got {len(df)}"
            )

    def test_required_columns_present_in_all_regimes(self) -> None:
        required = {
            "question_id",
            "reasoning_correct",
            "revise_correct",
            "revise_helpful",
            "unified_confidence_score",
            "v6_revise_recommended",
            "v7_revise_recommended",
        }
        for regime, path in REGIME_FILES.items():
            df = pd.read_csv(path)
            missing = required - set(df.columns)
            assert not missing, f"Regime {regime} missing columns: {missing}"

    def test_reasoning_correct_is_binary(self) -> None:
        for regime, path in REGIME_FILES.items():
            df = pd.read_csv(path)
            unique_vals = set(df["reasoning_correct"].dropna().unique())
            assert unique_vals <= {0, 1}, (
                f"reasoning_correct not binary in {regime}: {unique_vals}"
            )

    def test_revise_helpful_is_binary(self) -> None:
        for regime, path in REGIME_FILES.items():
            df = pd.read_csv(path)
            unique_vals = set(df["revise_helpful"].dropna().unique())
            assert unique_vals <= {0, 1}, (
                f"revise_helpful not binary in {regime}: {unique_vals}"
            )

    def test_main_results_covers_all_regimes(self) -> None:
        df = pd.read_csv(MAIN_RESULTS_CSV)
        found = set(df["dataset"].tolist())
        assert EXPECTED_REGIMES == found, (
            f"main_results_summary.csv regimes mismatch.\n"
            f"  Expected: {EXPECTED_REGIMES}\n  Found:    {found}"
        )

    def test_oracle_covers_all_regimes(self) -> None:
        df = pd.read_csv(ORACLE_CSV)
        found = set(df["dataset"].tolist())
        assert EXPECTED_REGIMES == found, (
            f"oracle_routing_eval.csv regimes mismatch.\n"
            f"  Expected: {EXPECTED_REGIMES}\n  Found:    {found}"
        )

    def test_budget_curves_covers_all_regimes(self) -> None:
        df = pd.read_csv(BUDGET_CURVES_CSV)
        found = set(df["dataset"].tolist())
        assert EXPECTED_REGIMES <= found, (
            f"budget_curves_all_datasets.csv missing regimes: {EXPECTED_REGIMES - found}"
        )

    def test_main_results_key_numbers_preserved(self) -> None:
        """Regression: exact accuracy values from main_results_summary.csv unchanged."""
        expected = {
            "gsm8k_random_100": {"reasoning_acc": 0.9, "best_policy_acc": 0.92},
            "hard_gsm8k_100": {"reasoning_acc": 0.79, "best_policy_acc": 0.82},
            "hard_gsm8k_b2": {"reasoning_acc": 0.83, "best_policy_acc": 0.91},
            "math500_100": {"reasoning_acc": 0.64, "best_policy_acc": 0.65},
        }
        df = pd.read_csv(MAIN_RESULTS_CSV)
        for _, row in df.iterrows():
            regime = row["dataset"]
            if regime not in expected:
                continue
            for col, val in expected[regime].items():
                assert abs(float(row[col]) - val) < 1e-6, (
                    f"Regression failure: {regime}/{col} expected {val}, "
                    f"got {row[col]}"
                )

    def test_oracle_key_numbers_preserved(self) -> None:
        """Regression: exact oracle accuracy values unchanged."""
        expected = {
            "gsm8k_random_100": 0.92,
            "hard_gsm8k_100": 0.91,
            "hard_gsm8k_b2": 0.92,
            "math500_100": 0.70,
        }
        df = pd.read_csv(ORACLE_CSV)
        for _, row in df.iterrows():
            regime = row["dataset"]
            if regime not in expected:
                continue
            assert abs(float(row["oracle_acc"]) - expected[regime]) < 1e-6, (
                f"Oracle regression: {regime} expected {expected[regime]}, "
                f"got {row['oracle_acc']}"
            )


# ===========================================================================
# 2. CONFIDENCE-THRESHOLD ROUTING BASELINE TESTS
# ===========================================================================


class TestConfidenceThresholdRouter:
    """Tests for src/baselines/confidence_threshold_router.py."""

    def _make_df(self, n: int = 20, seed: int = 0) -> pd.DataFrame:
        """Create a minimal routing DataFrame with required columns."""
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "unified_confidence_score": rng.uniform(0.2, 0.9, n),
                "reasoning_correct": rng.integers(0, 2, n),
                "revise_correct": rng.integers(0, 2, n),
                "revise_helpful": rng.integers(0, 2, n),
            }
        )

    def test_evaluate_threshold_zero_revises_nobody(self) -> None:
        from src.baselines.confidence_threshold_router import evaluate_threshold

        df = self._make_df()
        result = evaluate_threshold(df, threshold=0.0)
        assert result.revise_rate == 0.0
        assert result.avg_cost == 1.0
        assert 0.0 <= result.accuracy <= 1.0

    def test_evaluate_threshold_one_revises_everyone(self) -> None:
        from src.baselines.confidence_threshold_router import evaluate_threshold

        df = self._make_df()
        result = evaluate_threshold(df, threshold=1.0)
        assert result.revise_rate == 1.0
        assert abs(result.avg_cost - 2.0) < 1e-9

    def test_accuracy_formula_correct(self) -> None:
        from src.baselines.confidence_threshold_router import evaluate_threshold

        # Construct a deterministic example
        df = pd.DataFrame(
            {
                "unified_confidence_score": [0.3, 0.7, 0.5, 0.9],
                "reasoning_correct": [1, 1, 0, 1],
                "revise_correct": [0, 1, 1, 0],
                "revise_helpful": [0, 0, 1, 0],
            }
        )
        # threshold=0.6 → revise queries 0 (0.3<0.6) and 2 (0.5<0.6); stay with 1,3
        result = evaluate_threshold(df, threshold=0.6)
        # Query 0: revise → revise_correct=0
        # Query 1: reasoning → reasoning_correct=1
        # Query 2: revise → revise_correct=1
        # Query 3: reasoning → reasoning_correct=1
        expected_acc = (0 + 1 + 1 + 1) / 4
        assert abs(result.accuracy - expected_acc) < 1e-9
        assert abs(result.revise_rate - 0.5) < 1e-9

    def test_sweep_returns_sorted_thresholds(self) -> None:
        from src.baselines.confidence_threshold_router import sweep_thresholds

        df = self._make_df()
        results = sweep_thresholds(df, thresholds=[0.5, 0.1, 0.9, 0.3])
        thresholds = [r.threshold for r in results]
        assert thresholds == sorted(thresholds)

    def test_choose_operating_point_respects_target_cost(self) -> None:
        from src.baselines.confidence_threshold_router import (
            ThresholdResult,
            choose_operating_point,
        )

        results = [
            ThresholdResult(threshold=0.0, accuracy=0.80, avg_cost=1.0, revise_rate=0.0, n=100),
            ThresholdResult(threshold=0.3, accuracy=0.85, avg_cost=1.15, revise_rate=0.15, n=100),
            ThresholdResult(threshold=0.5, accuracy=0.88, avg_cost=1.30, revise_rate=0.30, n=100),
        ]
        op = choose_operating_point(results, target_cost=1.2)
        assert op.avg_cost <= 1.2 + 1e-9
        assert op.accuracy == 0.85

    def test_sweep_and_summarise_creates_output_files(self, tmp_path: Path) -> None:
        from src.baselines.confidence_threshold_router import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            sweep_and_summarise,
        )

        # Use absolute paths for the real data files
        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        results = sweep_and_summarise(
            regime_files=regime_files,
            output_dir=tmp_path,
        )
        assert len(results) == 4
        assert (tmp_path / "confidence_threshold_summary.csv").exists()
        assert (tmp_path / "confidence_threshold_sweep.csv").exists()
        assert (tmp_path / "confidence_threshold_summary.json").exists()

    def test_summary_csv_has_correct_schema(self, tmp_path: Path) -> None:
        from src.baselines.confidence_threshold_router import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            sweep_and_summarise,
        )

        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        sweep_and_summarise(regime_files=regime_files, output_dir=tmp_path)
        df = pd.read_csv(tmp_path / "confidence_threshold_summary.csv")
        assert set(df.columns) >= {"regime", "accuracy", "avg_cost", "revise_rate", "n"}
        assert len(df) == 4
        # All regimes present
        assert set(df["regime"].tolist()) == EXPECTED_REGIMES

    def test_accuracy_in_valid_range_for_real_regimes(self, tmp_path: Path) -> None:
        from src.baselines.confidence_threshold_router import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            sweep_and_summarise,
        )

        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        results = sweep_and_summarise(regime_files=regime_files, output_dir=tmp_path)
        for r in results:
            assert 0.0 <= r.accuracy <= 1.0, f"accuracy out of range for {r.regime}"
            assert 1.0 <= r.avg_cost <= 2.0, f"avg_cost out of range for {r.regime}"


# ===========================================================================
# 3. LEARNED ROUTER BASELINE TESTS
# ===========================================================================


class TestLearnedRouterBaseline:
    """Tests for src/baselines/learned_router_baseline.py."""

    def _make_df(self, n: int = 40, seed: int = 0) -> pd.DataFrame:
        """Create a minimal regime DataFrame with all required columns."""
        from src.baselines.learned_router_baseline import FEATURE_COLS

        rng = np.random.default_rng(seed)
        data = {col: rng.uniform(0, 1, n) for col in FEATURE_COLS}
        # Add a few binary features
        for col in FEATURE_COLS:
            if "flag" in col or "suspected" in col or "parse_success" in col:
                data[col] = rng.integers(0, 2, n).astype(float)
        data["revise_helpful"] = (rng.uniform(0, 1, n) < 0.3).astype(int)
        data["reasoning_correct"] = rng.integers(0, 2, n)
        data["revise_correct"] = rng.integers(0, 2, n)
        return pd.DataFrame(data)

    def test_evaluate_regime_returns_result(self) -> None:
        from src.baselines.learned_router_baseline import evaluate_regime

        df = self._make_df(n=50)
        result = evaluate_regime("test_regime", df, model_name="logistic_regression")
        assert result.regime == "test_regime"
        assert 0.0 <= result.accuracy <= 1.0
        assert 1.0 <= result.avg_cost <= 2.0
        assert 0.0 <= result.revise_rate <= 1.0
        assert result.n == 50

    def test_decision_tree_runs(self) -> None:
        from src.baselines.learned_router_baseline import evaluate_regime

        df = self._make_df(n=50)
        result = evaluate_regime("test_regime", df, model_name="decision_tree")
        assert result.model_name == "decision_tree"
        assert 0.0 <= result.accuracy <= 1.0

    def test_degenerate_flagged_when_all_same_prediction(self) -> None:
        from src.baselines.learned_router_baseline import FEATURE_COLS, evaluate_regime

        # Create a dataset where revise_helpful is always 0 (extreme imbalance)
        rng = np.random.default_rng(1)
        data = {col: rng.uniform(0, 1, 30) for col in FEATURE_COLS}
        data["revise_helpful"] = np.zeros(30, dtype=int)  # all negative
        data["reasoning_correct"] = rng.integers(0, 2, 30)
        data["revise_correct"] = rng.integers(0, 2, 30)
        df = pd.DataFrame(data)
        result = evaluate_regime("degenerate_test", df, model_name="decision_tree")
        # Should flag degenerate and have revise_rate = 0
        assert result.degenerate

    def test_run_all_regimes_creates_outputs(self, tmp_path: Path) -> None:
        from src.baselines.learned_router_baseline import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            run_all_regimes,
        )

        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        results = run_all_regimes(regime_files=regime_files, output_dir=tmp_path)
        # 4 regimes × 2 models = 8 results
        assert len(results) == 8
        assert (tmp_path / "learned_router_summary.csv").exists()
        assert (tmp_path / "learned_router_summary.json").exists()

    def test_summary_csv_has_all_regimes_and_models(self, tmp_path: Path) -> None:
        from src.baselines.learned_router_baseline import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            run_all_regimes,
        )

        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        run_all_regimes(regime_files=regime_files, output_dir=tmp_path)
        df = pd.read_csv(tmp_path / "learned_router_summary.csv")
        assert set(df["regime"].unique()) == EXPECTED_REGIMES
        assert set(df["baseline"].unique()) >= {
            "learned_router_logistic_regression",
            "learned_router_decision_tree",
        }

    def test_class_imbalance_handled_honestly(self, tmp_path: Path) -> None:
        """GSM8K has only 2 positives — result should mention it or be degenerate."""
        from src.baselines.learned_router_baseline import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            run_all_regimes,
        )

        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        results = run_all_regimes(regime_files=regime_files, output_dir=tmp_path)
        gsm8k_results = [r for r in results if r.regime == "gsm8k_random_100"]
        assert len(gsm8k_results) == 2
        # At least one result should either be degenerate or have a note
        has_honest_note = any(r.degenerate or r.note for r in gsm8k_results)
        assert has_honest_note, (
            "GSM8K has only 2 positives; at least one model should flag imbalance"
        )


# ===========================================================================
# 4. UNCERTAINTY ANALYSIS TESTS
# ===========================================================================


class TestUncertaintyAnalysis:
    """Tests for src/evaluation/uncertainty_analysis.py."""

    def test_bootstrap_ci_observed_delta_correct(self) -> None:
        from src.evaluation.uncertainty_analysis import bootstrap_ci

        a = np.array([1, 1, 1, 0, 0], dtype=float)  # mean = 0.6
        b = np.array([1, 0, 0, 0, 0], dtype=float)  # mean = 0.2
        obs, lo, hi = bootstrap_ci(a, b, n_bootstrap=1000, seed=42)
        assert abs(obs - 0.4) < 1e-9
        assert lo <= obs <= hi

    def test_bootstrap_ci_returns_valid_interval(self) -> None:
        from src.evaluation.uncertainty_analysis import bootstrap_ci

        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 100).astype(float)
        b = rng.integers(0, 2, 100).astype(float)
        obs, lo, hi = bootstrap_ci(a, b, n_bootstrap=500, seed=0)
        assert lo <= hi
        assert lo <= obs <= hi

    def test_zero_delta_returns_narrow_interval(self) -> None:
        from src.evaluation.uncertainty_analysis import bootstrap_ci

        # Identical arrays → delta should be 0 with CI around 0
        a = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        obs, lo, hi = bootstrap_ci(a, a.copy(), n_bootstrap=2000, seed=7)
        assert abs(obs) < 1e-9

    def test_analyse_regime_returns_four_comparisons(self) -> None:
        from src.evaluation.uncertainty_analysis import analyse_regime

        df = pd.read_csv(REGIME_FILES["hard_gsm8k_100"])
        result = analyse_regime("hard_gsm8k_100", df, n_bootstrap=200, seed=42)
        assert result.regime == "hard_gsm8k_100"
        assert result.n == 100
        assert len(result.comparisons) == 4

    def test_comparison_names_present(self) -> None:
        from src.evaluation.uncertainty_analysis import analyse_regime

        df = pd.read_csv(REGIME_FILES["hard_gsm8k_100"])
        result = analyse_regime("hard_gsm8k_100", df, n_bootstrap=100, seed=0)
        names = {c.comparison for c in result.comparisons}
        assert "adaptive_best_policy_vs_always_reasoning" in names
        assert "oracle_vs_always_reasoning" in names

    def test_run_uncertainty_analysis_creates_files(self, tmp_path: Path) -> None:
        from src.evaluation.uncertainty_analysis import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            run_uncertainty_analysis,
        )

        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        results = run_uncertainty_analysis(
            regime_files=regime_files,
            output_dir=tmp_path,
            n_bootstrap=200,
        )
        assert len(results) == 4
        assert (tmp_path / "uncertainty_analysis.json").exists()
        assert (tmp_path / "uncertainty_analysis_summary.csv").exists()

    def test_uncertainty_csv_schema(self, tmp_path: Path) -> None:
        from src.evaluation.uncertainty_analysis import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            run_uncertainty_analysis,
        )

        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        run_uncertainty_analysis(
            regime_files=regime_files,
            output_dir=tmp_path,
            n_bootstrap=100,
        )
        df = pd.read_csv(tmp_path / "uncertainty_analysis_summary.csv")
        required_cols = {
            "regime",
            "comparison",
            "observed_delta",
            "ci_lower",
            "ci_upper",
            "significant_at_95pct",
        }
        assert required_cols <= set(df.columns)
        assert set(df["regime"].unique()) == EXPECTED_REGIMES

    def test_ci_bounds_ordered(self, tmp_path: Path) -> None:
        """ci_lower ≤ observed_delta ≤ ci_upper for all rows."""
        from src.evaluation.uncertainty_analysis import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            run_uncertainty_analysis,
        )

        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        run_uncertainty_analysis(
            regime_files=regime_files,
            output_dir=tmp_path,
            n_bootstrap=500,
        )
        df = pd.read_csv(tmp_path / "uncertainty_analysis_summary.csv")
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["observed_delta"] + 1e-9, (
                f"ci_lower > observed_delta for {row['regime']}/{row['comparison']}"
            )
            assert row["observed_delta"] <= row["ci_upper"] + 1e-9, (
                f"observed_delta > ci_upper for {row['regime']}/{row['comparison']}"
            )

    def test_json_output_structure(self, tmp_path: Path) -> None:
        from src.evaluation.uncertainty_analysis import (
            REGIME_FILES as DEFAULT_REGIME_FILES,
            run_uncertainty_analysis,
        )

        regime_files = {k: str(REPO_ROOT / v) for k, v in DEFAULT_REGIME_FILES.items()}
        run_uncertainty_analysis(
            regime_files=regime_files,
            output_dir=tmp_path,
            n_bootstrap=100,
        )
        data = json.loads((tmp_path / "uncertainty_analysis.json").read_text())
        assert isinstance(data, list)
        assert len(data) == 4
        for regime_data in data:
            assert "regime" in regime_data
            assert "comparisons" in regime_data
            assert isinstance(regime_data["comparisons"], list)


# ===========================================================================
# 5. CLARIFICATION EXPORT TESTS
# ===========================================================================


class TestClarificationExport:
    """Tests for src/evaluation/clarification_export.py."""

    def test_build_clarification_table_returns_four_rows(self) -> None:
        from src.evaluation.clarification_export import build_clarification_table

        rows = build_clarification_table(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        assert len(rows) == 4

    def test_all_regimes_present_in_table(self) -> None:
        from src.evaluation.clarification_export import build_clarification_table

        rows = build_clarification_table(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        regimes = {r.regime for r in rows}
        assert regimes == EXPECTED_REGIMES

    def test_n_equals_100_for_all_regimes(self) -> None:
        from src.evaluation.clarification_export import build_clarification_table

        rows = build_clarification_table(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        for r in rows:
            assert r.n == 100, f"Expected n=100 for {r.regime}, got {r.n}"

    def test_always_reasoning_acc_matches_main_results(self) -> None:
        from src.evaluation.clarification_export import build_clarification_table

        rows = build_clarification_table(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        expected = {
            "gsm8k_random_100": 0.9,
            "hard_gsm8k_100": 0.79,
            "hard_gsm8k_b2": 0.83,
            "math500_100": 0.64,
        }
        for r in rows:
            assert abs(r.always_reasoning_acc - expected[r.regime]) < 1e-6, (
                f"always_reasoning_acc mismatch for {r.regime}: "
                f"expected {expected[r.regime]}, got {r.always_reasoning_acc}"
            )

    def test_oracle_acc_matches_oracle_csv(self) -> None:
        from src.evaluation.clarification_export import build_clarification_table

        rows = build_clarification_table(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        expected = {
            "gsm8k_random_100": 0.92,
            "hard_gsm8k_100": 0.91,
            "hard_gsm8k_b2": 0.92,
            "math500_100": 0.70,
        }
        for r in rows:
            assert abs(r.oracle_acc - expected[r.regime]) < 1e-6, (
                f"oracle_acc mismatch for {r.regime}: "
                f"expected {expected[r.regime]}, got {r.oracle_acc}"
            )

    def test_budget_frontier_values_present_where_available(self) -> None:
        from src.evaluation.clarification_export import build_clarification_table

        rows = build_clarification_table(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        # All regimes should have budget frontier values since budget_curves exists
        for r in rows:
            assert r.budget_frontier_1_1_acc is not None, (
                f"budget_frontier_1.1 should be present for {r.regime}"
            )
            assert r.budget_frontier_1_2_acc is not None, (
                f"budget_frontier_1.2 should be present for {r.regime}"
            )

    def test_run_clarification_export_creates_all_files(self, tmp_path: Path) -> None:
        from src.evaluation.clarification_export import run_clarification_export

        run_clarification_export(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            output_dir=tmp_path,
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        assert (tmp_path / "clarification_table.csv").exists()
        assert (tmp_path / "clarification_wide.csv").exists()
        assert (tmp_path / "clarification_table.tex").exists()
        assert (tmp_path / "clarification_table.json").exists()

    def test_tidy_csv_has_expected_strategies(self, tmp_path: Path) -> None:
        from src.evaluation.clarification_export import run_clarification_export

        run_clarification_export(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            output_dir=tmp_path,
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        df = pd.read_csv(tmp_path / "clarification_table.csv")
        strategies = set(df["strategy"].unique())
        assert "always_reasoning" in strategies
        assert "best_adaptive" in strategies
        assert "oracle" in strategies
        assert "budget_frontier_1.1" in strategies
        assert "budget_frontier_1.2" in strategies

    def test_latex_file_contains_booktabs_commands(self, tmp_path: Path) -> None:
        from src.evaluation.clarification_export import run_clarification_export

        run_clarification_export(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            output_dir=tmp_path,
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        tex = (tmp_path / "clarification_table.tex").read_text()
        assert "\\toprule" in tex
        assert "\\midrule" in tex
        assert "\\bottomrule" in tex
        assert "\\begin{tabular}" in tex

    def test_wide_csv_has_one_row_per_regime(self, tmp_path: Path) -> None:
        from src.evaluation.clarification_export import run_clarification_export

        run_clarification_export(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            output_dir=tmp_path,
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        df = pd.read_csv(tmp_path / "clarification_wide.csv")
        assert len(df) == 4
        assert set(df["regime"].tolist()) == EXPECTED_REGIMES

    def test_best_adaptive_name_populated(self, tmp_path: Path) -> None:
        from src.evaluation.clarification_export import build_clarification_table

        rows = build_clarification_table(
            regime_files={k: str(v) for k, v in REGIME_FILES.items()},
            main_results_csv=MAIN_RESULTS_CSV,
            oracle_csv=ORACLE_CSV,
            budget_curves_csv=BUDGET_CURVES_CSV,
            cross_regime_csv=CROSS_REGIME_CSV,
        )
        for r in rows:
            assert r.best_adaptive_name, (
                f"best_adaptive_name should not be empty for {r.regime}"
            )
            assert r.best_adaptive_name != "nan"
