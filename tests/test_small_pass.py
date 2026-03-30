"""Tests for the small experiment pass: AIME-2024 evaluation and confidence baseline.

All tests are fully offline — they use committed data files and tmp_path fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AIME_CSV = REPO_ROOT / "data/real_aime2024_routing_dataset.csv"


# ===========================================================================
# 1. AIME evaluation tests
# ===========================================================================


class TestSmallPassAimeEval:
    def test_aime_csv_exists(self) -> None:
        assert AIME_CSV.exists(), f"AIME CSV missing: {AIME_CSV}"

    def test_aime_csv_has_required_columns(self) -> None:
        df = pd.read_csv(AIME_CSV)
        required = {
            "question",
            "reasoning_raw",
            "reasoning_correct",
            "revise_correct",
            "revise_helpful",
            "unified_confidence_score",
        }
        missing = required - set(df.columns)
        assert not missing, f"AIME CSV missing columns: {missing}"

    def test_aime_csv_has_30_rows(self) -> None:
        df = pd.read_csv(AIME_CSV)
        assert len(df) == 30

    def test_run_small_pass_aime_eval_completes(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_aime_eval import run_small_pass_aime_eval

        result = run_small_pass_aime_eval(
            dataset_csv=AIME_CSV,
            output_dir=tmp_path,
        )
        assert result["summary"]["run_status"] == "COMPLETED"
        assert result["summary"]["num_rows"] == 30

    def test_aime_eval_produces_output_files(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_aime_eval import run_small_pass_aime_eval

        run_small_pass_aime_eval(dataset_csv=AIME_CSV, output_dir=tmp_path)
        assert (tmp_path / "aime_summary.json").exists()
        assert (tmp_path / "aime_policy_comparison.csv").exists()
        assert (tmp_path / "aime_per_query_decisions.csv").exists()
        assert (tmp_path / "aime_confidence_sweep.csv").exists()

    def test_aime_comparison_has_all_routes(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_aime_eval import run_small_pass_aime_eval

        result = run_small_pass_aime_eval(dataset_csv=AIME_CSV, output_dir=tmp_path)
        routes = {r["route"] for r in result["summary"]["comparison"]}
        expected = {
            "reasoning_greedy",
            "direct_plus_revise",
            "adaptive_policy_v5",
            "adaptive_policy_v6",
            "adaptive_policy_v7",
            "confidence_threshold",
            "oracle",
        }
        assert expected == routes

    def test_aime_revise_helpful_is_zero(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_aime_eval import run_small_pass_aime_eval

        result = run_small_pass_aime_eval(dataset_csv=AIME_CSV, output_dir=tmp_path)
        assert result["summary"]["revise_helpful_prevalence"] == 0.0

    def test_aime_oracle_matches_cheap_baseline(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_aime_eval import run_small_pass_aime_eval

        result = run_small_pass_aime_eval(dataset_csv=AIME_CSV, output_dir=tmp_path)
        comp = {r["route"]: r for r in result["summary"]["comparison"]}
        assert abs(comp["oracle"]["accuracy"] - comp["reasoning_greedy"]["accuracy"]) < 1e-9

    def test_aime_eval_blocked_on_missing_file(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_aime_eval import run_small_pass_aime_eval

        result = run_small_pass_aime_eval(
            dataset_csv=tmp_path / "nonexistent.csv",
            output_dir=tmp_path,
        )
        assert result["summary"]["run_status"] == "BLOCKED"

    def test_aime_accuracy_values_in_range(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_aime_eval import run_small_pass_aime_eval

        result = run_small_pass_aime_eval(dataset_csv=AIME_CSV, output_dir=tmp_path)
        for row in result["summary"]["comparison"]:
            assert 0.0 <= row["accuracy"] <= 1.0
            assert row["avg_cost"] >= 1.0


# ===========================================================================
# 2. Confidence baseline small-pass tests
# ===========================================================================


class TestConfidenceBaselineSmallPass:
    def test_confidence_baseline_on_aime(self, tmp_path: Path) -> None:
        from src.baselines.confidence_threshold_router import (
            evaluate_threshold,
            sweep_thresholds,
            choose_operating_point,
        )

        df = pd.read_csv(AIME_CSV)
        sweep = sweep_thresholds(df)
        op = choose_operating_point(sweep, target_cost=1.2)
        assert 0.0 <= op.accuracy <= 1.0
        assert op.avg_cost <= 1.2 + 1e-9
        assert op.n == 30

    def test_aime_confidence_threshold_zero_is_optimal(self) -> None:
        """When revise_helpful=0 for all rows, threshold=0 is the best point."""
        from src.baselines.confidence_threshold_router import (
            sweep_thresholds,
            choose_operating_point,
        )

        df = pd.read_csv(AIME_CSV)
        sweep = sweep_thresholds(df)
        op = choose_operating_point(sweep, target_cost=1.2)
        # At threshold=0, nobody is revised (confidence is always ≥ 0)
        # so revise_rate=0 and avg_cost=1.0, which is ≤ 1.2
        # The best feasible accuracy should be reasoning_greedy accuracy
        reasoning_acc = float(df["reasoning_correct"].mean())
        assert abs(op.accuracy - reasoning_acc) < 1e-9


# ===========================================================================
# 3. Combined small-pass run tests
# ===========================================================================


class TestSmallPassCombinedEval:
    def test_run_small_pass_completes(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_combined_eval import run_small_pass

        result = run_small_pass(
            output_dir=tmp_path / "small_pass",
            tables_dir=tmp_path / "tables",
        )
        assert result["run_status"]["run_status"] == "COMPLETED"
        assert result["run_status"]["aime_status"] == "COMPLETED"
        assert result["run_status"]["gpqa_status"] == "BLOCKED"

    def test_combined_eval_produces_tables(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_combined_eval import run_small_pass

        tables = tmp_path / "tables"
        run_small_pass(output_dir=tmp_path / "small_pass", tables_dir=tables)
        assert (tables / "aime_policy_comparison.csv").exists()
        assert (tables / "confidence_baseline_main_regimes.csv").exists()

    def test_gpqa_always_blocked(self, tmp_path: Path) -> None:
        from src.evaluation.small_pass_combined_eval import run_small_pass

        result = run_small_pass(
            output_dir=tmp_path / "small_pass",
            tables_dir=tmp_path / "tables",
        )
        assert result["run_status"]["gpqa_status"] == "BLOCKED"
        assert "gpqa_blocker" in result["run_status"]
