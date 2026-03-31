"""Tests for the feature-method-fit offline analysis module.

These tests verify the core logic of feature derivation, outcome labelling,
and basic analysis functions without requiring the full routing datasets.
They use small synthetic fixtures to keep tests fast and offline.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_routing_row(**overrides: object) -> dict[str, str]:
    """Return a minimal routing-dataset row with sensible defaults."""
    defaults: dict[str, str] = {
        "question_id": "test_0",
        "question": "Alice has 3 apples. Bob has 5 apples. How many in total?",
        "gold_answer": "8",
        "reasoning_correct": "1",
        "revise_correct": "1",
        "revise_helpful": "0",
        # Question-side features
        "q_num_numeric_mentions": "2",
        "q_question_length_tokens_approx": "20",
        "tq_asks_remaining_or_left": "0",
        "tq_asks_total": "1",
        "tq_asks_difference": "0",
        "tq_asks_rate_or_unit": "0",
        "tq_asks_money": "0",
        "tq_asks_time": "0",
        "tq_has_subtraction_trap_verb": "0",
        "tq_has_addition_trap_structure": "1",
        "tq_has_multi_operation_hint": "0",
        "tq_potential_answer_echo_risk": "0",
        "q_has_percent_symbol": "0",
        "q_has_fraction_pattern": "0",
        # Output-side features
        "fp_first_pass_parse_success": "1",
        "v7_extra_answer_error": "0",
        "cons_target_quantity_mismatch_suspected": "0",
        "cons_answer_type_mismatch_suspected": "0",
        "cons_unit_mismatch_suspected": "0",
        "cons_impossible_sign_suspected": "0",
        "cons_integer_expected_but_noninteger_suspected": "0",
        "cons_constraint_word_conflict_suspected": "0",
        "cons_bound_violation_suspected": "0",
        "v6_final_answer_confident": "1",
        "v6_explanation_warning_score": "0",
        "v6_answer_error_score": "0",
        "v6_revise_recommended": "0",
        "v7_revise_recommended": "0",
    }
    defaults.update({str(k): str(v) for k, v in overrides.items()})
    return defaults


def _make_policy_row(**overrides: object) -> dict[str, str]:
    defaults: dict[str, str] = {
        "question_id": "test_0",
        "reasoning_correct": "1",
        "revise_correct": "1",
        "revise_helpful": "0",
        "policy_v5": "reasoning_greedy",
        "policy_v6": "reasoning_greedy",
        "policy_v7": "reasoning_greedy",
        "correct_if_v5": "1",
        "correct_if_v6": "1",
        "correct_if_v7": "1",
        "cost_v5": "1",
        "cost_v6": "1",
        "cost_v7": "1",
    }
    defaults.update({str(k): str(v) for k, v in overrides.items()})
    return defaults


# ---------------------------------------------------------------------------
# Tests: _derive_features
# ---------------------------------------------------------------------------

class TestDeriveFeatures:
    def test_basic_defaults(self) -> None:
        from src.analysis.feature_method_fit import _derive_features

        row = _make_routing_row()
        feats = _derive_features(row)

        assert feats["prompt_number_count"] == 2.0
        assert feats["prompt_token_length"] == 20.0
        assert feats["target_quantity_type"] == "total"
        assert feats["multi_stepness_proxy"] == 0
        assert feats["explicit_constraint_presence"] == 1  # addition_trap_structure=1
        assert feats["relational_wording_presence"] == 0
        assert feats["special_structure_presence"] == 0
        assert feats["final_answer_parseable"] == 1
        assert feats["body_final_numeric_mismatch"] == 0
        assert feats["target_quantity_mismatch"] == 0
        assert feats["constraint_violation_signal"] == 0
        assert feats["copied_question_number_as_final_answer"] == 0
        assert feats["cheap_route_confidence"] == 1
        assert feats["explanation_warning_signal"] == 0
        assert feats["answer_error_signal"] == 0

    def test_target_quantity_type_priority(self) -> None:
        from src.analysis.feature_method_fit import _derive_features

        # rate_or_unit should win over total
        row = _make_routing_row(tq_asks_rate_or_unit="1", tq_asks_total="1")
        assert _derive_features(row)["target_quantity_type"] == "rate_or_unit"

        # remaining_or_left should win over total
        row = _make_routing_row(tq_asks_remaining_or_left="1", tq_asks_total="1")
        assert _derive_features(row)["target_quantity_type"] == "remaining_or_left"

        # defaults to other
        row = _make_routing_row(
            tq_asks_remaining_or_left="0",
            tq_asks_total="0",
            tq_asks_difference="0",
            tq_asks_rate_or_unit="0",
            tq_asks_money="0",
            tq_asks_time="0",
        )
        assert _derive_features(row)["target_quantity_type"] == "other"

    def test_answer_error_signal_positive(self) -> None:
        from src.analysis.feature_method_fit import _derive_features

        row = _make_routing_row(v6_answer_error_score="3")
        assert _derive_features(row)["answer_error_signal"] == 1

    def test_explanation_warning_signal_positive(self) -> None:
        from src.analysis.feature_method_fit import _derive_features

        row = _make_routing_row(v6_explanation_warning_score="2")
        assert _derive_features(row)["explanation_warning_signal"] == 1

    def test_constraint_violation_signal_any_flag(self) -> None:
        from src.analysis.feature_method_fit import _derive_features

        row = _make_routing_row(cons_unit_mismatch_suspected="1")
        assert _derive_features(row)["constraint_violation_signal"] == 1

    def test_body_final_numeric_mismatch_v7_extra_error(self) -> None:
        from src.analysis.feature_method_fit import _derive_features

        row = _make_routing_row(v7_extra_answer_error="3")
        assert _derive_features(row)["body_final_numeric_mismatch"] == 1

    def test_special_structure_presence_fraction(self) -> None:
        from src.analysis.feature_method_fit import _derive_features

        row = _make_routing_row(q_has_fraction_pattern="1")
        assert _derive_features(row)["special_structure_presence"] == 1

    def test_relational_wording_difference(self) -> None:
        from src.analysis.feature_method_fit import _derive_features

        row = _make_routing_row(tq_asks_difference="1")
        assert _derive_features(row)["relational_wording_presence"] == 1


# ---------------------------------------------------------------------------
# Tests: _derive_outcomes
# ---------------------------------------------------------------------------

class TestDeriveOutcomes:
    def test_safe_cheap(self) -> None:
        from src.analysis.feature_method_fit import _derive_outcomes

        rr = _make_routing_row(reasoning_correct="1", revise_correct="1")
        out = _derive_outcomes(rr, None)
        assert out["safe_cheap"] == 1
        assert out["revise_helpful"] == 0
        assert out["both_wrong"] == 0

    def test_revise_helpful(self) -> None:
        from src.analysis.feature_method_fit import _derive_outcomes

        rr = _make_routing_row(reasoning_correct="0", revise_correct="1")
        out = _derive_outcomes(rr, None)
        assert out["revise_helpful"] == 1
        assert out["safe_cheap"] == 0
        assert out["both_wrong"] == 0

    def test_both_wrong(self) -> None:
        from src.analysis.feature_method_fit import _derive_outcomes

        rr = _make_routing_row(reasoning_correct="0", revise_correct="0")
        out = _derive_outcomes(rr, None)
        assert out["both_wrong"] == 1
        assert out["revise_helpful"] == 0
        assert out["safe_cheap"] == 0

    def test_unnecessary_revise_candidate(self) -> None:
        from src.analysis.feature_method_fit import _derive_outcomes

        rr = _make_routing_row(
            reasoning_correct="1", v6_revise_recommended="1", v7_revise_recommended="0"
        )
        out = _derive_outcomes(rr, None)
        assert out["unnecessary_revise_candidate"] == 1

    def test_method_best_rg_cheapest(self) -> None:
        from src.analysis.feature_method_fit import _derive_outcomes

        rr = _make_routing_row(reasoning_correct="1", revise_correct="1")
        pr = _make_policy_row(
            correct_if_v5="1", cost_v5="1",
            correct_if_v6="1", cost_v6="1",
            correct_if_v7="1", cost_v7="1",
        )
        out = _derive_outcomes(rr, pr)
        # All methods correct, RG has cost=1 (same as v5/v6/v7), picked first
        valid = ("reasoning_greedy", "adaptive_v5", "adaptive_v6", "adaptive_v7")
        assert out["method_best_label"] in valid

    def test_method_best_dpr_only_correct(self) -> None:
        from src.analysis.feature_method_fit import _derive_outcomes

        rr = _make_routing_row(reasoning_correct="0", revise_correct="1")
        pr = _make_policy_row(
            correct_if_v5="0", cost_v5="1",
            correct_if_v6="0", cost_v6="1",
            correct_if_v7="0", cost_v7="1",
        )
        out = _derive_outcomes(rr, pr)
        assert out["method_best_label"] == "direct_plus_revise"


# ---------------------------------------------------------------------------
# Tests: univariate summary
# ---------------------------------------------------------------------------

class TestUnivariateSummary:
    def _make_rows(self) -> list[dict]:
        """Small synthetic dataset for testing."""
        from src.analysis.feature_method_fit import _derive_features, _derive_outcomes

        configs = [
            # revise_helpful: RG wrong, DPR correct, answer_error high
            {"reasoning_correct": "0", "revise_correct": "1",
             "v6_answer_error_score": "3", "v6_final_answer_confident": "0"},
            # safe_cheap: RG correct, answer_error low
            {"reasoning_correct": "1", "revise_correct": "1",
             "v6_answer_error_score": "0", "v6_final_answer_confident": "1"},
            {"reasoning_correct": "1", "revise_correct": "1",
             "v6_answer_error_score": "0", "v6_final_answer_confident": "1"},
            # both_wrong: all wrong
            {"reasoning_correct": "0", "revise_correct": "0",
             "v6_answer_error_score": "2", "v6_final_answer_confident": "0"},
        ]
        rows = []
        for cfg in configs:
            rr = _make_routing_row(**cfg)
            feats = _derive_features(rr)
            outcomes = _derive_outcomes(rr, None)
            rows.append({"regime": "test", "question_id": "x", **outcomes, **feats})
        return rows

    def test_returns_list_of_dicts(self) -> None:
        from src.analysis.feature_method_fit import compute_univariate_summary

        rows = self._make_rows()
        result = compute_univariate_summary(rows)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_answer_error_in_result(self) -> None:
        from src.analysis.feature_method_fit import compute_univariate_summary

        rows = self._make_rows()
        result = compute_univariate_summary(rows)
        ae = next(r for r in result if r["feature"] == "answer_error_signal")
        # revise_helpful group has 1 row with answer_error=1 → mean=1.0
        assert ae["mean_revise_helpful"] == 1.0
        # safe_cheap group has 2 rows with answer_error=0 → mean=0.0
        assert ae["mean_safe_cheap"] == 0.0

    def test_effect_size_proxy_positive(self) -> None:
        from src.analysis.feature_method_fit import compute_univariate_summary

        rows = self._make_rows()
        result = compute_univariate_summary(rows)
        ae = next(r for r in result if r["feature"] == "answer_error_signal")
        assert ae["effect_size_proxy"] >= 0


# ---------------------------------------------------------------------------
# Tests: method fit summary
# ---------------------------------------------------------------------------

class TestMethodFitSummary:
    def test_basic(self) -> None:
        from src.analysis.feature_method_fit import (
            _derive_features,
            _derive_outcomes,
            compute_method_fit_summary,
        )

        rows = []
        for i in range(4):
            rr = _make_routing_row(
                reasoning_correct="1" if i < 3 else "0",
                revise_correct="1",
            )
            pr = _make_policy_row(
                correct_if_v5="1", cost_v5="1",
                correct_if_v6="1", cost_v6="1",
                correct_if_v7="1", cost_v7="1",
            )
            feats = _derive_features(rr)
            outcomes = _derive_outcomes(rr, pr)
            rows.append({"regime": "test", "question_id": str(i), **outcomes, **feats})

        result = compute_method_fit_summary(rows)
        assert isinstance(result, list)
        for entry in result:
            assert "method_best_label" in entry
            assert "n" in entry


# ---------------------------------------------------------------------------
# Tests: build_analysis_dataset (integration with real files)
# ---------------------------------------------------------------------------

class TestBuildAnalysisDataset:
    def test_returns_non_empty_with_real_data(self, tmp_path: Path) -> None:
        """Smoke test: at least one regime loads successfully."""
        from src.analysis.feature_method_fit import build_analysis_dataset

        # Use the real repo root
        repo_root = Path(__file__).resolve().parent.parent
        rows = build_analysis_dataset(repo_root)
        assert len(rows) > 0

    def test_all_features_present(self) -> None:
        from src.analysis.feature_method_fit import ALL_FEATURES, build_analysis_dataset

        repo_root = Path(__file__).resolve().parent.parent
        rows = build_analysis_dataset(repo_root)
        assert rows  # at least 1 row
        sample = rows[0]
        for feat in ALL_FEATURES:
            assert feat in sample, f"Feature '{feat}' missing from analysis dataset row"

    def test_regime_column_values(self) -> None:
        from src.analysis.feature_method_fit import REGIMES, build_analysis_dataset

        repo_root = Path(__file__).resolve().parent.parent
        rows = build_analysis_dataset(repo_root)
        found_regimes = {r["regime"] for r in rows}
        expected = {cfg["regime"] for cfg in REGIMES}
        assert found_regimes == expected

    def test_outcome_columns_present(self) -> None:
        from src.analysis.feature_method_fit import OUTCOME_LABELS, build_analysis_dataset

        repo_root = Path(__file__).resolve().parent.parent
        rows = build_analysis_dataset(repo_root)
        sample = rows[0]
        for label in OUTCOME_LABELS:
            assert label in sample, f"Outcome label '{label}' missing"

    def test_outcome_values_binary(self) -> None:
        from src.analysis.feature_method_fit import build_analysis_dataset

        repo_root = Path(__file__).resolve().parent.parent
        rows = build_analysis_dataset(repo_root)
        for label in ("revise_helpful", "safe_cheap", "both_wrong"):
            for r in rows:
                assert r[label] in (0, 1), (
                    f"Outcome '{label}' has non-binary value {r[label]!r}"
                )
