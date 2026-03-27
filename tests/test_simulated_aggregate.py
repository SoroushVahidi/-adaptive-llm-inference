import pytest

from src.evaluation.simulated_aggregate import (
    aggregate_budget_rows,
    aggregate_multi_seed_results,
    aggregate_noise_rows,
)


def test_aggregation_runs_on_multiple_seeds() -> None:
    per_seed_results = [
        {
            "seed": 1,
            "budget_comparisons": [
                {
                    "budget": 100,
                    "equal_total_expected_utility": 40.0,
                    "mckp_total_expected_utility": 42.0,
                    "utility_gap_mckp_minus_equal": 2.0,
                    "relative_improvement_vs_equal": 0.05,
                },
                {
                    "budget": 150,
                    "equal_total_expected_utility": 44.0,
                    "mckp_total_expected_utility": 45.0,
                    "utility_gap_mckp_minus_equal": 1.0,
                    "relative_improvement_vs_equal": 1.0 / 44.0,
                },
            ],
            "noise_comparisons": [
                {
                    "noise_name": "no_noise",
                    "noise_std": 0.0,
                    "equal_true_utility_achieved": 45.0,
                    "mckp_true_utility_achieved": 46.0,
                    "utility_gap_mckp_minus_equal": 1.0,
                    "relative_improvement_vs_equal": 1.0 / 45.0,
                },
                {
                    "noise_name": "small_gaussian",
                    "noise_std": 0.02,
                    "equal_true_utility_achieved": 44.5,
                    "mckp_true_utility_achieved": 44.4,
                    "utility_gap_mckp_minus_equal": -0.1,
                    "relative_improvement_vs_equal": -0.1 / 44.5,
                },
            ],
        },
        {
            "seed": 2,
            "budget_comparisons": [
                {
                    "budget": 100,
                    "equal_total_expected_utility": 39.5,
                    "mckp_total_expected_utility": 41.0,
                    "utility_gap_mckp_minus_equal": 1.5,
                    "relative_improvement_vs_equal": 1.5 / 39.5,
                },
                {
                    "budget": 150,
                    "equal_total_expected_utility": 43.0,
                    "mckp_total_expected_utility": 43.1,
                    "utility_gap_mckp_minus_equal": 0.1,
                    "relative_improvement_vs_equal": 0.1 / 43.0,
                },
            ],
            "noise_comparisons": [
                {
                    "noise_name": "no_noise",
                    "noise_std": 0.0,
                    "equal_true_utility_achieved": 44.0,
                    "mckp_true_utility_achieved": 45.2,
                    "utility_gap_mckp_minus_equal": 1.2,
                    "relative_improvement_vs_equal": 1.2 / 44.0,
                },
                {
                    "noise_name": "small_gaussian",
                    "noise_std": 0.02,
                    "equal_true_utility_achieved": 43.8,
                    "mckp_true_utility_achieved": 43.7,
                    "utility_gap_mckp_minus_equal": -0.1,
                    "relative_improvement_vs_equal": -0.1 / 43.8,
                },
            ],
        },
    ]

    aggregated = aggregate_multi_seed_results(
        per_seed_results,
        output_dir="/tmp/simulated_aggregate_test",
        small_gap_threshold=0.5,
    )

    assert len(aggregated["aggregated_budget_summary"]) == 2
    assert len(aggregated["aggregated_noise_summary"]) == 2
    assert aggregated["key_findings"]["budget_where_mean_gap_peaks"] == 100
    disappearing = aggregated["key_findings"][
        "first_noise_level_where_mean_mckp_advantage_disappears"
    ]
    assert disappearing is not None
    assert disappearing["noise_name"] == "small_gaussian"


def test_aggregated_outputs_are_well_formed() -> None:
    budget_summary = aggregate_budget_rows(
        [
            {
                "seed": 11,
                "budget": 120,
                "equal_total_expected_utility": 50.0,
                "mckp_total_expected_utility": 51.0,
                "utility_gap_mckp_minus_equal": 1.0,
                "relative_improvement_vs_equal": 0.02,
            },
            {
                "seed": 12,
                "budget": 120,
                "equal_total_expected_utility": 49.0,
                "mckp_total_expected_utility": 50.5,
                "utility_gap_mckp_minus_equal": 1.5,
                "relative_improvement_vs_equal": 1.5 / 49.0,
            },
        ]
    )

    row = budget_summary[0]
    expected_keys = {
        "budget",
        "n_seeds",
        "mean_equal_utility",
        "std_equal_utility",
        "mean_mckp_utility",
        "std_mckp_utility",
        "mean_utility_gap",
        "std_utility_gap",
        "fraction_mckp_beats_equal",
    }
    assert set(row.keys()) == expected_keys
    assert row["budget"] == 120
    assert row["n_seeds"] == 2


def test_beat_rate_calculations_are_valid_probabilities() -> None:
    noise_summary = aggregate_noise_rows(
        [
            {
                "seed": 1,
                "noise_name": "no_noise",
                "noise_std": 0.0,
                "equal_true_utility_achieved": 10.0,
                "mckp_true_utility_achieved": 11.0,
                "utility_gap_mckp_minus_equal": 1.0,
                "relative_improvement_vs_equal": 0.1,
            },
            {
                "seed": 2,
                "noise_name": "no_noise",
                "noise_std": 0.0,
                "equal_true_utility_achieved": 10.0,
                "mckp_true_utility_achieved": 9.5,
                "utility_gap_mckp_minus_equal": -0.5,
                "relative_improvement_vs_equal": -0.05,
            },
            {
                "seed": 1,
                "noise_name": "medium_gaussian",
                "noise_std": 0.05,
                "equal_true_utility_achieved": 9.8,
                "mckp_true_utility_achieved": 9.0,
                "utility_gap_mckp_minus_equal": -0.8,
                "relative_improvement_vs_equal": -0.8 / 9.8,
            },
            {
                "seed": 2,
                "noise_name": "medium_gaussian",
                "noise_std": 0.05,
                "equal_true_utility_achieved": 9.7,
                "mckp_true_utility_achieved": 9.6,
                "utility_gap_mckp_minus_equal": -0.1,
                "relative_improvement_vs_equal": -0.1 / 9.7,
            },
        ]
    )

    assert len(noise_summary) == 2
    for row in noise_summary:
        assert 0.0 <= row["fraction_mckp_beats_equal"] <= 1.0
        assert row["n_seeds"] == 2

    no_noise = next(row for row in noise_summary if row["noise_name"] == "no_noise")
    assert no_noise["fraction_mckp_beats_equal"] == pytest.approx(0.5)
