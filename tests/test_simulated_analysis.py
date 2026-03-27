import pytest

from src.datasets.synthetic_ttc import generate_synthetic_ttc_instance
from src.evaluation.simulated_analysis import (
    run_budget_sweep,
    run_noise_sensitivity_experiments,
    summarize_allocation_distribution,
    summarize_by_difficulty,
)


def test_budget_sweep_produces_multiple_budgets() -> None:
    instance = generate_synthetic_ttc_instance(
        n_queries=24,
        n_levels=5,
        curve_family="mixed_difficulty",
        costs=[0, 1, 2, 4, 7],
        seed=13,
    )

    sweep = run_budget_sweep(
        utility_table=instance["utility_table"],
        costs=instance["costs"],
        budgets=[20, 40, 60],
        allocator_names=["equal", "mckp"],
        difficulty_labels=instance["difficulty_labels"],
    )

    assert sweep["budgets"] == [20, 40, 60]
    assert len(sweep["runs"]) == 6
    assert {run["allocator"] for run in sweep["runs"]} == {"equal", "mckp"}
    assert {run["budget"] for run in sweep["runs"]} == {20, 40, 60}
    assert len(sweep["comparisons"]) == 3


def test_difficulty_summary_works_for_mixed_difficulty_instances() -> None:
    instance = generate_synthetic_ttc_instance(
        n_queries=40,
        n_levels=6,
        curve_family="mixed_difficulty",
        costs=[0, 1, 2, 3, 5, 8],
        seed=21,
    )
    selected_levels = [0] * 10 + [2] * 15 + [4] * 15

    summary = summarize_by_difficulty(
        difficulty_labels=instance["difficulty_labels"],
        selected_levels=selected_levels,
        costs=instance["costs"],
        utility_table=instance["utility_table"],
    )

    assert set(summary.keys()) == {"easy", "medium", "hard"}
    assert sum(group["num_queries"] for group in summary.values()) == 40
    for difficulty, group in summary.items():
        assert set(group["selected_level_counts"].keys()) == {"0", "1", "2", "3", "4", "5"}
        assert group["average_assigned_cost"] >= 0.0
        assert 0.0 <= group["average_achieved_utility"] <= 1.0
        assert group["num_queries"] > 0, difficulty


def test_allocation_distribution_summary_is_well_formed() -> None:
    summary = summarize_allocation_distribution(
        selected_levels=[0, 1, 1, 3, 3, 3],
        n_levels=4,
    )

    assert summary["selected_level_counts"] == {"0": 1, "1": 2, "2": 0, "3": 3}
    assert summary["selected_level_fractions"]["3"] == pytest.approx(0.5)
    assert summary["min_selected_level"] == 0
    assert summary["max_selected_level"] == 3
    assert summary["mean_selected_level"] == pytest.approx(11 / 6)


def test_noisy_mckp_experiment_runs_and_returns_valid_metrics() -> None:
    instance = generate_synthetic_ttc_instance(
        n_queries=30,
        n_levels=5,
        curve_family="mixed_difficulty",
        costs=[0, 1, 2, 4, 6],
        seed=99,
    )

    result = run_noise_sensitivity_experiments(
        utility_table=instance["utility_table"],
        costs=instance["costs"],
        budget=45,
        allocator_names=["equal", "mckp"],
        difficulty_labels=instance["difficulty_labels"],
        noise_levels=[
            "no_noise",
            {"name": "small_test_noise", "std": 0.03},
            {"name": "large_test_noise", "std": 0.12},
        ],
        seed=5,
    )

    assert len(result["runs"]) == 6
    assert len(result["comparisons"]) == 3

    for run in result["runs"]:
        assert run["allocator"] in {"equal", "mckp"}
        assert run["noise"]["noise_std"] >= 0.0
        assert run["total_cost"] <= 45
        assert 0.0 <= run["true_utility_achieved"] <= 30.0
        assert 0.0 <= run["estimated_utility_optimized"] <= 30.0
        assert "allocation_distribution" in run
        assert "difficulty_summary" in run
