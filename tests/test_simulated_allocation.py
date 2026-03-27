import pytest

from src.datasets.synthetic_ttc import generate_synthetic_ttc_instance
from src.evaluation.simulated_evaluator import evaluate_simulated_allocation


def _is_non_decreasing(values: list[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_synthetic_instance_shape_and_metadata() -> None:
    instance = generate_synthetic_ttc_instance(
        n_queries=12,
        n_levels=5,
        curve_family="mixed_difficulty",
        costs=[0, 1, 2, 4, 6],
        seed=99,
    )

    assert len(instance["query_ids"]) == 12
    assert len(instance["utility_table"]) == 12
    assert len(instance["costs"]) == 5
    assert instance["metadata"]["curve_family"] == "mixed_difficulty"
    assert instance["metadata"]["n_queries"] == 12
    assert instance["metadata"]["n_levels"] == 5


@pytest.mark.parametrize("family", ["monotone", "concave", "mixed_difficulty"])
def test_monotonicity_in_supported_families(family: str) -> None:
    instance = generate_synthetic_ttc_instance(
        n_queries=25,
        n_levels=6,
        curve_family=family,
        seed=1234,
    )
    for curve in instance["utility_table"]:
        assert _is_non_decreasing(curve)


def test_evaluator_correctness_for_simple_case() -> None:
    utility_table = [
        [0.0, 1.0, 5.0],
        [0.0, 2.0, 2.5],
    ]
    costs = [0, 1, 3]
    budget = 4

    result = evaluate_simulated_allocation(
        utility_table=utility_table,
        costs=costs,
        budget=budget,
        allocator_name="mckp",
    )

    assert result["selected_levels"] == [2, 1]
    assert result["total_expected_utility"] == pytest.approx(7.0)
    assert result["total_cost"] == 4
    assert result["average_cost_per_query"] == pytest.approx(2.0)
    assert result["average_utility_per_query"] == pytest.approx(3.5)


def test_mckp_not_worse_than_equal_on_nontrivial_instance() -> None:
    instance = generate_synthetic_ttc_instance(
        n_queries=60,
        n_levels=6,
        curve_family="mixed_difficulty",
        costs=[0, 1, 2, 3, 5, 8],
        seed=2026,
    )
    budget = 140

    equal_result = evaluate_simulated_allocation(
        utility_table=instance["utility_table"],
        costs=instance["costs"],
        budget=budget,
        allocator_name="equal",
    )
    mckp_result = evaluate_simulated_allocation(
        utility_table=instance["utility_table"],
        costs=instance["costs"],
        budget=budget,
        allocator_name="mckp",
    )

    assert mckp_result["total_cost"] <= budget
    assert equal_result["total_cost"] <= budget
    assert (
        mckp_result["total_expected_utility"] + 1e-12
        >= equal_result["total_expected_utility"]
    )
