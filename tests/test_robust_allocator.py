import pytest

from src.allocators.robust_equalized import RobustEqualizedAllocator
from src.datasets.synthetic_ttc import generate_synthetic_ttc_instance


def test_robust_allocator_returns_valid_selected_levels() -> None:
    instance = generate_synthetic_ttc_instance(
        n_queries=20,
        n_levels=5,
        curve_family="mixed_difficulty",
        costs=[0, 1, 2, 4, 6],
        seed=17,
    )
    allocator = RobustEqualizedAllocator(shrinkage=0.4)

    result = allocator.allocate(
        profits=instance["utility_table"],
        costs=instance["costs"],
        budget=32,
    )

    assert len(result["selected_levels"]) == 20
    assert all(0 <= level < 5 for level in result["selected_levels"])
    assert result["total_cost"] <= 32
    assert result["total_profit"] >= 0.0


def test_robust_allocator_respects_budget_on_simple_case() -> None:
    profits = [
        [0.1, 0.4, 0.9],
        [0.2, 0.5, 0.8],
        [0.3, 0.6, 0.7],
    ]
    costs = [0, 1, 3]
    allocator = RobustEqualizedAllocator(shrinkage=0.5)

    result = allocator.allocate(profits=profits, costs=costs, budget=4)

    assert result["total_cost"] <= 4
    assert len(result["selected_levels"]) == 3


def test_robust_allocator_works_on_noisy_utility_tables() -> None:
    clean_table = [
        [0.10, 0.45, 0.88, 0.92],
        [0.18, 0.30, 0.47, 0.75],
        [0.12, 0.20, 0.60, 0.83],
    ]
    noisy_estimates = [
        [0.11, 0.44, 0.91, 0.95],
        [0.21, 0.28, 0.52, 0.73],
        [0.10, 0.23, 0.58, 0.80],
    ]
    costs = [0, 1, 2, 4]
    allocator = RobustEqualizedAllocator(shrinkage=0.6)

    result = allocator.allocate(
        profits=noisy_estimates,
        costs=costs,
        budget=5,
    )

    realized_profit = sum(clean_table[i][level] for i, level in enumerate(result["selected_levels"]))
    assert result["total_cost"] <= 5
    assert realized_profit >= 0.0
    assert len(result["selected_levels"]) == len(clean_table)


def test_robust_allocator_rejects_invalid_shrinkage() -> None:
    with pytest.raises(ValueError):
        RobustEqualizedAllocator(shrinkage=-0.1)

    with pytest.raises(ValueError):
        RobustEqualizedAllocator(shrinkage=1.5)
