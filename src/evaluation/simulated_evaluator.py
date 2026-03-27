"""Evaluator for synthetic utility-table allocation experiments.

This synthetic mode is used for:
1) testing allocation algorithms directly on explicit per-query utility curves, and
2) validating optimization logic before connecting allocators to estimated
   utility curves from real LLM experiments.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.allocators.registry import get_allocator


def evaluate_simulated_allocation(
    utility_table: Sequence[Sequence[float]],
    costs: Sequence[int],
    budget: int,
    allocator_name: str,
) -> dict[str, Any]:
    """Run an allocator on a synthetic instance and compute summary metrics."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not utility_table:
        raise ValueError("utility_table must contain at least one query")

    n_queries = len(utility_table)
    n_levels = len(utility_table[0])
    if n_levels == 0:
        raise ValueError("utility_table must contain at least one level")
    if any(len(row) != n_levels for row in utility_table):
        raise ValueError("utility_table must be rectangular [n_queries][n_levels]")
    if len(costs) != n_levels:
        raise ValueError("len(costs) must equal number of levels")

    allocator = get_allocator(allocator_name)
    result = allocator.allocate(profits=utility_table, costs=costs, budget=budget)

    selected_levels = [int(level) for level in result["selected_levels"]]
    total_utility = float(result["total_profit"])
    total_cost = int(result["total_cost"])

    if len(selected_levels) != n_queries:
        raise ValueError("Allocator returned wrong number of selected levels")
    if total_cost > budget:
        raise ValueError("Allocator returned an over-budget solution")

    return {
        "allocator": allocator_name,
        "selected_levels": selected_levels,
        "total_expected_utility": total_utility,
        "total_cost": total_cost,
        "average_cost_per_query": total_cost / n_queries,
        "average_utility_per_query": total_utility / n_queries,
    }

