"""Equal-budget allocator utilities.

This allocator supports two call modes:
1) legacy mode used in the existing runner:
   ``allocate(n_queries, budget) -> list[int]`` (sample counts per query)
2) MCKP-style mode used by the simulated allocation pipeline:
   ``allocate(profits, costs, budget) -> dict``
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import BaseAllocator


class EqualAllocator(BaseAllocator):
    """Simple equal-share allocator for both legacy and simulated paths."""

    @property
    def name(self) -> str:
        return "equal"

    def allocate(
        self,
        profits: int | Sequence[Sequence[float]],
        costs: int | Sequence[int],
        budget: int | None = None,
    ) -> list[int] | dict[str, Any]:
        """Allocate compute.

        Legacy mode:
            allocate(n_queries: int, budget: int) -> list[int]
        Simulated mode:
            allocate(profits: [n_queries][n_levels], costs: [n_levels], budget: int)
            -> {"selected_levels", "total_profit", "total_cost"}
        """
        if isinstance(profits, int):
            if not isinstance(costs, int):
                raise TypeError("Legacy mode requires integer budget")
            return self._allocate_legacy(n_queries=profits, budget=costs)

        if budget is None:
            raise TypeError(
                "Simulated mode requires allocate(profits, costs, budget) with budget set"
            )
        if not isinstance(costs, Sequence):
            raise TypeError("Simulated mode requires costs as a sequence of integers")

        return self._allocate_simulated(profits=profits, costs=costs, budget=budget)

    @staticmethod
    def _allocate_legacy(n_queries: int, budget: int) -> list[int]:
        if n_queries < 0:
            raise ValueError("n_queries must be non-negative")
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if n_queries == 0:
            return []
        base = budget // n_queries
        remainder = budget % n_queries
        return [base + (1 if i < remainder else 0) for i in range(n_queries)]

    @staticmethod
    def _allocate_simulated(
        profits: Sequence[Sequence[float]],
        costs: Sequence[int],
        budget: int,
    ) -> dict[str, Any]:
        profits_table = [[float(v) for v in row] for row in profits]
        costs_vec = [int(c) for c in costs]

        if budget < 0:
            raise ValueError("budget must be non-negative")
        if not profits_table:
            raise ValueError("profits must contain at least one query")
        n_levels = len(profits_table[0])
        if n_levels == 0:
            raise ValueError("profits must contain at least one level")
        if any(len(row) != n_levels for row in profits_table):
            raise ValueError("profits must be rectangular [n_queries][n_levels]")
        if len(costs_vec) != n_levels:
            raise ValueError("len(costs) must equal n_levels")
        if any(c < 0 for c in costs_vec):
            raise ValueError("costs must be non-negative integers")
        if any(costs_vec[i] > costs_vec[i + 1] for i in range(len(costs_vec) - 1)):
            raise ValueError("costs must be non-decreasing across levels")

        n_queries = len(profits_table)
        min_level = min(range(n_levels), key=lambda idx: costs_vec[idx])
        min_total_cost = n_queries * costs_vec[min_level]
        if min_total_cost > budget:
            raise ValueError(
                f"budget {budget} is infeasible: minimum required cost is {min_total_cost}"
            )

        per_query_share = budget // n_queries
        base_level_candidates = [k for k, c in enumerate(costs_vec) if c <= per_query_share]
        base_level = base_level_candidates[-1] if base_level_candidates else min_level

        selected_levels = [base_level] * n_queries
        total_cost = n_queries * costs_vec[base_level]
        total_profit = float(sum(profits_table[i][base_level] for i in range(n_queries)))
        remaining = budget - total_cost

        # Spend leftover budget by local one-level upgrades with best marginal gain.
        while remaining > 0:
            best_query = -1
            best_gain = 0.0
            best_extra_cost = 0

            for i in range(n_queries):
                current = selected_levels[i]
                if current + 1 >= n_levels:
                    continue
                nxt = current + 1
                extra_cost = costs_vec[nxt] - costs_vec[current]
                if extra_cost <= 0 or extra_cost > remaining:
                    continue
                gain = profits_table[i][nxt] - profits_table[i][current]
                if gain > best_gain or (
                    gain == best_gain and best_query >= 0 and extra_cost < best_extra_cost
                ):
                    best_query = i
                    best_gain = float(gain)
                    best_extra_cost = int(extra_cost)

            if best_query < 0 or best_gain <= 0.0:
                break

            selected_levels[best_query] += 1
            total_cost += best_extra_cost
            total_profit += best_gain
            remaining -= best_extra_cost

        return {
            "selected_levels": selected_levels,
            "total_profit": float(total_profit),
            "total_cost": int(total_cost),
        }
