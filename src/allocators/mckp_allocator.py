"""
MCKP-style offline compute allocator for adaptive LLM inference.

This module implements a classical Multiple-Choice Knapsack Problem (MCKP)
solver using exact dynamic programming on integer budgets.

Role in the project
-------------------
In our adaptive test-time compute allocation research, each query must be
assigned exactly one *compute level* (e.g., number of reasoning steps or
sampling passes). Given predicted utilities and per-level costs, we want to
maximise total utility across a batch of queries while staying within a global
compute budget.  This module provides the **offline combinatorial optimisation
baseline**: it assumes all predictions are available before allocation begins,
and solves the resulting MCKP exactly.

MCKP formulation
----------------
* n queries, each with m+1 candidate levels  (0 … m).
* profits[i][k]  – predicted utility of assigning level k to query i.
* costs[k]       – integer compute cost of level k (same for all queries).
* budget B       – total integer compute budget.
* Decision: choose exactly one level k_i ∈ {0, …, m} for every query i,
  subject to Σ costs[k_i] ≤ B, maximising Σ profits[i][k_i].

DP state
--------
dp[b] = maximum total profit achievable for the first i queries
        using exactly budget b.

After processing all n queries the answer is max(dp[0..B]).
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from .base import BaseAllocator


class MCKPAllocator(BaseAllocator):
    """Exact MCKP solver via dynamic programming on integer budgets.

    Parameters
    ----------
    None – this allocator is stateless; all inputs are passed to
    :meth:`allocate`.

    Examples
    --------
    >>> allocator = MCKPAllocator()
    >>> profits = [[0.0, 1.0, 3.0], [0.0, 2.0, 2.5]]
    >>> costs   = [0, 1, 3]
    >>> result  = allocator.allocate(profits, costs, budget=4)
    >>> result["selected_levels"]
    [2, 1]
    >>> result["total_profit"]
    5.0
    >>> result["total_cost"]
    4
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        profits: Union[List[List[float]], np.ndarray],
        costs: Union[List[int], np.ndarray],
        budget: int,
    ) -> dict:
        """Solve the MCKP and return the optimal allocation.

        Parameters
        ----------
        profits:
            2-D array-like of shape ``[n_queries, n_levels]``.
            ``profits[i][k]`` is the predicted utility when query *i* is
            assigned compute level *k*.
        costs:
            1-D array-like of length ``n_levels``.  ``costs[k]`` is the
            **non-negative integer** compute cost of level *k*.
            Level 0 is conventionally the zero-compute fallback
            (``costs[0] == 0``), but any non-negative integer is accepted.
        budget:
            Non-negative integer total compute budget.

        Returns
        -------
        dict with keys:

        ``selected_levels`` : list[int]
            One chosen level index per query (length = n_queries).
        ``total_profit`` : float
            Sum of profits for the chosen levels.
        ``total_cost`` : int
            Sum of costs for the chosen levels.

        Raises
        ------
        ValueError
            If any input validation check fails (see :meth:`_validate`).
        """
        profits_arr, costs_arr = self._validate_and_convert(profits, costs, budget)
        selected_levels, total_profit, total_cost = self._solve_dp(
            profits_arr, costs_arr, int(budget)
        )
        return {
            "selected_levels": selected_levels,
            "total_profit": total_profit,
            "total_cost": total_cost,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_convert(
        profits: Union[List[List[float]], np.ndarray],
        costs: Union[List[int], np.ndarray],
        budget: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate inputs and return numpy arrays.

        Raises
        ------
        ValueError
            On any invalid input.
        TypeError
            If ``budget`` is not an integer type.
        """
        # Convert to numpy for uniform handling
        try:
            profits_arr = np.array(profits, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("profits must be convertible to a 2-D float array") from exc

        try:
            costs_arr = np.array(costs, dtype=int)
        except (ValueError, TypeError) as exc:
            raise ValueError("costs must be convertible to a 1-D integer array") from exc

        # Shape checks – a plain [] converts to shape (0,); treat as 0 queries.
        if profits_arr.ndim == 1 and profits_arr.shape[0] == 0:
            raise ValueError("profits must contain at least one query")
        if profits_arr.ndim != 2:
            raise ValueError(
                f"profits must be 2-D (n_queries × n_levels), got shape {profits_arr.shape}"
            )
        if costs_arr.ndim != 1:
            raise ValueError(
                f"costs must be 1-D (n_levels,), got shape {costs_arr.shape}"
            )

        n_queries, n_levels = profits_arr.shape

        if n_queries == 0:
            raise ValueError("profits must contain at least one query")
        if n_levels == 0:
            raise ValueError("each query must have at least one level")
        if len(costs_arr) != n_levels:
            raise ValueError(
                f"len(costs) = {len(costs_arr)} must equal n_levels = {n_levels}"
            )
        if np.any(costs_arr < 0):
            raise ValueError("all costs must be non-negative integers")

        # Budget check
        if not isinstance(budget, (int, np.integer)):
            raise TypeError(f"budget must be an integer, got {type(budget).__name__}")
        if budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}")

        return profits_arr, costs_arr

    # ------------------------------------------------------------------
    # DP solver
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_dp(
        profits: np.ndarray,
        costs: np.ndarray,
        budget: int,
    ) -> Tuple[List[int], float, int]:
        """Solve the MCKP with exact DP.

        State
        -----
        ``dp[b]`` = maximum total profit achievable for the first *i* queries
        spending exactly *b* units of budget (−∞ if infeasible).

        After each query we update dp in-place (sweeping from high to low
        budget, analogous to the 0/1 knapsack but replacing add/keep with a
        per-group maximisation over all levels).

        Reconstruction
        --------------
        ``choice[i][b]`` stores the level chosen for query *i* when the
        remaining budget entering query *i* was *b*.  We back-track from the
        optimal final budget to recover the full allocation.

        Parameters
        ----------
        profits:
            Shape ``[n_queries, n_levels]``.
        costs:
            Shape ``[n_levels]``.
        budget:
            Non-negative integer.

        Returns
        -------
        selected_levels : list[int]
        total_profit    : float
        total_cost      : int
        """
        n_queries, n_levels = profits.shape
        NEG_INF = -np.inf

        # dp[b] = best profit for queries processed so far using budget b.
        # Initialise: before processing any query, budget 0 costs nothing.
        dp = np.full(budget + 1, NEG_INF, dtype=float)
        dp[0] = 0.0

        # choice[i][b] = level chosen for query i given remaining budget b.
        choice = np.full((n_queries, budget + 1), -1, dtype=int)

        for i in range(n_queries):
            new_dp = np.full(budget + 1, NEG_INF, dtype=float)

            for k in range(n_levels):
                ck = int(costs[k])
                pk = profits[i, k]
                # For every previous budget b, we can use level k for query i
                # if b + ck <= budget and dp[b] is feasible.
                for b in range(budget - ck + 1):
                    if dp[b] == NEG_INF:
                        continue
                    candidate = dp[b] + pk
                    nb = b + ck
                    if candidate > new_dp[nb]:
                        new_dp[nb] = candidate
                        choice[i][nb] = k

            dp = new_dp

        # Find the budget achieving maximum profit
        best_b = int(np.argmax(dp))
        best_profit = dp[best_b]

        if best_profit == NEG_INF:
            # Infeasible (shouldn't happen when costs[0]==0 for all queries,
            # but handle gracefully by falling back to level 0 everywhere)
            selected = [0] * n_queries
            total_cost = int(np.sum(costs[selected]))
            total_profit = float(np.sum(profits[i, 0] for i in range(n_queries)))
            return selected, total_profit, total_cost

        # Back-track to recover selected levels
        selected_levels = [0] * n_queries
        remaining = best_b
        for i in range(n_queries - 1, -1, -1):
            k = int(choice[i][remaining])
            selected_levels[i] = k
            remaining -= int(costs[k])

        total_profit = float(best_profit)
        total_cost = int(best_b)
        return selected_levels, total_profit, total_cost
