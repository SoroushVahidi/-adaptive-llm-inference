"""Base allocator interface for adaptive test-time compute allocation."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAllocator(ABC):
    """Abstract base class for compute budget allocators.

    All allocators must implement the ``allocate`` method, which takes a
    profit matrix, a cost vector, and a budget, and returns an allocation
    result.
    """

    @abstractmethod
    def allocate(self, profits: Any, costs: Any, budget: int) -> dict:
        """Allocate compute levels to queries under a total budget.

        Parameters
        ----------
        profits:
            2-D array-like of shape ``[n_queries, n_levels]`` where
            ``profits[i][k]`` is the predicted utility of assigning level
            *k* to query *i*.
        costs:
            1-D array-like of length ``n_levels`` where ``costs[k]`` is
            the integer compute cost of level *k*.
        budget:
            Total integer compute budget available across all queries.

        Returns
        -------
        dict
            A dictionary containing at least:

            * ``selected_levels`` – list of ints, one per query.
            * ``total_profit`` – total achieved profit (float).
            * ``total_cost`` – total compute cost used (int).
        """
