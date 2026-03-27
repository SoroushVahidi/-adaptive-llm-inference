"""Allocator base interfaces for adaptive test-time compute allocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Allocator(ABC):
    """Decides how many samples to allocate per query given a global budget."""

    @abstractmethod
    def allocate(self, n_queries: int, budget: int) -> list[int]:
        """Return a list of per-query sample counts summing to at most *budget*.

        Args:
            n_queries: Total number of queries.
            budget: Total number of samples allowed.

        Returns:
            List of length *n_queries* with per-query sample counts.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the allocator."""


class BaseAllocator(ABC):
    """Abstract base class for MCKP-style compute budget allocators.

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
