"""Allocator base interfaces for adaptive test-time compute allocation."""

from abc import ABC, abstractmethod


class Allocator(ABC):
    """Decides how many samples to allocate per query given a global budget."""

    @abstractmethod
    def allocate(self, n_queries: int, budget: int) -> list[int]:
        """Return per-query sample counts summing to at most *budget*."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the allocator."""


class BaseAllocator(ABC):
    """Base interface for compute allocation algorithms.

    Subclasses must implement ``allocate``, which assigns one compute level
    per query while respecting a global integer budget.
    """

    @abstractmethod
    def allocate(self, profits, costs, budget: int) -> dict:
        """Allocate compute levels to queries under a total budget.

        Args:
            profits: 2-D array-like ``[n_queries, n_levels]`` of utilities.
            costs: 1-D array-like ``[n_levels]`` of integer compute costs.
            budget: Total integer compute budget.

        Returns:
            Dict with ``selected_levels``, ``total_profit``, ``total_cost``.
        """
