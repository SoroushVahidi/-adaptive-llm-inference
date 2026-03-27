"""Abstract allocator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


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
