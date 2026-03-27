"""Equal-budget allocator: splits the budget uniformly across queries."""

from __future__ import annotations

class EqualAllocator:
    """Divide the global budget equally among all queries.

    Any remainder samples are distributed round-robin to the first queries.
    """

    @property
    def name(self) -> str:
        return "equal"

    def allocate(self, n_queries: int, budget: int) -> list[int]:
        if n_queries == 0:
            return []
        base = budget // n_queries
        remainder = budget % n_queries
        return [base + (1 if i < remainder else 0) for i in range(n_queries)]
