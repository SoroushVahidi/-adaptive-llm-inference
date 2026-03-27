from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseAllocator(ABC):
    """
    Base interface for compute allocation algorithms.

    An allocator assigns a compute level to each query under a total budget.
    """

    @abstractmethod
    def allocate(
        self,
        profits: List[List[float]],
        costs: List[int],
        budget: int,
    ) -> Tuple[List[int], float, int]:
        """
        Args:
            profits: 2D list [n_queries][n_levels]
            costs: list of length n_levels
            budget: total budget

        Returns:
            selected_levels: list[int]
            total_profit: float
            total_cost: int
        """
        pass