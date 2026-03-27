"""Conservative diagnostic allocator for noisy utility estimates.

This is intentionally not a final proposed method. Its role is to provide a
simple classical baseline that is less sensitive to noisy utility differences
than exact MCKP on raw estimates. We shrink each query's utility curve toward
its query-level mean before solving the same discrete allocation problem.
"""

from __future__ import annotations

from typing import Any, List, Union

import numpy as np

from .base import BaseAllocator
from .mckp_allocator import MCKPAllocator


class RobustEqualizedAllocator(BaseAllocator):
    """Shrink per-query utilities toward a query mean before MCKP allocation."""

    def __init__(self, shrinkage: float = 0.35) -> None:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        self.shrinkage = float(shrinkage)
        self._inner_allocator = MCKPAllocator()

    @property
    def name(self) -> str:
        return "robust_equalized"

    def allocate(
        self,
        profits: Union[List[List[float]], np.ndarray],
        costs: Union[List[int], np.ndarray],
        budget: int,
    ) -> dict[str, Any]:
        profits_arr = np.asarray(profits, dtype=float)
        if profits_arr.ndim != 2:
            raise ValueError("profits must be a rectangular 2-D table")

        query_means = np.mean(profits_arr, axis=1, keepdims=True)
        shrunk = (1.0 - self.shrinkage) * profits_arr + self.shrinkage * query_means
        shrunk = np.clip(shrunk, 0.0, 1.0)
        shrunk = np.maximum.accumulate(shrunk, axis=1)

        result = self._inner_allocator.allocate(
            profits=shrunk.tolist(),
            costs=costs,
            budget=budget,
        )
        result["adjusted_profits"] = shrunk.tolist()
        result["shrinkage"] = self.shrinkage
        return result
