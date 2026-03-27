"""Allocator registry for simulation experiments."""

from __future__ import annotations

from src.allocators.equal import EqualAllocator
from src.allocators.mckp_allocator import MCKPAllocator
from src.allocators.robust_equalized import RobustEqualizedAllocator


def get_allocator(name: str):
    """Return an allocator instance by name.

    Supported names:
      - "equal"
      - "mckp"
      - "robust_equalized"
    """
    normalized = name.strip().lower()
    if normalized == "equal":
        return EqualAllocator()
    if normalized == "mckp":
        return MCKPAllocator()
    if normalized == "robust_equalized":
        return RobustEqualizedAllocator()
    raise ValueError(
        "Unknown allocator "
        f"'{name}'. Supported: ['equal', 'mckp', 'robust_equalized']"
    )

