"""Allocator registry for simulation experiments."""

from __future__ import annotations

from src.allocators.equal import EqualAllocator
from src.allocators.mckp_allocator import MCKPAllocator


def get_allocator(name: str):
    """Return an allocator instance by name.

    Supported names:
      - "equal"
      - "mckp"
    """
    normalized = name.strip().lower()
    if normalized == "equal":
        return EqualAllocator()
    if normalized == "mckp":
        return MCKPAllocator()
    raise ValueError(f"Unknown allocator '{name}'. Supported: ['equal', 'mckp']")

