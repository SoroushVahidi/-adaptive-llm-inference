"""Allocator modules for adaptive test-time compute allocation."""

from .equal import EqualAllocator
from .mckp_allocator import MCKPAllocator

__all__ = ["EqualAllocator", "MCKPAllocator"]
