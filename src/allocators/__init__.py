"""Allocator modules for adaptive test-time compute allocation."""

from .equal import EqualAllocator
from .mckp_allocator import MCKPAllocator
from .registry import get_allocator
from .robust_equalized import RobustEqualizedAllocator

__all__ = [
    "EqualAllocator",
    "MCKPAllocator",
    "RobustEqualizedAllocator",
    "get_allocator",
]
