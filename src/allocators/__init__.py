"""Allocator modules for adaptive test-time compute allocation."""

from .equal import EqualAllocator
from .mckp_allocator import MCKPAllocator
from .registry import get_allocator

__all__ = ["EqualAllocator", "MCKPAllocator", "get_allocator"]
