"""Tests for MCKPAllocator.

Each test is self-contained and uses small, hand-checkable examples so
that the expected answers can be verified independently of the code.
"""

import pytest
import numpy as np

from src.allocators.mckp_allocator import MCKPAllocator
from src.allocators import MCKPAllocator as MCKPAllocatorFromInit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def allocator():
    return MCKPAllocator()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    """Hand-checkable examples with known optimal solutions."""

    def test_tiny_example(self, allocator):
        """Two queries, three levels – manually verify the optimum.

        profits = [[0, 1, 3],
                   [0, 2, 2.5]]
        costs   = [0, 1, 3]
        budget  = 4

        Enumeration:
          (0,0): profit=0+0=0,   cost=0+0=0  ✓
          (0,1): profit=0+2=2,   cost=0+1=1  ✓
          (0,2): profit=0+2.5,   cost=0+3=3  ✓
          (1,0): profit=1+0=1,   cost=1+0=1  ✓
          (1,1): profit=1+2=3,   cost=1+1=2  ✓
          (1,2): profit=1+2.5=3.5, cost=1+3=4 ✓  <-- optimum
          (2,0): profit=3+0=3,   cost=3+0=3  ✓
          (2,1): profit=3+2=5,   cost=3+1=4  ✓  <-- optimal
          (2,2): profit=3+2.5=5.5, cost=3+3=6 ✗ budget exceeded

        Best feasible: (2,1) with profit=5.0 and cost=4.
        """
        profits = [[0.0, 1.0, 3.0], [0.0, 2.0, 2.5]]
        costs   = [0, 1, 3]
        result  = allocator.allocate(profits, costs, budget=4)

        assert result["selected_levels"] == [2, 1]
        assert result["total_profit"] == pytest.approx(5.0)
        assert result["total_cost"] == 4

    def test_single_query(self, allocator):
        """With one query, choose the best affordable level."""
        profits = [[0.0, 1.5, 5.0, 2.0]]
        costs   = [0, 1, 3, 2]
        result  = allocator.allocate(profits, costs, budget=3)

        assert result["selected_levels"] == [2]
        assert result["total_profit"] == pytest.approx(5.0)
        assert result["total_cost"] == 3

    def test_exact_one_choice_per_query(self, allocator):
        """Verify exactly one level is selected per query."""
        profits = [[0.0, 1.0], [0.0, 2.0], [0.0, 1.5]]
        costs   = [0, 2]
        result  = allocator.allocate(profits, costs, budget=4)

        levels = result["selected_levels"]
        assert len(levels) == 3
        for lvl in levels:
            assert lvl in (0, 1)

        # Reconstruct cost and profit from selections
        recon_cost   = sum(costs[lvl] for lvl in levels)
        recon_profit = sum(profits[i][levels[i]] for i in range(3))
        assert recon_cost <= 4
        assert recon_profit == pytest.approx(result["total_profit"])
        assert recon_cost   == result["total_cost"]

    def test_numpy_input(self, allocator):
        """The allocator must accept numpy arrays as inputs."""
        profits = np.array([[0.0, 1.0, 3.0], [0.0, 2.0, 2.5]])
        costs   = np.array([0, 1, 3])
        result  = allocator.allocate(profits, costs, budget=4)

        assert result["selected_levels"] == [2, 1]
        assert result["total_profit"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Budget-constrained behaviour
# ---------------------------------------------------------------------------

class TestBudgetConstraints:
    """Cases where the budget forces some queries to lower levels."""

    def test_tight_budget_forces_lower_levels(self, allocator):
        """Budget of 1 can only afford one query at level 1, the rest at 0."""
        profits = [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]
        costs   = [0, 1]
        result  = allocator.allocate(profits, costs, budget=1)

        levels = result["selected_levels"]
        assert sum(levels) == 1           # exactly one query at level 1
        assert result["total_cost"] == 1
        assert result["total_profit"] == pytest.approx(10.0)

    def test_budget_zero(self, allocator):
        """With zero budget every query must be assigned level 0 (cost 0)."""
        profits = [[0.0, 5.0, 9.0], [0.0, 3.0, 7.0]]
        costs   = [0, 2, 4]
        result  = allocator.allocate(profits, costs, budget=0)

        assert result["selected_levels"] == [0, 0]
        assert result["total_profit"] == pytest.approx(0.0)
        assert result["total_cost"] == 0

    def test_large_budget_picks_best_levels(self, allocator):
        """With plenty of budget every query should get its best level."""
        profits = [[0.0, 1.0, 5.0], [0.0, 2.0, 6.0]]
        costs   = [0, 1, 3]
        result  = allocator.allocate(profits, costs, budget=100)

        assert result["selected_levels"] == [2, 2]
        assert result["total_profit"] == pytest.approx(11.0)
        assert result["total_cost"] == 6

    def test_budget_exactly_at_optimum(self, allocator):
        """Budget equals the exact cost of the optimal solution."""
        profits = [[0.0, 1.0, 4.0], [0.0, 2.0, 3.0]]
        costs   = [0, 1, 2]
        # Optimal: both at level 2, cost = 4, profit = 7
        result  = allocator.allocate(profits, costs, budget=4)

        assert result["selected_levels"] == [2, 2]
        assert result["total_profit"] == pytest.approx(7.0)
        assert result["total_cost"] == 4

    def test_partial_budget_forces_tradeoff(self, allocator):
        """Budget forces a tradeoff; verify the greedy-incorrect choice is avoided.

        profits = [[0, 1, 10],
                   [0, 5,  6]]
        costs   = [0, 1,  3]
        budget  = 4

        Options (c0+c1 ≤ 4):
          (0,0): 0+0=0,  cost 0
          (0,1): 0+5=5,  cost 1
          (0,2): 0+6=6,  cost 3
          (1,0): 1+0=1,  cost 1
          (1,1): 1+5=6,  cost 2
          (1,2): 1+6=7,  cost 4  ← best profit at cost 4
          (2,0): 10+0=10, cost 3
          (2,1): 10+5=15, cost 4  ← **global optimum**
          (2,2): 10+6=16, cost 6  ✗ exceeds budget
        """
        profits = [[0.0, 1.0, 10.0], [0.0, 5.0, 6.0]]
        costs   = [0, 1, 3]
        result  = allocator.allocate(profits, costs, budget=4)

        assert result["selected_levels"] == [2, 1]
        assert result["total_profit"] == pytest.approx(15.0)
        assert result["total_cost"] == 4


# ---------------------------------------------------------------------------
# Consistency / invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Invariants that must hold for all valid inputs."""

    def test_cost_and_profit_are_consistent(self, allocator):
        """Reconstructed cost/profit from selected_levels must match returned values."""
        profits = [[0.0, 2.0, 5.0], [0.0, 3.0, 4.0], [0.0, 1.0, 6.0]]
        costs   = [0, 2, 3]
        result  = allocator.allocate(profits, costs, budget=6)

        levels = result["selected_levels"]
        recon_profit = sum(profits[i][levels[i]] for i in range(3))
        recon_cost   = sum(costs[levels[i]] for i in range(3))

        assert recon_profit == pytest.approx(result["total_profit"])
        assert recon_cost   == result["total_cost"]
        assert recon_cost   <= 6

    def test_returned_levels_are_valid_indices(self, allocator):
        """Each selected level must be a valid index into profits[i]."""
        profits = [[0.0, 1.0, 2.0, 3.0]] * 5
        costs   = [0, 1, 2, 3]
        result  = allocator.allocate(profits, costs, budget=7)

        for lvl in result["selected_levels"]:
            assert 0 <= lvl < 4


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """The allocator must raise ValueError for invalid inputs."""

    def test_profits_not_2d(self, allocator):
        with pytest.raises(ValueError, match="2-D"):
            allocator.allocate([1.0, 2.0], [0, 1], budget=1)

    def test_costs_not_1d(self, allocator):
        with pytest.raises(ValueError, match="1-D"):
            allocator.allocate([[0.0, 1.0]], [[0, 1]], budget=1)

    def test_costs_length_mismatch(self, allocator):
        with pytest.raises(ValueError, match="n_levels"):
            allocator.allocate([[0.0, 1.0, 2.0]], [0, 1], budget=2)

    def test_negative_budget(self, allocator):
        with pytest.raises(ValueError, match="non-negative"):
            allocator.allocate([[0.0, 1.0]], [0, 1], budget=-1)

    def test_negative_cost(self, allocator):
        with pytest.raises(ValueError, match="non-negative"):
            allocator.allocate([[0.0, 1.0]], [0, -1], budget=2)

    def test_empty_queries(self, allocator):
        with pytest.raises(ValueError, match="at least one query"):
            allocator.allocate([], [0, 1], budget=2)

    def test_empty_levels(self, allocator):
        with pytest.raises(ValueError, match="at least one level"):
            allocator.allocate([[], []], [], budget=2)

    def test_non_integer_budget(self, allocator):
        with pytest.raises(TypeError, match="integer"):
            allocator.allocate([[0.0, 1.0]], [0, 1], budget=1.5)


# ---------------------------------------------------------------------------
# Import surface
# ---------------------------------------------------------------------------

class TestImportSurface:
    """The allocator must be importable from the package __init__."""

    def test_import_from_package(self):
        assert MCKPAllocatorFromInit is MCKPAllocator
