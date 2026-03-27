from src.allocators.equal import EqualAllocator


def test_equal_split():
    alloc = EqualAllocator()
    result = alloc.allocate(5, 25)
    assert result == [5, 5, 5, 5, 5]
    assert sum(result) == 25


def test_equal_split_with_remainder():
    alloc = EqualAllocator()
    result = alloc.allocate(3, 10)
    assert sum(result) == 10
    assert result == [4, 3, 3]


def test_zero_queries():
    alloc = EqualAllocator()
    assert alloc.allocate(0, 10) == []


def test_budget_less_than_queries():
    alloc = EqualAllocator()
    result = alloc.allocate(5, 3)
    assert sum(result) == 3
    assert result == [1, 1, 1, 0, 0]
