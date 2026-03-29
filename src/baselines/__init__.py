from src.baselines.best_of_n import BestOfNBaseline
from src.baselines.greedy import GreedyBaseline
from src.baselines.self_consistency import (
    SelfConsistencyBaseline,
    majority_vote_self_consistency,
)

__all__ = [
    "GreedyBaseline",
    "BestOfNBaseline",
    "SelfConsistencyBaseline",
    "majority_vote_self_consistency",
]
