from __future__ import annotations

from src.routing_hybrid.optimizers.greedy_upgrade import GreedyUpgradeOptimizer
from src.routing_hybrid.optimizers.lambda_search import LambdaSearchOptimizer
from src.routing_hybrid.optimizers.mckp_exact import MCKPExactOptimizer
from src.routing_hybrid.optimizers.per_prompt_argmax import PerPromptArgmaxOptimizer


def _toy_rows() -> list[dict[str, object]]:
    return [
        {"prompt_id": "p1", "action_name": "a0", "action_cost": 1.0, "final_utility": 0.4},
        {"prompt_id": "p1", "action_name": "a1", "action_cost": 2.0, "final_utility": 0.9},
        {"prompt_id": "p2", "action_name": "a0", "action_cost": 1.0, "final_utility": 0.5},
        {"prompt_id": "p2", "action_name": "a1", "action_cost": 2.0, "final_utility": 0.7},
    ]


def test_per_prompt_argmax() -> None:
    out = PerPromptArgmaxOptimizer().solve(_toy_rows(), budget=10.0)
    assert out["chosen_by_prompt"]["p1"] == "a1"
    assert out["chosen_by_prompt"]["p2"] == "a1"


def test_greedy_upgrade() -> None:
    out = GreedyUpgradeOptimizer().solve(_toy_rows(), budget=3.0)
    assert set(out["chosen_by_prompt"]) == {"p1", "p2"}


def test_mckp_exact() -> None:
    out = MCKPExactOptimizer(cost_scale=10).solve(_toy_rows(), budget=3.0)
    assert set(out["chosen_by_prompt"]) == {"p1", "p2"}
    assert float(out["total_cost"]) <= 3.0 + 1e-6


def test_lambda_search_optimizer() -> None:
    out = LambdaSearchOptimizer(iters=10).solve(_toy_rows(), budget=3.0)
    assert set(out["chosen_by_prompt"]) == {"p1", "p2"}
    assert float(out["total_cost"]) <= 3.0 + 1e-6
