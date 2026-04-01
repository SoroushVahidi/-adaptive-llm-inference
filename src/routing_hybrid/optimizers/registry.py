from __future__ import annotations

from typing import Any

from src.routing_hybrid.optimizers.greedy_upgrade import GreedyUpgradeOptimizer
from src.routing_hybrid.optimizers.lambda_search import LambdaSearchOptimizer
from src.routing_hybrid.optimizers.mckp_exact import MCKPExactOptimizer
from src.routing_hybrid.optimizers.per_prompt_argmax import PerPromptArgmaxOptimizer


def make_optimizer(name: str, optimizer_params: dict[str, Any] | None = None) -> Any:
    params = optimizer_params or {}
    if name == "per_prompt_argmax":
        return PerPromptArgmaxOptimizer()
    if name == "greedy_upgrade":
        return GreedyUpgradeOptimizer()
    if name == "mckp_exact":
        return MCKPExactOptimizer(cost_scale=int(params.get("cost_scale", 100)))
    if name == "lambda_search":
        return LambdaSearchOptimizer(iters=int(params.get("iters", 24)))
    raise ValueError(f"Unknown optimizer '{name}'")

