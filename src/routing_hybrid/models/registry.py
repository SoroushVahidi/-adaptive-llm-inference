from __future__ import annotations

from typing import Any

from src.routing_hybrid.models.adaboost import AdaBoostHybridModel
from src.routing_hybrid.models.bagging import BaggingTreeHybridModel
from src.routing_hybrid.models.dummy import DummyHybridModel
from src.routing_hybrid.models.gradient_boosting import GradientBoostingHybridModel
from src.routing_hybrid.models.logistic import LogisticHybridModel
from src.routing_hybrid.models.random_forest import RandomForestHybridModel
from src.routing_hybrid.models.tree import TreeHybridModel

MODEL_REGISTRY = {
    "dummy": DummyHybridModel,
    "logistic": LogisticHybridModel,
    "tree": TreeHybridModel,
    "bagging_tree": BaggingTreeHybridModel,
    "random_forest": RandomForestHybridModel,
    "gradient_boosting": GradientBoostingHybridModel,
    "adaboost": AdaBoostHybridModel,
}


def make_model(name: str, seed: int = 42, model_params: dict[str, Any] | None = None) -> Any:
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}")
    return cls(seed=seed, **(model_params or {}))

