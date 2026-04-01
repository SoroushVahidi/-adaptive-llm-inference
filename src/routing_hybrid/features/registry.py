from __future__ import annotations

from typing import Any, Callable

from src.routing_hybrid.features.action_features import add_action_features
from src.routing_hybrid.features.heuristic_features import add_heuristic_features
from src.routing_hybrid.features.interaction_features import add_interaction_features
from src.routing_hybrid.features.prompt_features import add_prompt_features
from src.routing_hybrid.features.risk_features import add_risk_features

FeatureFn = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]

FEATURE_REGISTRY: dict[str, FeatureFn] = {
    "prompt_features": add_prompt_features,
    "action_features": add_action_features,
    "interaction_features": add_interaction_features,
    "heuristic_features": add_heuristic_features,
    "risk_features": add_risk_features,
}


def apply_feature_families(
    candidate_rows: list[dict[str, Any]],
    families: list[str],
) -> list[dict[str, Any]]:
    out = candidate_rows
    for fam in families:
        fn = FEATURE_REGISTRY.get(fam)
        if fn is None:
            raise ValueError(f"Unknown feature family '{fam}'")
        out = fn(out)
    return out

