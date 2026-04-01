from __future__ import annotations

from typing import Any, Callable

from src.routing_hybrid.heuristics.default_rules import (
    rule_ambiguity_penalty,
    rule_answer_format_risk,
    rule_cheap_safe,
    rule_prune_dominated_cost,
    rule_revise_favorable,
)

RuleFn = Callable[[dict[str, Any]], dict[str, Any]]

HEURISTIC_REGISTRY: dict[str, RuleFn] = {
    "prune_dominated_cost": rule_prune_dominated_cost,
    "cheap_safe": rule_cheap_safe,
    "revise_favorable": rule_revise_favorable,
    "answer_format_risk": rule_answer_format_risk,
    "ambiguity_penalty": rule_ambiguity_penalty,
}


def apply_heuristics(candidate_rows: list[dict[str, Any]], rules: list[str]) -> list[dict[str, Any]]:
    out = candidate_rows
    for r in out:
        r.setdefault("heur_forbidden", 0.0)
        r.setdefault("heur_dominated_action", 0.0)
        r.setdefault("heur_utility_adjustment", 0.0)
    for rule_name in rules:
        fn = HEURISTIC_REGISTRY.get(rule_name)
        if fn is None:
            raise ValueError(f"Unknown heuristic rule '{rule_name}'")
        for i, row in enumerate(out):
            out[i] = fn(row)
            if float(out[i].get("heur_dominated_action", 0.0)) > 0.5:
                out[i]["heur_forbidden"] = 1.0
    return out

