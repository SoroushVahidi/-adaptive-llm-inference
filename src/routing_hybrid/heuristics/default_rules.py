from __future__ import annotations

from typing import Any


def rule_prune_dominated_cost(candidate_row: dict[str, Any]) -> dict[str, Any]:
    if float(candidate_row.get("gain_vs_cheapest", 0.0)) <= 0.0 and float(
        candidate_row.get("hfeat_action_cost_rank", 0.0)
    ) > 0.0:
        candidate_row["heur_dominated_action"] = 1.0
    else:
        candidate_row["heur_dominated_action"] = 0.0
    return candidate_row


def rule_cheap_safe(candidate_row: dict[str, Any]) -> dict[str, Any]:
    conf = float(candidate_row.get("hfeat_prompt_unified_confidence", 0.0))
    is_revise = float(candidate_row.get("hfeat_action_is_revise", 0.0)) > 0.5
    fired = conf > 0.75 and not is_revise
    candidate_row["heur_cheap_safe_flag"] = 1.0 if fired else 0.0
    if fired:
        candidate_row["heur_utility_adjustment"] = float(candidate_row.get("heur_utility_adjustment", 0.0)) + 0.03
    return candidate_row


def rule_revise_favorable(candidate_row: dict[str, Any]) -> dict[str, Any]:
    parse_risk = float(candidate_row.get("hfeat_heuristic_parse_risk", 0.0))
    is_revise = float(candidate_row.get("hfeat_action_is_revise", 0.0)) > 0.5
    fired = parse_risk > 0.35 and is_revise
    candidate_row["heur_revise_favorable_flag"] = 1.0 if fired else 0.0
    if fired:
        candidate_row["heur_utility_adjustment"] = float(candidate_row.get("heur_utility_adjustment", 0.0)) + 0.05
    return candidate_row


def rule_answer_format_risk(candidate_row: dict[str, Any]) -> dict[str, Any]:
    parse_success = float(candidate_row.get("feat_fp_first_pass_parse_success", 1.0))
    candidate_row["heur_parse_risk_flag"] = 1.0 if parse_success < 0.5 else 0.0
    if parse_success < 0.5 and float(candidate_row.get("hfeat_action_is_direct", 0.0)) > 0.5:
        candidate_row["heur_utility_adjustment"] = float(candidate_row.get("heur_utility_adjustment", 0.0)) - 0.04
    return candidate_row


def rule_ambiguity_penalty(candidate_row: dict[str, Any]) -> dict[str, Any]:
    ambiguity = float(
        candidate_row.get(
            "hfeat_heuristic_ambiguity_risk",
            candidate_row.get("feat_cons_constraint_word_conflict_suspected", 0.0),
        )
    )
    candidate_row["heur_ambiguity_risk_flag"] = 1.0 if ambiguity > 0.5 else 0.0
    if ambiguity > 0.5 and float(candidate_row.get("hfeat_action_is_direct", 0.0)) > 0.5:
        candidate_row["heur_utility_adjustment"] = float(candidate_row.get("heur_utility_adjustment", 0.0)) - 0.03
    return candidate_row

