from __future__ import annotations

from typing import Any


def add_heuristic_features(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in candidate_rows:
        confidence = float(r.get("hfeat_prompt_unified_confidence", 0.0))
        parse_risk = 1.0 - float(r.get("feat_fp_first_pass_parse_success", 1.0))
        revise = float(r.get("hfeat_action_is_revise", 0.0))
        r["hfeat_heuristic_revise_favorable"] = 1.0 if (parse_risk > 0.3 and revise > 0.5) else 0.0
        r["hfeat_heuristic_cheap_safe"] = 1.0 if (confidence > 0.75 and revise < 0.5) else 0.0
        r["hfeat_heuristic_parse_risk"] = parse_risk
        r["hfeat_heuristic_ambiguity_risk"] = float(r.get("feat_cons_constraint_word_conflict_suspected", 0.0))
    return candidate_rows

