from __future__ import annotations

from typing import Any


def add_interaction_features(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in candidate_rows:
        complexity = float(r.get("hfeat_prompt_length_tokens", 0.0)) / 50.0 + float(
            r.get("hfeat_prompt_num_count", 0.0)
        ) / 5.0
        revise = float(r.get("hfeat_action_is_revise", 0.0))
        parse_risk = 1.0 - float(r.get("feat_fp_first_pass_parse_success", 1.0))
        r["hfeat_interaction_complexity_x_revise"] = complexity * revise
        r["hfeat_interaction_parse_risk_x_direct"] = parse_risk * float(r.get("hfeat_action_is_direct", 0.0))
        r["hfeat_interaction_confidence_x_reasoning"] = float(
            r.get("hfeat_prompt_unified_confidence", 0.0)
        ) * float(r.get("hfeat_action_is_reasoning", 0.0))
    return candidate_rows

