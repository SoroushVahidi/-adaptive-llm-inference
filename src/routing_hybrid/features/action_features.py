from __future__ import annotations

from typing import Any


def add_action_features(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for r in candidate_rows:
        by_prompt.setdefault(str(r["prompt_id"]), []).append(r)
    for rows in by_prompt.values():
        ordered = sorted(rows, key=lambda x: (float(x["action_cost"]), str(x["action_name"])))
        rank = {str(r["action_name"]): i for i, r in enumerate(ordered)}
        for r in rows:
            name = str(r["action_name"])
            fam = str(r["action_family"])
            r["hfeat_action_cost"] = float(r["action_cost"])
            r["hfeat_action_cost_rank"] = float(rank[name])
            r["hfeat_action_is_revise"] = 1.0 if "revise" in name or fam == "revise" else 0.0
            r["hfeat_action_is_direct"] = 1.0 if "direct" in name else 0.0
            r["hfeat_action_is_reasoning"] = 1.0 if "reasoning" in name or fam == "reasoning" else 0.0
            r["hfeat_action_is_multi_sample"] = 1.0 if fam == "multi_sample" else 0.0
    return candidate_rows

