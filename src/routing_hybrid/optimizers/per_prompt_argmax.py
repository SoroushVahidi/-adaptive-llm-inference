from __future__ import annotations

from typing import Any


class PerPromptArgmaxOptimizer:
    name = "per_prompt_argmax"

    def solve(self, candidate_rows: list[dict[str, Any]], budget: float) -> dict[str, Any]:
        by_prompt: dict[str, list[dict[str, Any]]] = {}
        for r in candidate_rows:
            by_prompt.setdefault(str(r["prompt_id"]), []).append(r)
        chosen: dict[str, str] = {}
        total_cost = 0.0
        obj = 0.0
        for pid, rows in by_prompt.items():
            best = max(rows, key=lambda x: (float(x["final_utility"]), -float(x["action_cost"]), str(x["action_name"])))
            chosen[pid] = str(best["action_name"])
            total_cost += float(best["action_cost"])
            obj += float(best["final_utility"])
        return {
            "chosen_by_prompt": chosen,
            "objective_value": obj,
            "total_cost": total_cost,
            "budget": budget,
            "optimizer_name": self.name,
        }

