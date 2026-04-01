from __future__ import annotations

from typing import Any


class GreedyUpgradeOptimizer:
    name = "greedy_upgrade"

    def solve(self, candidate_rows: list[dict[str, Any]], budget: float) -> dict[str, Any]:
        by_prompt: dict[str, list[dict[str, Any]]] = {}
        for r in candidate_rows:
            by_prompt.setdefault(str(r["prompt_id"]), []).append(r)

        chosen_rows: dict[str, dict[str, Any]] = {}
        for pid, rows in by_prompt.items():
            chosen_rows[pid] = min(rows, key=lambda x: (float(x["action_cost"]), str(x["action_name"])))

        total_cost = sum(float(r["action_cost"]) for r in chosen_rows.values())
        upgrades: list[tuple[float, float, str, dict[str, Any]]] = []
        for pid, rows in by_prompt.items():
            base = chosen_rows[pid]
            for cand in rows:
                delta_cost = float(cand["action_cost"]) - float(base["action_cost"])
                delta_u = float(cand["final_utility"]) - float(base["final_utility"])
                if delta_cost > 0 and delta_u > 0:
                    ratio = delta_u / delta_cost
                    upgrades.append((ratio, delta_cost, pid, cand))
        upgrades.sort(key=lambda x: (-x[0], x[1], x[2], str(x[3]["action_name"])))
        for _, dcost, pid, cand in upgrades:
            if total_cost + dcost <= budget and float(cand["final_utility"]) > float(chosen_rows[pid]["final_utility"]):
                total_cost += dcost
                chosen_rows[pid] = cand

        chosen = {pid: str(r["action_name"]) for pid, r in chosen_rows.items()}
        obj = sum(float(r["final_utility"]) for r in chosen_rows.values())
        return {
            "chosen_by_prompt": chosen,
            "objective_value": obj,
            "total_cost": total_cost,
            "budget": budget,
            "optimizer_name": self.name,
        }

