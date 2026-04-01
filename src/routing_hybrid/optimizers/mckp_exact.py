from __future__ import annotations

from typing import Any


class MCKPExactOptimizer:
    name = "mckp_exact"

    def __init__(self, cost_scale: int = 100) -> None:
        self.cost_scale = cost_scale

    def solve(self, candidate_rows: list[dict[str, Any]], budget: float) -> dict[str, Any]:
        by_prompt: dict[str, list[dict[str, Any]]] = {}
        for r in candidate_rows:
            by_prompt.setdefault(str(r["prompt_id"]), []).append(r)
        prompts = sorted(by_prompt.keys())
        int_budget = max(0, int(round(budget * self.cost_scale)))

        # dp[i][b] = (utility, backpointer_budget, chosen_action_name)
        dp: list[dict[int, tuple[float, int, str]]] = [dict() for _ in range(len(prompts) + 1)]
        dp[0][0] = (0.0, -1, "")
        for i, pid in enumerate(prompts, start=1):
            prev = dp[i - 1]
            cur: dict[int, tuple[float, int, str]] = {}
            for b_prev, (u_prev, _, _) in prev.items():
                for cand in by_prompt[pid]:
                    c = max(0, int(round(float(cand["action_cost"]) * self.cost_scale)))
                    b = b_prev + c
                    if b > int_budget:
                        continue
                    u = u_prev + float(cand["final_utility"])
                    old = cur.get(b)
                    action_name = str(cand["action_name"])
                    if old is None or u > old[0] or (u == old[0] and action_name < old[2]):
                        cur[b] = (u, b_prev, action_name)
            dp[i] = cur

        if not dp[len(prompts)]:
            raise RuntimeError("No feasible solution found for MCKP exact optimizer.")
        best_b = max(dp[len(prompts)].keys(), key=lambda b: (dp[len(prompts)][b][0], -b))
        best_u = dp[len(prompts)][best_b][0]

        chosen: dict[str, str] = {}
        b = best_b
        for i in range(len(prompts), 0, -1):
            pid = prompts[i - 1]
            u, b_prev, action = dp[i][b]
            chosen[pid] = action
            b = b_prev
        total_cost = float(best_b) / float(self.cost_scale)
        return {
            "chosen_by_prompt": chosen,
            "objective_value": float(best_u),
            "total_cost": total_cost,
            "budget": budget,
            "optimizer_name": self.name,
            "cost_scale": self.cost_scale,
        }

