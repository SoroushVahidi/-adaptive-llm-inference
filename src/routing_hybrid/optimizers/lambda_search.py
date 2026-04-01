from __future__ import annotations

from typing import Any


class LambdaSearchOptimizer:
    """Per-prompt argmax with lambda search to satisfy global budget."""

    name = "lambda_search"

    def __init__(self, iters: int = 24) -> None:
        self.iters = iters

    def _solve_with_lambda(
        self,
        by_prompt: dict[str, list[dict[str, Any]]],
        lam: float,
    ) -> tuple[dict[str, str], float, float]:
        chosen: dict[str, str] = {}
        total_cost = 0.0
        total_obj = 0.0
        for pid, rows in by_prompt.items():
            best = max(
                rows,
                key=lambda x: (
                    float(x["final_utility"]) - lam * float(x["action_cost"]),
                    -float(x["action_cost"]),
                    str(x["action_name"]),
                ),
            )
            chosen[pid] = str(best["action_name"])
            total_cost += float(best["action_cost"])
            total_obj += float(best["final_utility"])
        return chosen, total_cost, total_obj

    def solve(self, candidate_rows: list[dict[str, Any]], budget: float) -> dict[str, Any]:
        by_prompt: dict[str, list[dict[str, Any]]] = {}
        for r in candidate_rows:
            by_prompt.setdefault(str(r["prompt_id"]), []).append(r)

        lo, hi = 0.0, 50.0
        best_feasible: tuple[dict[str, str], float, float] | None = None
        for _ in range(self.iters):
            mid = (lo + hi) / 2.0
            chosen, cost, obj = self._solve_with_lambda(by_prompt, lam=mid)
            if cost <= budget:
                best_feasible = (chosen, cost, obj)
                hi = mid
            else:
                lo = mid
        if best_feasible is None:
            chosen, cost, obj = self._solve_with_lambda(by_prompt, lam=hi)
        else:
            chosen, cost, obj = best_feasible
        return {
            "chosen_by_prompt": chosen,
            "objective_value": obj,
            "total_cost": cost,
            "budget": budget,
            "optimizer_name": self.name,
        }

