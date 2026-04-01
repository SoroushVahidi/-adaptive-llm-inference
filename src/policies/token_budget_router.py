"""Token-budget / length-based routing baseline.

Compute-only router between:
- reasoning_greedy (RG, cheap)
- direct_plus_revise (DPR, expensive)

No semantic error signals are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

RG_ROUTE = "reasoning_greedy"
DPR_ROUTE = "direct_plus_revise"


@dataclass(frozen=True)
class TokenBudgetRouterConfig:
    policy_name: str = "token_budget_router"
    length_field: str = "fp_first_pass_output_length"
    question_length_field: str = "q_question_length_tokens_approx"
    feature_mode: str = "raw"  # raw | ratio_question_tokens | zscore
    min_len_threshold: float | None = None
    max_len_threshold: float | None = None
    zscore_mean: float | None = None
    zscore_std: float | None = None


class TokenBudgetRouterPolicy:
    """Interpretable threshold-band policy on a single compute-side feature."""

    def __init__(self, config: TokenBudgetRouterConfig) -> None:
        self.config = config

    def _feature_value(self, row: dict[str, Any]) -> float:
        def _to_float(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        rg_len = _to_float(row.get(self.config.length_field, 0.0))
        mode = self.config.feature_mode

        if mode == "raw":
            return rg_len
        if mode == "ratio_question_tokens":
            q_len = _to_float(row.get(self.config.question_length_field, 0.0))
            return rg_len / max(q_len, 1.0)
        if mode == "zscore":
            mean = float(self.config.zscore_mean or 0.0)
            std = float(self.config.zscore_std or 1.0)
            std = 1.0 if abs(std) < 1e-9 else std
            return (rg_len - mean) / std

        raise ValueError(f"Unsupported feature_mode: {mode!r}")

    def decide(self, row: dict[str, Any]) -> str:
        x = self._feature_value(row)
        min_t = self.config.min_len_threshold
        max_t = self.config.max_len_threshold

        escalate = False
        if min_t is not None and x <= min_t:
            escalate = True
        if max_t is not None and x >= max_t:
            escalate = True
        return DPR_ROUTE if escalate else RG_ROUTE

    def feature_series(self, df: pd.DataFrame) -> pd.Series:
        required = [self.config.length_field]
        if self.config.feature_mode == "ratio_question_tokens":
            required.append(self.config.question_length_field)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        rg = pd.to_numeric(df[self.config.length_field], errors="coerce").fillna(0.0)
        if self.config.feature_mode == "raw":
            return rg

        if self.config.feature_mode == "ratio_question_tokens":
            q = pd.to_numeric(df[self.config.question_length_field], errors="coerce").fillna(0.0)
            return rg / q.clip(lower=1.0)

        mean = float(self.config.zscore_mean or rg.mean())
        std = float(self.config.zscore_std or rg.std(ddof=0))
        std = 1.0 if abs(std) < 1e-9 else std
        return (rg - mean) / std

    def decide_batch(self, df: pd.DataFrame) -> pd.Series:
        x = self.feature_series(df)
        revise = pd.Series(False, index=df.index)
        if self.config.min_len_threshold is not None:
            revise = revise | (x <= float(self.config.min_len_threshold))
        if self.config.max_len_threshold is not None:
            revise = revise | (x >= float(self.config.max_len_threshold))
        return revise.map({True: DPR_ROUTE, False: RG_ROUTE})


def evaluate_router(df: pd.DataFrame, policy: TokenBudgetRouterPolicy) -> dict[str, float]:
    required = ["reasoning_correct", "revise_correct", policy.config.length_field]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    chosen = policy.decide_batch(df)
    rg = pd.to_numeric(df["reasoning_correct"], errors="coerce").fillna(0).astype(int)
    dpr = pd.to_numeric(df["revise_correct"], errors="coerce").fillna(0).astype(int)

    revise_mask = chosen == DPR_ROUTE
    chosen_correct = np.where(revise_mask.to_numpy(), dpr.to_numpy(), rg.to_numpy())

    if {"reasoning_cost", "revise_cost"}.issubset(df.columns):
        rg_cost = pd.to_numeric(df["reasoning_cost"], errors="coerce").fillna(1.0)
        dpr_cost = pd.to_numeric(df["revise_cost"], errors="coerce").fillna(2.0)
        chosen_cost = np.where(revise_mask.to_numpy(), dpr_cost.to_numpy(), rg_cost.to_numpy())
        avg_cost = float(np.mean(chosen_cost))
    else:
        avg_cost = float(1.0 + np.mean(revise_mask.to_numpy()))

    oracle = np.maximum(rg.to_numpy(), dpr.to_numpy())

    return {
        "accuracy": float(np.mean(chosen_correct)),
        "avg_cost": avg_cost,
        "revise_rate": float(np.mean(revise_mask.to_numpy())),
        "oracle_accuracy": float(np.mean(oracle)),
        "oracle_gap": float(np.mean(oracle) - np.mean(chosen_correct)),
        "n": int(len(df)),
    }


def build_threshold_grid(
    min_grid: list[float | None],
    max_grid: list[float | None],
) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    for min_t, max_t in product(min_grid, max_grid):
        if min_t is not None and max_t is not None and min_t > max_t:
            continue
        rows.append({"min_len_threshold": min_t, "max_len_threshold": max_t})
    return rows


def quantile_threshold_grid(
    values: pd.Series,
    quantiles: list[float],
    include_one_sided: bool = True,
) -> list[dict[str, float | None]]:
    qs = sorted(set(float(q) for q in quantiles if 0.0 <= float(q) <= 1.0))
    if not qs:
        raise ValueError("quantiles must not be empty")
    cutoffs = [float(values.quantile(q)) for q in qs]

    min_grid: list[float | None] = [None]
    max_grid: list[float | None] = [None]
    if include_one_sided:
        min_grid.extend(cutoffs)
        max_grid.extend(cutoffs)
    else:
        min_grid = cutoffs
        max_grid = cutoffs
    return build_threshold_grid(min_grid=min_grid, max_grid=max_grid)


def select_operating_point(
    candidates: list[dict[str, Any]],
    *,
    target_revise_rate: float | None = None,
    max_avg_cost: float | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No candidates to select from")

    if target_revise_rate is not None:
        return sorted(
            candidates,
            key=lambda r: (
                abs(float(r["revise_rate"]) - float(target_revise_rate)),
                -float(r["accuracy"]),
                float(r["avg_cost"]),
            ),
        )[0]

    if max_avg_cost is not None:
        feasible = [r for r in candidates if float(r["avg_cost"]) <= float(max_avg_cost)]
        if feasible:
            return sorted(feasible, key=lambda r: (-float(r["accuracy"]), float(r["avg_cost"])))[0]

    return sorted(candidates, key=lambda r: (-float(r["accuracy"]), float(r["avg_cost"])))[0]
