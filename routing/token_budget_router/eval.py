from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.policies.token_budget_router import (
    DPR_ROUTE,
    RG_ROUTE,
    TokenBudgetRouterConfig,
    TokenBudgetRouterPolicy,
    build_threshold_grid,
    evaluate_router,
)
from src.utils.config import load_config


def _load_tuned(config: dict[str, Any]) -> dict[str, Any]:
    tuned_path = Path(config.get("tuned_params_output", "outputs/token_budget_router/tuned_thresholds.json"))
    if tuned_path.exists():
        return json.loads(tuned_path.read_text())
    return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _oracle_rows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rg = pd.to_numeric(df["reasoning_correct"], errors="coerce").fillna(0).astype(int).to_numpy()
    dpr = pd.to_numeric(df["revise_correct"], errors="coerce").fillna(0).astype(int).to_numpy()

    if "revise_helpful" in df.columns:
        revise = pd.to_numeric(df["revise_helpful"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        revise = ((rg == 0) & (dpr == 1)).astype(int)
    correct = np.where(revise == 1, dpr, rg)
    return revise, correct


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    regimes = config["evaluation"]["regimes"]
    tuned = _load_tuned(config)

    selected = tuned.get("selected_thresholds", config.get("selected_thresholds", {}))
    feat_stats = tuned.get("feature_stats", {})

    output_dir = Path(config.get("evaluation_output_dir", "outputs/token_budget_router"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for regime, dataset_path in regimes.items():
        df = pd.read_csv(dataset_path)
        router_cfg = TokenBudgetRouterConfig(
            length_field=str(config.get("length_field", "fp_first_pass_output_length")),
            question_length_field=str(config.get("question_length_field", "q_question_length_tokens_approx")),
            feature_mode=str(config.get("feature_mode", tuned.get("feature_mode", "raw"))),
            min_len_threshold=selected.get("min_len_threshold"),
            max_len_threshold=selected.get("max_len_threshold"),
            zscore_mean=feat_stats.get("mean"),
            zscore_std=feat_stats.get("std"),
        )
        policy = TokenBudgetRouterPolicy(router_cfg)
        chosen = policy.decide_batch(df)
        stats = evaluate_router(df, policy)

        rg = pd.to_numeric(df["reasoning_correct"], errors="coerce").fillna(0).astype(int)
        dpr = pd.to_numeric(df["revise_correct"], errors="coerce").fillna(0).astype(int)
        revise_mask = (chosen == DPR_ROUTE).to_numpy()
        pred_correct = np.where(revise_mask, dpr.to_numpy(), rg.to_numpy())

        rg_acc = float(rg.mean())
        dpr_acc = float(dpr.mean())
        oracle_revise, oracle_correct = _oracle_rows(df)
        oracle_acc = float(np.mean(oracle_correct))
        oracle_cost = float(1.0 + np.mean(oracle_revise))

        comp_rows = [
            {"route": RG_ROUTE, "accuracy": rg_acc, "avg_cost": 1.0, "revise_rate": 0.0},
            {"route": DPR_ROUTE, "accuracy": dpr_acc, "avg_cost": 2.0, "revise_rate": 1.0},
            {
                "route": "token_budget_router",
                "accuracy": stats["accuracy"],
                "avg_cost": stats["avg_cost"],
                "revise_rate": stats["revise_rate"],
            },
            {
                "route": "oracle",
                "accuracy": oracle_acc,
                "avg_cost": oracle_cost,
                "revise_rate": float(np.mean(oracle_revise)),
            },
        ]
        _write_csv(output_dir / regime / "policy_comparison.csv", comp_rows)

        length_series = policy.feature_series(df)
        per_query_rows: list[dict[str, Any]] = []
        for i, (_, row) in enumerate(df.iterrows()):
            revise = bool(revise_mask[i])
            per_query_rows.append(
                {
                    "question_id": row.get("question_id", i),
                    "reasoning_correct": int(rg.iloc[i]),
                    "revise_correct": int(dpr.iloc[i]),
                    "revise_helpful": int(row.get("revise_helpful", 0)),
                    "policy_token_budget": DPR_ROUTE if revise else RG_ROUTE,
                    "correct_if_token_budget": int(pred_correct[i]),
                    "cost_token_budget": float(row.get("revise_cost", 2.0) if revise else row.get("reasoning_cost", 1.0)),
                    "length_feature_value": float(length_series.iloc[i]),
                }
            )
        _write_csv(output_dir / regime / "per_query_policy_decisions.csv", per_query_rows)

        sweep_cfg = config.get("sweep", {})
        min_grid = sweep_cfg.get("min_threshold_grid", [None])
        max_grid = sweep_cfg.get("max_threshold_grid", [None])
        curve_rows: list[dict[str, Any]] = []
        for g in build_threshold_grid(min_grid=min_grid, max_grid=max_grid):
            cfg = TokenBudgetRouterConfig(
                length_field=router_cfg.length_field,
                question_length_field=router_cfg.question_length_field,
                feature_mode=router_cfg.feature_mode,
                min_len_threshold=g["min_len_threshold"],
                max_len_threshold=g["max_len_threshold"],
                zscore_mean=router_cfg.zscore_mean,
                zscore_std=router_cfg.zscore_std,
            )
            curve_stats = evaluate_router(df, TokenBudgetRouterPolicy(cfg))
            curve_rows.append(
                {
                    "regime": regime,
                    "min_len_threshold": g["min_len_threshold"],
                    "max_len_threshold": g["max_len_threshold"],
                    "accuracy": curve_stats["accuracy"],
                    "avg_cost": curve_stats["avg_cost"],
                    "revise_rate": curve_stats["revise_rate"],
                    "oracle_gap": curve_stats["oracle_gap"],
                    "n": curve_stats["n"],
                }
            )
        _write_csv(output_dir / "budget_curves" / f"{regime}_token_budget_curve.csv", curve_rows)

        summary_rows.append(
            {
                "regime": regime,
                "policy": "token_budget_router",
                "accuracy": stats["accuracy"],
                "avg_cost": stats["avg_cost"],
                "revise_rate": stats["revise_rate"],
                "oracle_gap": oracle_acc - stats["accuracy"],
                "n": stats["n"],
                "min_len_threshold": selected.get("min_len_threshold"),
                "max_len_threshold": selected.get("max_len_threshold"),
            }
        )

    _write_csv(output_dir / "token_budget_router_summary.csv", summary_rows)
    payload = {"policy": "token_budget_router", "regime_results": summary_rows}
    (output_dir / "token_budget_router_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate token-budget router on manuscript regimes")
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    print(json.dumps(evaluate(cfg), indent=2))


if __name__ == "__main__":
    main()
