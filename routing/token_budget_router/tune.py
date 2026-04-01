from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.policies.token_budget_router import (
    TokenBudgetRouterConfig,
    TokenBudgetRouterPolicy,
    build_threshold_grid,
    evaluate_router,
    quantile_threshold_grid,
    select_operating_point,
)
from src.utils.config import load_config


def tune(config: dict[str, Any]) -> dict[str, Any]:
    tune_cfg = config.get("tuning", {})
    dataset_path = Path(tune_cfg["validation_dataset"])
    df = pd.read_csv(dataset_path)

    base_cfg = TokenBudgetRouterConfig(
        length_field=str(config.get("length_field", "fp_first_pass_output_length")),
        question_length_field=str(config.get("question_length_field", "q_question_length_tokens_approx")),
        feature_mode=str(config.get("feature_mode", "raw")),
    )
    probe_policy = TokenBudgetRouterPolicy(base_cfg)
    feat_values = probe_policy.feature_series(df)

    zmean = float(feat_values.mean())
    zstd = float(feat_values.std(ddof=0)) if len(feat_values) else 1.0

    min_grid = tune_cfg.get("min_threshold_grid")
    max_grid = tune_cfg.get("max_threshold_grid")
    if min_grid is None or max_grid is None:
        quantiles = tune_cfg.get("quantile_grid", [0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95])
        grid = quantile_threshold_grid(feat_values, quantiles=quantiles, include_one_sided=True)
    else:
        grid = build_threshold_grid(min_grid=min_grid, max_grid=max_grid)

    rows: list[dict[str, Any]] = []
    for g in grid:
        cfg = TokenBudgetRouterConfig(
            length_field=base_cfg.length_field,
            question_length_field=base_cfg.question_length_field,
            feature_mode=base_cfg.feature_mode,
            min_len_threshold=g["min_len_threshold"],
            max_len_threshold=g["max_len_threshold"],
            zscore_mean=zmean if base_cfg.feature_mode == "zscore" else None,
            zscore_std=zstd if base_cfg.feature_mode == "zscore" else None,
        )
        stats = evaluate_router(df, TokenBudgetRouterPolicy(cfg))
        rows.append({**g, **stats})

    selected = select_operating_point(
        rows,
        target_revise_rate=tune_cfg.get("target_revise_rate"),
        max_avg_cost=tune_cfg.get("max_avg_cost"),
    )

    out = {
        "policy_name": "token_budget_router",
        "feature_mode": base_cfg.feature_mode,
        "length_field": base_cfg.length_field,
        "question_length_field": base_cfg.question_length_field,
        "selection_mode": tune_cfg.get("selection_mode", "target_revise_rate"),
        "selected_thresholds": {
            "min_len_threshold": selected["min_len_threshold"],
            "max_len_threshold": selected["max_len_threshold"],
        },
        "feature_stats": {
            "mean": zmean,
            "std": zstd,
        },
        "validation_metrics": {
            k: selected[k]
            for k in ["accuracy", "avg_cost", "revise_rate", "oracle_accuracy", "oracle_gap", "n"]
        },
        "candidate_results": sorted(rows, key=lambda r: (r["avg_cost"], -r["accuracy"])),
    }

    output_path = Path(config.get("tuned_params_output", "outputs/token_budget_router/tuned_thresholds.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Tune token-budget router thresholds")
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    result = tune(cfg)
    print(json.dumps({"selected_thresholds": result["selected_thresholds"], "validation": result["validation_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
