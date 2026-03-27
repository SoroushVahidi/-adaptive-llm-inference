#!/usr/bin/env python3
"""Run diagnostic synthetic allocation sweeps over budgets and noise levels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.synthetic_ttc import generate_synthetic_ttc_instance
from src.evaluation.simulated_analysis import (
    flatten_budget_runs_for_csv,
    flatten_noise_runs_for_csv,
    resolve_budget_list,
    run_budget_sweep,
    run_noise_sensitivity_experiments,
)
from src.utils.config import load_config


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _build_metadata(config: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_type": "simulated_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "instance_metadata": instance["metadata"],
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    synthetic_cfg = config.get("synthetic", {})
    noise_cfg = config.get("noise", {})

    curve_family = str(
        synthetic_cfg.get(
            "curve_family",
            config.get("difficulty_mode", "mixed_difficulty"),
        )
    )
    instance = generate_synthetic_ttc_instance(
        n_queries=int(synthetic_cfg.get("n_queries", 120)),
        n_levels=int(synthetic_cfg.get("n_levels", 6)),
        curve_family=curve_family,
        costs=synthetic_cfg.get("costs"),
        seed=synthetic_cfg.get("seed"),
    )

    budgets = resolve_budget_list(
        configured_budgets=config.get("budgets"),
        budget_range=config.get("budget_range"),
    )
    allocator_names = [str(name) for name in config.get("allocators", ["equal", "mckp"])]
    output_dir = Path(
        config.get("output_dir", config.get("output_prefix", "outputs/simulated_sweep"))
    )
    difficulty_labels = instance.get("difficulty_labels")

    budget_sweep = run_budget_sweep(
        utility_table=instance["utility_table"],
        costs=instance["costs"],
        budgets=budgets,
        allocator_names=allocator_names,
        difficulty_labels=difficulty_labels,
    )

    noise_budget = int(noise_cfg.get("budget", budgets[-1]))
    noise_results = None
    if bool(noise_cfg.get("enabled", True)):
        noise_results = run_noise_sensitivity_experiments(
            utility_table=instance["utility_table"],
            costs=instance["costs"],
            budget=noise_budget,
            allocator_names=allocator_names,
            difficulty_labels=difficulty_labels,
            noise_levels=noise_cfg.get("levels"),
            seed=noise_cfg.get("seed", synthetic_cfg.get("seed")),
            preserve_monotonicity=bool(noise_cfg.get("preserve_monotonicity", True)),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **_build_metadata(config=config, instance=instance),
        "instance": instance,
        "budget_sweep": budget_sweep,
        "noise_sensitivity": noise_results,
    }

    json_path = output_dir / "simulated_sweep_results.json"
    json_path.write_text(json.dumps(payload, indent=2))
    _write_csv(
        rows=flatten_budget_runs_for_csv(budget_sweep["runs"]),
        output_path=output_dir / "budget_sweep_runs.csv",
    )
    _write_csv(
        rows=_flatten_comparisons(budget_sweep["comparisons"]),
        output_path=output_dir / "budget_sweep_comparisons.csv",
    )
    if noise_results is not None:
        _write_csv(
            rows=flatten_noise_runs_for_csv(noise_results["runs"]),
            output_path=output_dir / "noise_sensitivity_runs.csv",
        )
        _write_csv(
            rows=_flatten_comparisons(noise_results["comparisons"]),
            output_path=output_dir / "noise_sensitivity_comparisons.csv",
        )

    print("--- Simulated Sweep Results ---")
    print(f"queries:                 {len(instance['query_ids'])}")
    print(f"levels:                  {len(instance['costs'])}")
    print(f"difficulty_mode:         {curve_family}")
    print(f"budgets:                 {budgets}")
    print(f"allocators:              {allocator_names}")
    print(f"budget_results_json:     {json_path}")
    print(f"budget_results_csv:      {output_dir / 'budget_sweep_runs.csv'}")
    print(f"budget_compare_csv:      {output_dir / 'budget_sweep_comparisons.csv'}")
    if noise_results is not None:
        print(f"noise_budget:            {noise_budget}")
        print(f"noise_results_csv:       {output_dir / 'noise_sensitivity_runs.csv'}")
        print(
            f"noise_compare_csv:       {output_dir / 'noise_sensitivity_comparisons.csv'}"
        )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diagnostic synthetic allocation budget/noise sweeps"
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
