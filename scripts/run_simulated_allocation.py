#!/usr/bin/env python3
"""Run a simulated allocation experiment on synthetic utility tables."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.synthetic_ttc import generate_synthetic_ttc_instance
from src.evaluation.simulated_evaluator import evaluate_simulated_allocation
from src.utils.config import load_config


def _build_output_payload(
    config: dict[str, Any], instance: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "run_type": "simulated_allocation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "instance": instance,
        "evaluation": evaluation,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    synthetic_cfg = config.get("synthetic", {})
    allocator_cfg = config.get("allocator", {})
    budget = int(config["budget"])
    output_path = Path(config.get("output", "outputs/simulated_allocation_results.json"))

    instance = generate_synthetic_ttc_instance(
        n_queries=int(synthetic_cfg.get("n_queries", 100)),
        n_levels=int(synthetic_cfg.get("n_levels", 5)),
        curve_family=str(synthetic_cfg.get("curve_family", "mixed_difficulty")),
        costs=synthetic_cfg.get("costs"),
        seed=synthetic_cfg.get("seed"),
    )

    allocator_name = str(allocator_cfg.get("name", "equal"))
    evaluation = evaluate_simulated_allocation(
        utility_table=instance["utility_table"],
        costs=instance["costs"],
        budget=budget,
        allocator_name=allocator_name,
    )

    print("--- Simulated Allocation Results ---")
    print(f"allocator:             {allocator_name}")
    print(f"queries:               {len(instance['query_ids'])}")
    print(f"levels:                {len(instance['costs'])}")
    print(f"budget:                {budget}")
    print(f"total_expected_utility:{evaluation['total_expected_utility']:.6f}")
    print(f"total_cost:            {evaluation['total_cost']}")
    print(f"avg_utility/query:     {evaluation['average_utility_per_query']:.6f}")
    print(f"avg_cost/query:        {evaluation['average_cost_per_query']:.6f}")

    output_payload = _build_output_payload(config=config, instance=instance, evaluation=evaluation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2))
    print(f"saved_results:         {output_path}")

    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run synthetic utility-table allocation experiment"
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
