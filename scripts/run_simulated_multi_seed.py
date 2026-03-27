#!/usr/bin/env python3
"""Run simulated sweep diagnostics across multiple synthetic seeds."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_simulated_sweep import run as run_simulated_sweep
from src.evaluation.simulated_aggregate import (
    aggregate_multi_seed_results,
)
from src.utils.config import load_config


def _resolve_seed_list(config: dict[str, Any]) -> list[int]:
    if config.get("seeds") is not None:
        seeds = [int(seed) for seed in config["seeds"]]
        if not seeds:
            raise ValueError("seeds must contain at least one value")
        return seeds

    seed_count = int(config.get("n_seeds", 0))
    start_seed = int(config.get("seed_start", 0))
    if seed_count <= 0:
        raise ValueError("Provide either a non-empty seeds list or n_seeds > 0")
    return list(range(start_seed, start_seed + seed_count))


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def _clone_config_for_seed(
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    seed_config = copy.deepcopy(config)
    synthetic_cfg = seed_config.setdefault("synthetic", {})
    noise_cfg = seed_config.setdefault("noise", {})

    synthetic_cfg["seed"] = int(seed)
    if noise_cfg.get("enabled", True):
        noise_cfg["seed"] = int(noise_cfg.get("seed_offset", 1000)) + int(seed)

    seed_config["output_dir"] = str(output_dir)
    return seed_config


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config.get("output_dir", "outputs/simulated_multi_seed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = _resolve_seed_list(config)
    seed_results: list[dict[str, Any]] = []
    seed_runs: list[dict[str, Any]] = []

    for seed in seeds:
        seed_output_dir = output_dir / f"seed_{seed}"
        seed_config = _clone_config_for_seed(config=config, seed=seed, output_dir=seed_output_dir)
        seed_result = run_simulated_sweep(seed_config)
        noise_payload = seed_result.get("noise_sensitivity")
        seed_results.append(
            {
                "seed": seed,
                "budget_comparisons": seed_result["budget_sweep"]["comparisons"],
                "noise_comparisons": [] if noise_payload is None else noise_payload["comparisons"],
            }
        )
        seed_runs.append(
            {
                "seed": seed,
                "output_dir": str(seed_output_dir),
                "instance_metadata": seed_result["instance"]["metadata"],
            }
        )

    aggregate_results = aggregate_multi_seed_results(
        seed_results=seed_results,
        output_dir=output_dir,
        small_gap_threshold=float(config.get("small_gap_threshold", 0.5)),
    )

    payload = {
        "run_type": "simulated_multi_seed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "seeds": seeds,
        "seed_runs": seed_runs,
        **aggregate_results,
    }
    _write_json(payload, output_dir / "aggregated_summary.json")

    print("--- Simulated Multi-Seed Results ---")
    print(f"seeds:                     {seeds}")
    print(
        "per_seed_budget_csv:       "
        f"{aggregate_results['paths']['per_seed_budget_runs']}"
    )
    print(
        "per_seed_noise_csv:        "
        f"{aggregate_results['paths']['per_seed_noise_runs']}"
    )
    print(
        "budget_summary_csv:        "
        f"{aggregate_results['paths']['aggregated_budget_summary']}"
    )
    print(
        "noise_summary_csv:         "
        f"{aggregate_results['paths']['aggregated_noise_summary']}"
    )
    print(f"summary_json:              {output_dir / 'aggregated_summary.json'}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simulated allocation sweeps across multiple seeds"
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
