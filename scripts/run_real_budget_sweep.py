#!/usr/bin/env python3
"""Run a lightweight real-data budget sweep on GSM8K."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.real_budget_analysis import (
    flatten_real_budget_comparisons_for_csv,
    flatten_real_budget_runs_for_csv,
    run_real_budget_sweep,
    write_csv_rows,
)
from src.utils.config import load_config


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config.get("output_dir", "outputs/real_budget_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_real_budget_sweep(config)
    per_run_csv = write_csv_rows(
        flatten_real_budget_runs_for_csv(result["runs"]),
        output_dir / "real_budget_runs.csv",
    )
    comparison_csv = write_csv_rows(
        flatten_real_budget_comparisons_for_csv(result["comparisons"]),
        output_dir / "real_budget_comparisons.csv",
    )
    payload = {
        "run_type": "real_budget_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "paths": {
            "per_run_csv": str(per_run_csv),
            "comparison_csv": str(comparison_csv),
        },
        **result,
    }

    json_path = output_dir / "real_budget_sweep_results.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print("--- Real Budget Sweep Results ---")
    print(f"dataset:                   {payload['dataset']}")
    print(f"model:                     {payload['model_type']}")
    print(f"budgets:                   {payload['budgets']}")
    print(f"baselines:                 {payload['available_baselines']}")
    if payload["unavailable_baselines"]:
        print(f"unavailable_baselines:     {payload['unavailable_baselines']}")
    print(
        "per_run_csv:               "
        f"{payload['paths']['per_run_csv']}"
    )
    print(
        "comparison_csv:            "
        f"{payload['paths']['comparison_csv']}"
    )
    print(f"summary_json:              {json_path}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GSM8K real-data budget sweep")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
