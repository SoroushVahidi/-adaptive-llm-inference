#!/usr/bin/env python3
"""Run the strategy expansion experiment on GSM8K.

Usage:
    python3 scripts/run_strategy_expansion.py --config configs/strategy_expansion_gsm8k.yaml

Requires OPENAI_API_KEY to be set in the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.gsm8k import load_gsm8k
from src.evaluation.strategy_expansion_eval import (
    ALL_STRATEGIES,
    format_strategy_expansion_summary,
    run_strategy_expansion_eval,
    write_strategy_expansion_outputs,
)
from src.models.openai_llm import OpenAILLMModel
from src.utils.config import load_config


def _write_blocker_summary(output_dir: str, blocker: str, detail: str) -> str:
    """Write a summary.json documenting why the run was blocked."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_status": "BLOCKED",
        "blocker_type": blocker,
        "blocker_detail": detail,
        "strategy_summaries": {},
        "pairwise_comparisons": [],
        "total_queries": 0,
        "strategies_run": [],
    }
    path = base / "summary.json"
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strategy expansion evaluation on GSM8K"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. configs/strategy_expansion_gsm8k.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "outputs/strategy_expansion")

    # --- Dataset ---
    dataset_cfg = config.get("dataset", {})
    data_file = dataset_cfg.get("data_file")
    source = f"local file {data_file}" if data_file else "HuggingFace"
    print(
        f"Loading GSM8K ({dataset_cfg.get('split', 'test')} split, "
        f"max {dataset_cfg.get('max_samples', 20)} queries from {source})..."
    )
    try:
        queries = load_gsm8k(
            split=str(dataset_cfg.get("split", "test")),
            max_samples=int(dataset_cfg.get("max_samples", 20)),
            data_file=data_file,
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        print(f"BLOCKER: Failed to load GSM8K dataset.\n  Error: {msg}", file=sys.stderr)
        blocker_path = _write_blocker_summary(output_dir, "dataset_access", msg)
        print(f"Blocker summary written to: {blocker_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(queries)} queries.")

    # --- Model ---
    model_cfg = config.get("model", {})
    print(f"Initialising OpenAI model '{model_cfg.get('name', 'gpt-4o-mini')}'...")
    try:
        model = OpenAILLMModel(
            model_name=str(model_cfg["name"]),
            base_url=model_cfg.get("base_url"),
            greedy_temperature=float(model_cfg.get("greedy_temperature", 0.0)),
            sample_temperature=float(model_cfg.get("sample_temperature", 0.7)),
            max_tokens=int(model_cfg.get("max_tokens", 256)),
            timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
        )
    except ValueError as exc:
        msg = str(exc)
        print(f"BLOCKER: Cannot initialise OpenAI model.\n  Error: {msg}", file=sys.stderr)
        blocker_path = _write_blocker_summary(output_dir, "openai_api_key", msg)
        print(f"Blocker summary written to: {blocker_path}", file=sys.stderr)
        sys.exit(1)

    # --- Strategies ---
    strategies: list[str] = config.get("strategies", ALL_STRATEGIES)
    print(f"Strategies: {strategies}\n")

    # --- Run evaluation ---
    print("Running evaluation...")
    try:
        result = run_strategy_expansion_eval(
            model=model,
            queries=queries,
            strategies=strategies,
        )
    except RuntimeError as exc:
        msg = str(exc)
        print(f"BLOCKER: Evaluation failed.\n  Error: {msg}", file=sys.stderr)
        blocker_path = _write_blocker_summary(output_dir, "openai_api_error", msg)
        print(f"Blocker summary written to: {blocker_path}", file=sys.stderr)
        sys.exit(1)

    # --- Save outputs ---
    paths = write_strategy_expansion_outputs(result, output_dir)

    # --- Print summary ---
    print(format_strategy_expansion_summary(result, paths))


if __name__ == "__main__":
    main()
