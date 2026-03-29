#!/usr/bin/env python3
"""Run oracle strategy evaluation on GSM8K with literature-informed strategies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.gsm8k import load_gsm8k
from src.evaluation.oracle_strategy_eval import (
    BASE_REQUIRED_STRATEGIES,
    OPTIONAL_STRATEGIES,
    run_oracle_strategy_eval,
    write_oracle_strategy_outputs,
)
from src.models.openai_llm import OpenAILLMModel
from src.utils.config import load_config


def _write_blocker_summary(output_dir: str, blocker: str, detail: str) -> str:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_status": "BLOCKED",
        "blocker_type": blocker,
        "blocker_detail": detail,
    }
    path = base / "summary.json"
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def _build_model(model_cfg: dict[str, object], default_name: str = "gpt-4o-mini") -> OpenAILLMModel:
    return OpenAILLMModel(
        model_name=str(model_cfg.get("name", default_name)),
        base_url=model_cfg.get("base_url"),
        greedy_temperature=float(model_cfg.get("greedy_temperature", 0.0)),
        sample_temperature=float(model_cfg.get("sample_temperature", 0.7)),
        max_tokens=int(model_cfg.get("max_tokens", 512)),
        timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run oracle strategy evaluation on GSM8K")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = str(config.get("output_dir", "outputs/oracle_strategy_eval"))

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

    model_cfg = config.get("model", {})
    strong_model_cfg = config.get("strong_model")

    print(f"Initialising base model '{model_cfg.get('name', 'gpt-4o-mini')}'...")
    try:
        model = _build_model(model_cfg)
    except ValueError as exc:
        msg = str(exc)
        print(f"BLOCKER: Cannot initialise OpenAI model.\n  Error: {msg}", file=sys.stderr)
        blocker_path = _write_blocker_summary(output_dir, "openai_api_key", msg)
        print(f"Blocker summary written to: {blocker_path}", file=sys.stderr)
        sys.exit(1)

    strong_model = None
    if strong_model_cfg:
        if isinstance(strong_model_cfg, str):
            strong_model_cfg = {"name": strong_model_cfg}
        print(f"Initialising strong model '{strong_model_cfg.get('name')}'...")
        try:
            strong_model = _build_model(strong_model_cfg, default_name=str(model_cfg.get("name")))
        except ValueError as exc:
            msg = str(exc)
            print(
                f"BLOCKER: Cannot initialise strong OpenAI model.\n  Error: {msg}",
                file=sys.stderr,
            )
            blocker_path = _write_blocker_summary(output_dir, "openai_api_key", msg)
            print(f"Blocker summary written to: {blocker_path}", file=sys.stderr)
            sys.exit(1)

    strategies = config.get("strategies")
    if strategies is None:
        strategies = list(BASE_REQUIRED_STRATEGIES)
        if strong_model is not None:
            strategies += OPTIONAL_STRATEGIES

    required_missing = [s for s in BASE_REQUIRED_STRATEGIES if s not in strategies]
    if required_missing:
        raise ValueError(f"Config strategies missing required entries: {required_missing}")

    lambda_penalty = float(config.get("lambda_penalty", 0.0))
    print(f"Strategies: {strategies}")
    print(f"Lambda penalty: {lambda_penalty}")

    try:
        result = run_oracle_strategy_eval(
            model=model,
            queries=queries,
            strategies=strategies,
            strong_model=strong_model,
            lambda_penalty=lambda_penalty,
        )
    except RuntimeError as exc:
        msg = str(exc)
        print(f"BLOCKER: Evaluation failed.\n  Error: {msg}", file=sys.stderr)
        blocker_path = _write_blocker_summary(output_dir, "openai_api_error", msg)
        print(f"Blocker summary written to: {blocker_path}", file=sys.stderr)
        sys.exit(1)

    paths = write_oracle_strategy_outputs(result, output_dir)

    summary = result["summary"]
    print("--- Oracle Strategy Evaluation ---")
    print(f"queries: {summary['total_queries']}")
    print(f"direct_accuracy: {summary['direct_accuracy']:.4f}")
    print(f"oracle_accuracy: {summary['oracle_accuracy']:.4f}")
    print(f"oracle_direct_gap: {summary['oracle_direct_gap']:.4f}")
    print(f"fraction_direct_optimal: {summary['fraction_direct_optimal']:.4f}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_csv: {paths['summary_csv']}")
    print(f"per_query_csv: {paths['per_query_csv']}")
    print(f"oracle_assignments_csv: {paths['oracle_assignments_csv']}")


if __name__ == "__main__":
    main()
