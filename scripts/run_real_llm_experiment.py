#!/usr/bin/env python3
"""Run a minimal GSM8K experiment with a real OpenAI-style LLM backend."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.best_of_n import BestOfNBaseline
from src.baselines.greedy import GreedyBaseline
from src.datasets.gsm8k import load_gsm8k
from src.evaluation.logger import ExperimentLogger
from src.models.llm_model import OpenAICompatibleLLMModel
from src.utils.config import load_config

BASELINES = {
    "greedy": GreedyBaseline,
    "best_of_n": BestOfNBaseline,
}


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    experiment_cfg = config.get("experiment", {})
    output_dir = Path(config.get("output_dir", "outputs/real_llm"))
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_gsm8k(
        split=str(dataset_cfg.get("split", "test")),
        max_samples=dataset_cfg.get("max_samples"),
    )
    provider = str(model_cfg.get("provider", "openai"))
    if provider != "openai":
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            "This validation script currently supports only an OpenAI-style API."
        )

    model = OpenAICompatibleLLMModel(
        model_name=str(model_cfg["name"]),
        base_url=model_cfg.get("base_url"),
        system_prompt=model_cfg.get("system_prompt"),
        greedy_temperature=float(model_cfg.get("greedy_temperature", 0.0)),
        sample_temperature=float(model_cfg.get("sample_temperature", 0.7)),
        max_tokens=int(model_cfg.get("max_tokens", 256)),
        timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
    )

    baseline_specs = experiment_cfg.get(
        "baselines",
        [
            {"name": "greedy", "n_samples": 1},
            {"name": "best_of_n", "n_samples": 3},
        ],
    )

    runs: list[dict[str, Any]] = []
    for spec in baseline_specs:
        baseline_name = str(spec["name"])
        if baseline_name not in BASELINES:
            raise ValueError(
                f"Unsupported baseline '{baseline_name}'. Supported: {sorted(BASELINES)}"
            )

        n_samples = int(spec.get("n_samples", 1))
        baseline = BASELINES[baseline_name](model)
        logger = ExperimentLogger()
        for query in queries:
            result = baseline.solve(
                query_id=query.id,
                question=query.question,
                ground_truth=query.answer,
                n_samples=n_samples,
            )
            logger.log(result)

        summary = logger.summary()
        run_payload = {
            "baseline": baseline_name,
            "n_samples": n_samples,
            "accuracy": float(summary["accuracy"]),
            "total_samples_used": int(summary["total_samples"]),
            "total_queries": int(summary["total_queries"]),
            "avg_samples_per_query": float(summary["avg_samples_per_query"]),
        }
        runs.append(run_payload)

    payload = {
        "run_type": "real_llm_experiment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "provider": provider,
        "model_name": model.model_name,
        "total_queries": len(queries),
        "runs": runs,
    }
    _write_json(payload, output_dir / "real_llm_results.json")

    print("--- Real LLM Experiment Results ---")
    print(f"provider:                  {payload['provider']}")
    print(f"model:                     {payload['model_name']}")
    print(f"queries:                   {payload['total_queries']}")
    for run_payload in runs:
        print(
            f"{run_payload['baseline']}: "
            f"accuracy={run_payload['accuracy']:.4f}, "
            f"total_samples={run_payload['total_samples_used']}"
        )
    print(f"summary_json:              {output_dir / 'real_llm_results.json'}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal GSM8K experiment with a real LLM backend"
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
