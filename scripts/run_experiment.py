#!/usr/bin/env python3
"""Run an experiment: load data, apply a baseline/allocator, evaluate, and save results.

Usage:
    python scripts/run_experiment.py --config configs/greedy.yaml
    python scripts/run_experiment.py --config configs/best_of_n.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.allocators.equal import EqualAllocator
from src.baselines.best_of_n import BestOfNBaseline
from src.baselines.greedy import GreedyBaseline
from src.baselines.self_consistency import SelfConsistencyBaseline
from src.datasets.gsm8k import load_gsm8k
from src.evaluation.logger import ExperimentLogger
from src.models.dummy import DummyModel
from src.utils.config import load_config

BASELINES = {
    "greedy": GreedyBaseline,
    "best_of_n": BestOfNBaseline,
    "self_consistency": SelfConsistencyBaseline,
}


def run(config: dict) -> None:
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    baseline_name = config.get("baseline", "greedy")
    budget = config.get("budget", 100)
    n_samples = config.get("n_samples", 1)
    output_path = config.get("output", "outputs/results.json")

    # --- dataset ---
    dataset_name = dataset_cfg.get("name", "gsm8k")
    split = dataset_cfg.get("split", "test")
    max_samples = dataset_cfg.get("max_samples")
    print(f"Loading dataset: {dataset_name} (split={split}, max_samples={max_samples})")

    if dataset_name == "gsm8k":
        queries = load_gsm8k(split=split, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  Loaded {len(queries)} queries")

    # --- model ---
    model_type = model_cfg.get("type", "dummy")
    if model_type == "dummy":
        model = DummyModel(
            correct_prob=model_cfg.get("correct_prob", 0.3),
            seed=model_cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # --- baseline ---
    if baseline_name not in BASELINES:
        raise ValueError(f"Unknown baseline: {baseline_name}. Choose from {list(BASELINES)}")
    baseline = BASELINES[baseline_name](model)
    print(f"Baseline: {baseline.name}")

    # --- allocation ---
    allocator = EqualAllocator()
    if baseline_name == "greedy":
        per_query_samples = [1] * len(queries)
    else:
        per_query_samples = allocator.allocate(len(queries), budget)
    total_budget = sum(per_query_samples)
    print(f"Budget: {total_budget} total samples across {len(queries)} queries")

    # --- run ---
    logger = ExperimentLogger()
    for query, n in zip(queries, per_query_samples):
        samples = max(n, n_samples) if baseline_name != "greedy" else 1
        if isinstance(model, DummyModel):
            model.set_ground_truth(query.answer)
        result = baseline.solve(query.id, query.question, query.answer, samples)
        logger.log(result)

    # --- report ---
    summary = logger.summary()
    print("\n--- Results ---")
    print(f"  Accuracy:            {summary['accuracy']:.4f}")
    print(f"  Total samples used:  {summary['total_samples']}")
    print(f"  Total queries:       {summary['total_queries']}")
    print(f"  Avg samples/query:   {summary['avg_samples_per_query']:.2f}")

    # --- save ---
    logger.save(output_path)
    print(f"\nResults saved to {output_path}")

    # Also save a compact summary next to the full log
    summary_path = Path(output_path).with_suffix(".summary.json")
    summary["config"] = config
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an adaptive inference experiment")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
