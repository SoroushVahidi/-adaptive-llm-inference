"""Helpers for budget sweeps on the current real-data evaluation path.

This module intentionally stays close to the existing GSM8K + native-baseline
pipeline. Its role is to make budget-based comparisons easier to inspect
without changing core experiment logic.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.baselines.base import Baseline
from src.baselines.best_of_n import BestOfNBaseline
from src.baselines.greedy import GreedyBaseline
from src.baselines.self_consistency import SelfConsistencyBaseline
from src.datasets.gsm8k import Query, load_gsm8k
from src.evaluation.logger import ExperimentLogger
from src.models.base import Model
from src.models.dummy import DummyModel

BASELINES: dict[str, type[Baseline]] = {
    "greedy": GreedyBaseline,
    "best_of_n": BestOfNBaseline,
    "self_consistency": SelfConsistencyBaseline,
}


def available_baselines(requested: list[str]) -> tuple[list[str], list[str]]:
    """Split requested baselines into runnable and unsupported names."""
    runnable: list[str] = []
    unsupported: list[str] = []
    for name in requested:
        normalized = str(name)
        if normalized in BASELINES:
            runnable.append(normalized)
        else:
            unsupported.append(normalized)
    return runnable, unsupported


def load_real_queries(config: dict[str, Any]) -> list[Query]:
    """Load the requested public evaluation dataset."""
    dataset_cfg = config.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "gsm8k"))
    split = str(dataset_cfg.get("split", "test"))
    max_samples = dataset_cfg.get("max_samples")

    if dataset_name != "gsm8k":
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Only gsm8k is supported.")

    return load_gsm8k(split=split, max_samples=max_samples)


def build_model(model_config: dict[str, Any]) -> Model:
    """Instantiate the currently supported model path."""
    model_type = str(model_config.get("type", "dummy"))
    if model_type == "dummy":
        return DummyModel(
            correct_prob=float(model_config.get("correct_prob", 0.3)),
            seed=model_config.get("seed", 42),
        )
    raise ValueError(
        f"Unsupported model type '{model_type}'. "
        "The current real budget sweep only supports the dummy model path."
    )


def _resolve_n_samples_cap(configured_n_samples: int | dict[str, int], baseline_name: str) -> int:
    if isinstance(configured_n_samples, dict):
        return int(configured_n_samples.get(baseline_name, 1))
    return int(configured_n_samples)


def _samples_for_budget(
    baseline_name: str,
    budget: int,
    total_queries: int,
    configured_n_samples: int | dict[str, int],
) -> int:
    if total_queries <= 0:
        raise ValueError("total_queries must be positive")
    if budget < total_queries:
        raise ValueError(
            f"Budget {budget} is infeasible for {total_queries} queries: "
            "all current real-data baselines require at least one sample per query."
        )
    if baseline_name == "greedy":
        return 1

    sample_cap = max(1, _resolve_n_samples_cap(configured_n_samples, baseline_name))
    return min(sample_cap, int(budget) // total_queries)


def _run_one_budget(
    queries: list[Query],
    model_config: dict[str, Any],
    baseline_name: str,
    budget: int,
    configured_n_samples: int | dict[str, int],
) -> dict[str, Any]:
    model = build_model(model_config)
    baseline = BASELINES[baseline_name](model)
    n_samples = _samples_for_budget(
        baseline_name=baseline_name,
        budget=int(budget),
        total_queries=len(queries),
        configured_n_samples=configured_n_samples,
    )

    logger = ExperimentLogger()
    for query in queries:
        if isinstance(model, DummyModel):
            model.set_ground_truth(query.answer)
        result = baseline.solve(
            query_id=query.id,
            question=query.question,
            ground_truth=query.answer,
            n_samples=n_samples,
        )
        logger.log(result)

    summary = logger.summary()
    return {
        "budget": int(budget),
        "baseline": baseline_name,
        "n_samples_per_query": int(n_samples),
        "total_queries": int(summary["total_queries"]),
        "accuracy": float(summary["accuracy"]),
        "total_compute_used": int(summary["total_samples"]),
        "average_compute_per_query": float(summary["avg_samples_per_query"]),
    }


def _build_budget_comparisons(
    grouped_runs: dict[int, dict[str, dict[str, Any]]],
    budgets: list[int],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for budget in budgets:
        budget_runs = grouped_runs[budget]
        sorted_runs = sorted(
            budget_runs.values(),
            key=lambda item: float(item["accuracy"]),
            reverse=True,
        )
        if not sorted_runs:
            continue

        best_run = sorted_runs[0]
        runner_up = None if len(sorted_runs) < 2 else sorted_runs[1]
        comparisons.append(
            {
                "budget": budget,
                "best_baseline": best_run["baseline"],
                "best_accuracy": float(best_run["accuracy"]),
                "runner_up_baseline": None if runner_up is None else runner_up["baseline"],
                "runner_up_accuracy": (
                    None if runner_up is None else float(runner_up["accuracy"])
                ),
                "accuracy_gap_to_runner_up": (
                    None
                    if runner_up is None
                    else float(best_run["accuracy"]) - float(runner_up["accuracy"])
                ),
            }
        )
    return comparisons


def run_real_budget_sweep(config: dict[str, Any]) -> dict[str, Any]:
    """Run multiple real-data baseline evaluations across budgets."""
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    dataset_name = str(dataset_cfg.get("name", "gsm8k"))
    model_type = str(model_cfg.get("type", "dummy"))
    requested_baselines = [str(name) for name in config.get("baselines", ["greedy"])]
    budgets = [int(value) for value in config.get("budgets", [])]
    if not budgets:
        raise ValueError("budgets must contain at least one value")

    runnable_baselines, unsupported_baselines = available_baselines(requested_baselines)
    if not runnable_baselines:
        raise ValueError(
            "No requested baselines are runnable. "
            f"Unsupported baselines: {unsupported_baselines}"
        )

    queries = load_real_queries(config)
    configured_n_samples = config.get("n_samples", 5)

    runs: list[dict[str, Any]] = []
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for budget in budgets:
        for baseline_name in runnable_baselines:
            run = _run_one_budget(
                queries=queries,
                model_config=model_cfg,
                baseline_name=baseline_name,
                budget=budget,
                configured_n_samples=configured_n_samples,
            )
            runs.append(run)
            grouped[budget][baseline_name] = run

    return {
        "dataset": dataset_name,
        "model_type": model_type,
        "budgets": budgets,
        "total_queries": len(queries),
        "available_baselines": runnable_baselines,
        "unavailable_baselines": unsupported_baselines,
        "runs": runs,
        "comparisons": _build_budget_comparisons(grouped_runs=grouped, budgets=budgets),
    }


def flatten_real_budget_runs_for_csv(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-budget real-data results into CSV-friendly rows."""
    return [
        {
            "budget": int(run["budget"]),
            "baseline": str(run["baseline"]),
            "total_queries": int(run["total_queries"]),
            "accuracy": float(run["accuracy"]),
            "total_compute_used": int(run["total_compute_used"]),
            "average_compute_per_query": float(run["average_compute_per_query"]),
            "n_samples_per_query": int(run.get("n_samples_per_query", 1)),
        }
        for run in runs
    ]


def flatten_real_budget_comparisons_for_csv(
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten comparison summaries into CSV-friendly rows."""
    return [
        {
            "budget": int(row["budget"]),
            "best_baseline": row["best_baseline"],
            "best_accuracy": float(row["best_accuracy"]),
            "runner_up_baseline": row["runner_up_baseline"],
            "runner_up_accuracy": row["runner_up_accuracy"],
            "accuracy_gap_to_runner_up": row["accuracy_gap_to_runner_up"],
        }
        for row in comparisons
    ]


def write_csv_rows(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write rows to CSV if non-empty."""
    if not rows:
        raise ValueError("rows must be non-empty")

    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write a JSON payload to disk."""
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2))
    return resolved
