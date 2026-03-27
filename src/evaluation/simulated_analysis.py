"""Diagnostic helpers for synthetic allocation experiments.

This analysis layer is intentionally small and readable. Its scientific role is
to help diagnose:
1) when classical optimization helps over simple equal allocation,
2) whether gains come from shifting budget toward harder queries, and
3) how sensitive optimization is to noisy utility estimates.

These diagnostics are also useful for identifying failure modes that future
adaptive allocation methods may face when their difficulty predictions are
miscalibrated.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from src.allocators.registry import get_allocator

DIFFICULTY_ORDER = ("easy", "medium", "hard", "n/a")
DEFAULT_NOISE_LEVELS = (
    {"name": "no_noise", "std": 0.0},
    {"name": "small_gaussian", "std": 0.02},
    {"name": "medium_gaussian", "std": 0.05},
    {"name": "large_gaussian", "std": 0.1},
)


def _to_2d_float_array(table: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(table, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rectangular 2-D table")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one query and one level")
    return array


def _to_1d_int_array(values: Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=int)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value")
    return array


def _level_key(level: int) -> str:
    return str(int(level))


def _empty_level_map(n_levels: int) -> dict[str, int]:
    return {_level_key(level): 0 for level in range(n_levels)}


def _preferred_difficulty_order(labels: Sequence[str]) -> list[str]:
    observed = {str(label) for label in labels}
    ordered = [label for label in DIFFICULTY_ORDER if label in observed]

    if observed and observed.issubset({"easy", "medium", "hard"}):
        return ["easy", "medium", "hard"]

    extras = sorted(observed.difference(DIFFICULTY_ORDER))
    return ordered + extras


def score_selected_levels(
    utility_table: Sequence[Sequence[float]],
    costs: Sequence[int],
    selected_levels: Sequence[int],
) -> dict[str, Any]:
    """Score an allocation against a utility table."""
    utility_array = _to_2d_float_array(utility_table, "utility_table")
    cost_array = _to_1d_int_array(costs, "costs")
    levels = [int(level) for level in selected_levels]

    n_queries, n_levels = utility_array.shape
    if len(levels) != n_queries:
        raise ValueError("selected_levels length must match number of queries")
    if len(cost_array) != n_levels:
        raise ValueError("len(costs) must equal number of utility levels")
    if any(level < 0 or level >= n_levels for level in levels):
        raise ValueError("selected_levels contains an out-of-range level index")

    per_query_costs = [int(cost_array[level]) for level in levels]
    per_query_utilities = [float(utility_array[idx, level]) for idx, level in enumerate(levels)]
    total_cost = int(sum(per_query_costs))
    total_utility = float(sum(per_query_utilities))

    return {
        "selected_levels": levels,
        "per_query_costs": per_query_costs,
        "per_query_utilities": per_query_utilities,
        "total_cost": total_cost,
        "total_expected_utility": total_utility,
        "average_cost_per_query": total_cost / n_queries,
        "average_utility_per_query": total_utility / n_queries,
    }


def summarize_allocation_distribution(
    selected_levels: Sequence[int],
    n_levels: int,
) -> dict[str, Any]:
    """Summarize how often each compute level is selected."""
    if n_levels <= 0:
        raise ValueError("n_levels must be positive")

    levels = [int(level) for level in selected_levels]
    if not levels:
        raise ValueError("selected_levels must be non-empty")
    if any(level < 0 or level >= n_levels for level in levels):
        raise ValueError("selected_levels contains an out-of-range level index")

    counts = _empty_level_map(n_levels)
    for level in levels:
        counts[_level_key(level)] += 1

    total = len(levels)
    fractions = {level: count / total for level, count in counts.items()}

    return {
        "selected_level_counts": counts,
        "selected_level_fractions": fractions,
        "min_selected_level": int(min(levels)),
        "max_selected_level": int(max(levels)),
        "mean_selected_level": float(sum(levels) / total),
    }


def summarize_by_difficulty(
    difficulty_labels: Sequence[str],
    selected_levels: Sequence[int],
    costs: Sequence[int],
    utility_table: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Summarize allocation behavior and achieved utility by difficulty group."""
    labels = [str(label) for label in difficulty_labels]
    scored = score_selected_levels(
        utility_table=utility_table,
        costs=costs,
        selected_levels=selected_levels,
    )
    levels = scored["selected_levels"]
    n_queries = len(levels)
    if len(labels) != n_queries:
        raise ValueError("difficulty_labels length must match number of queries")

    n_levels = len(costs)
    summaries: dict[str, Any] = {}

    for difficulty in _preferred_difficulty_order(labels):
        indices = [idx for idx, label in enumerate(labels) if label == difficulty]
        level_counts = _empty_level_map(n_levels)
        if not indices:
            summaries[difficulty] = {
                "num_queries": 0,
                "selected_level_counts": level_counts,
                "selected_level_fractions": {
                    level_key: 0.0 for level_key in level_counts
                },
                "average_assigned_cost": 0.0,
                "average_achieved_utility": 0.0,
            }
            continue

        group_levels = [levels[idx] for idx in indices]
        for level in group_levels:
            level_counts[_level_key(level)] += 1

        group_costs = [scored["per_query_costs"][idx] for idx in indices]
        group_utilities = [scored["per_query_utilities"][idx] for idx in indices]
        group_size = len(indices)

        summaries[difficulty] = {
            "num_queries": group_size,
            "selected_level_counts": level_counts,
            "selected_level_fractions": {
                level_key: count / group_size for level_key, count in level_counts.items()
            },
            "average_assigned_cost": float(sum(group_costs) / group_size),
            "average_achieved_utility": float(sum(group_utilities) / group_size),
        }

    return summaries


def add_gaussian_noise(
    utility_table: Sequence[Sequence[float]],
    noise_std: float,
    seed: int | None = None,
    preserve_monotonicity: bool = True,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Create a noisy estimated utility table for allocator-side prediction."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    utility_array = _to_2d_float_array(utility_table, "utility_table")
    rng = np.random.default_rng(seed)
    noisy = utility_array + rng.normal(loc=0.0, scale=noise_std, size=utility_array.shape)
    noisy = np.clip(noisy, 0.0, 1.0)

    monotonic_projection_applied = False
    if preserve_monotonicity:
        noisy = np.maximum.accumulate(noisy, axis=1)
        noisy = np.clip(noisy, 0.0, 1.0)
        monotonic_projection_applied = True

    return noisy.tolist(), {
        "noise_std": float(noise_std),
        "preserve_monotonicity": preserve_monotonicity,
        "monotonic_projection_applied": monotonic_projection_applied,
    }


def run_allocation_analysis(
    utility_table: Sequence[Sequence[float]],
    costs: Sequence[int],
    budget: int,
    allocator_name: str,
    difficulty_labels: Sequence[str] | None = None,
    estimated_utility_table: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Run one allocator and attach reusable diagnostics."""
    true_utility_array = _to_2d_float_array(utility_table, "utility_table")
    estimated_utility_array = (
        true_utility_array
        if estimated_utility_table is None
        else _to_2d_float_array(estimated_utility_table, "estimated_utility_table")
    )
    cost_array = _to_1d_int_array(costs, "costs")

    if estimated_utility_array.shape != true_utility_array.shape:
        raise ValueError("estimated_utility_table must match utility_table shape")
    if len(cost_array) != true_utility_array.shape[1]:
        raise ValueError("len(costs) must equal number of levels")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    allocator = get_allocator(allocator_name)
    allocation_result = allocator.allocate(
        profits=estimated_utility_array.tolist(),
        costs=cost_array.tolist(),
        budget=int(budget),
    )

    selected_levels = [int(level) for level in allocation_result["selected_levels"]]
    true_metrics = score_selected_levels(
        utility_table=true_utility_array.tolist(),
        costs=cost_array.tolist(),
        selected_levels=selected_levels,
    )
    estimated_metrics = score_selected_levels(
        utility_table=estimated_utility_array.tolist(),
        costs=cost_array.tolist(),
        selected_levels=selected_levels,
    )

    analysis = {
        "budget": int(budget),
        "allocator": allocator_name,
        "selected_levels": selected_levels,
        "total_expected_utility": true_metrics["total_expected_utility"],
        "total_cost": true_metrics["total_cost"],
        "average_cost_per_query": true_metrics["average_cost_per_query"],
        "average_utility_per_query": true_metrics["average_utility_per_query"],
        "true_utility_achieved": true_metrics["total_expected_utility"],
        "estimated_utility_optimized": estimated_metrics["total_expected_utility"],
        "estimation_error_on_selected_allocation": (
            estimated_metrics["total_expected_utility"]
            - true_metrics["total_expected_utility"]
        ),
        "allocation_distribution": summarize_allocation_distribution(
            selected_levels=selected_levels,
            n_levels=true_utility_array.shape[1],
        ),
    }

    if difficulty_labels is not None:
        analysis["difficulty_summary"] = summarize_by_difficulty(
            difficulty_labels=difficulty_labels,
            selected_levels=selected_levels,
            costs=cost_array.tolist(),
            utility_table=true_utility_array.tolist(),
        )

    return analysis


def run_budget_sweep(
    utility_table: Sequence[Sequence[float]],
    costs: Sequence[int],
    budgets: Sequence[int],
    allocator_names: Sequence[str],
    difficulty_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run repeated synthetic experiments over a fixed instance and budget list."""
    resolved_budgets = [int(budget) for budget in budgets]
    if not resolved_budgets:
        raise ValueError("budgets must contain at least one value")
    resolved_allocators = [str(name) for name in allocator_names]
    if not resolved_allocators:
        raise ValueError("allocator_names must contain at least one allocator")

    runs: list[dict[str, Any]] = []
    by_budget: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)

    for budget in resolved_budgets:
        for allocator_name in resolved_allocators:
            run = run_allocation_analysis(
                utility_table=utility_table,
                costs=costs,
                budget=budget,
                allocator_name=allocator_name,
                difficulty_labels=difficulty_labels,
            )
            runs.append(run)
            by_budget[budget][allocator_name] = run

    comparisons: list[dict[str, Any]] = []
    for budget in resolved_budgets:
        equal_run = by_budget[budget].get("equal")
        mckp_run = by_budget[budget].get("mckp")
        if equal_run is None or mckp_run is None:
            continue

        equal_utility = float(equal_run["total_expected_utility"])
        mckp_utility = float(mckp_run["total_expected_utility"])
        utility_gap = mckp_utility - equal_utility
        relative_improvement = None
        if abs(equal_utility) > 1e-12:
            relative_improvement = utility_gap / equal_utility

        comparisons.append(
            {
                "budget": budget,
                "equal_total_expected_utility": equal_utility,
                "mckp_total_expected_utility": mckp_utility,
                "utility_gap_mckp_minus_equal": utility_gap,
                "relative_improvement_vs_equal": relative_improvement,
            }
        )

    return {
        "budgets": resolved_budgets,
        "allocator_names": resolved_allocators,
        "runs": runs,
        "comparisons": comparisons,
    }


def resolve_noise_levels(
    noise_levels: Sequence[str | dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize configured noise levels into a simple list of name/std pairs."""
    if noise_levels is None:
        return [dict(level) for level in DEFAULT_NOISE_LEVELS]

    presets = {level["name"]: dict(level) for level in DEFAULT_NOISE_LEVELS}
    resolved: list[dict[str, Any]] = []
    for level in noise_levels:
        if isinstance(level, str):
            if level not in presets:
                raise ValueError(f"Unknown noise preset '{level}'")
            resolved.append(dict(presets[level]))
            continue

        if "name" not in level or "std" not in level:
            raise ValueError("Noise level dicts must contain 'name' and 'std'")
        resolved.append({"name": str(level["name"]), "std": float(level["std"])})

    return resolved


def run_noise_sensitivity_experiments(
    utility_table: Sequence[Sequence[float]],
    costs: Sequence[int],
    budget: int,
    allocator_names: Sequence[str],
    difficulty_labels: Sequence[str] | None = None,
    noise_levels: Sequence[str | dict[str, Any]] | None = None,
    seed: int | None = None,
    preserve_monotonicity: bool = True,
) -> dict[str, Any]:
    """Evaluate how noisy utility estimates change allocator behavior."""
    resolved_noise_levels = resolve_noise_levels(noise_levels)
    resolved_allocators = [str(name) for name in allocator_names]
    if not resolved_allocators:
        raise ValueError("allocator_names must contain at least one allocator")

    runs: list[dict[str, Any]] = []
    baseline_true_utility: dict[str, float] = {}

    for offset, noise_spec in enumerate(resolved_noise_levels):
        noise_seed = None if seed is None else int(seed) + offset
        estimated_utility_table, noise_metadata = add_gaussian_noise(
            utility_table=utility_table,
            noise_std=float(noise_spec["std"]),
            seed=noise_seed,
            preserve_monotonicity=preserve_monotonicity,
        )

        for allocator_name in resolved_allocators:
            run = run_allocation_analysis(
                utility_table=utility_table,
                costs=costs,
                budget=budget,
                allocator_name=allocator_name,
                difficulty_labels=difficulty_labels,
                estimated_utility_table=estimated_utility_table,
            )
            run["noise"] = {
                "name": str(noise_spec["name"]),
                **noise_metadata,
            }

            if float(noise_spec["std"]) == 0.0 and allocator_name not in baseline_true_utility:
                baseline_true_utility[allocator_name] = float(run["true_utility_achieved"])

            baseline = baseline_true_utility.get(allocator_name)
            run["degradation_vs_no_noise"] = (
                None if baseline is None else baseline - float(run["true_utility_achieved"])
            )
            runs.append(run)

    comparisons: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        grouped[run["noise"]["name"]][run["allocator"]] = run

    for noise_name, allocator_runs in grouped.items():
        equal_run = allocator_runs.get("equal")
        mckp_run = allocator_runs.get("mckp")
        if equal_run is None or mckp_run is None:
            continue
        equal_true = float(equal_run["true_utility_achieved"])
        mckp_true = float(mckp_run["true_utility_achieved"])
        utility_gap = mckp_true - equal_true
        relative_improvement = None
        if abs(equal_true) > 1e-12:
            relative_improvement = utility_gap / equal_true

        comparisons.append(
            {
                "noise_name": noise_name,
                "noise_std": float(mckp_run["noise"]["noise_std"]),
                "equal_true_utility_achieved": equal_true,
                "mckp_true_utility_achieved": mckp_true,
                "utility_gap_mckp_minus_equal": utility_gap,
                "relative_improvement_vs_equal": relative_improvement,
            }
        )

    return {
        "budget": int(budget),
        "allocator_names": resolved_allocators,
        "noise_levels": resolved_noise_levels,
        "runs": runs,
        "comparisons": comparisons,
    }


def resolve_budget_list(configured_budgets: Sequence[int] | None, budget_range: dict[str, int] | None) -> list[int]:
    """Resolve either an explicit budget list or an inclusive integer range."""
    if configured_budgets:
        return [int(budget) for budget in configured_budgets]

    if budget_range is None:
        raise ValueError("Either budgets or budget_range must be provided")

    start = int(budget_range["start"])
    stop = int(budget_range["stop"])
    step = int(budget_range.get("step", 1))
    if step <= 0:
        raise ValueError("budget_range.step must be positive")
    if stop < start:
        raise ValueError("budget_range.stop must be >= budget_range.start")

    return list(range(start, stop + 1, step))


def flatten_budget_runs_for_csv(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the most useful budget-sweep metrics for simple CSV export."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.append(
            {
                "budget": run["budget"],
                "allocator": run["allocator"],
                "total_expected_utility": run["total_expected_utility"],
                "total_cost": run["total_cost"],
                "average_cost_per_query": run["average_cost_per_query"],
                "average_utility_per_query": run["average_utility_per_query"],
                "min_selected_level": run["allocation_distribution"]["min_selected_level"],
                "max_selected_level": run["allocation_distribution"]["max_selected_level"],
                "mean_selected_level": run["allocation_distribution"]["mean_selected_level"],
            }
        )
    return rows


def flatten_noise_runs_for_csv(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the most useful noise-sensitivity metrics for simple CSV export."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.append(
            {
                "budget": run["budget"],
                "allocator": run["allocator"],
                "noise_name": run["noise"]["name"],
                "noise_std": run["noise"]["noise_std"],
                "true_utility_achieved": run["true_utility_achieved"],
                "estimated_utility_optimized": run["estimated_utility_optimized"],
                "estimation_error_on_selected_allocation": run[
                    "estimation_error_on_selected_allocation"
                ],
                "degradation_vs_no_noise": run["degradation_vs_no_noise"],
                "total_cost": run["total_cost"],
            }
        )
    return rows
