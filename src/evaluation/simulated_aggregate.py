"""Aggregate simulated sweep diagnostics across multiple random seeds.

This layer is intentionally lightweight and sits on top of the existing
single-seed synthetic sweep tooling. Its scientific role is to test whether:
1) optimization gains are stable across different synthetic instance draws,
2) fragility under prediction noise is robust rather than incidental, and
3) conclusions from one seed generalize beyond a single synthetic sample.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _std(values: list[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("rows must be non-empty")
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def _write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2))
    return resolved


def flatten_per_seed_budget_rows(
    seed_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten per-seed budget comparisons for CSV export."""
    rows: list[dict[str, Any]] = []
    for result in seed_results:
        seed = int(result["seed"])
        for row in result["budget_comparisons"]:
            rows.append(
                {
                    "seed": seed,
                    "budget": int(row["budget"]),
                    "equal_total_expected_utility": float(
                        row["equal_total_expected_utility"]
                    ),
                    "mckp_total_expected_utility": float(
                        row["mckp_total_expected_utility"]
                    ),
                    "utility_gap_mckp_minus_equal": float(
                        row["utility_gap_mckp_minus_equal"]
                    ),
                    "relative_improvement_vs_equal": float(
                        row["relative_improvement_vs_equal"]
                    ),
                }
            )
    return rows


def flatten_per_seed_noise_rows(
    seed_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten per-seed noise comparisons for CSV export."""
    rows: list[dict[str, Any]] = []
    for result in seed_results:
        seed = int(result["seed"])
        for row in result["noise_comparisons"]:
            rows.append(
                {
                    "seed": seed,
                    "noise_name": str(row["noise_name"]),
                    "noise_std": float(row["noise_std"]),
                    "equal_true_utility_achieved": float(
                        row["equal_true_utility_achieved"]
                    ),
                    "mckp_true_utility_achieved": float(
                        row["mckp_true_utility_achieved"]
                    ),
                    "utility_gap_mckp_minus_equal": float(
                        row["utility_gap_mckp_minus_equal"]
                    ),
                    "relative_improvement_vs_equal": float(
                        row["relative_improvement_vs_equal"]
                    ),
                }
            )
    return rows


def aggregate_budget_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate budget comparisons across seeds."""
    if not rows:
        raise ValueError("rows must be non-empty")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["budget"])].append(row)

    aggregated: list[dict[str, Any]] = []
    for budget in sorted(grouped):
        budget_rows = grouped[budget]
        equal_values = [float(row["equal_total_expected_utility"]) for row in budget_rows]
        mckp_values = [float(row["mckp_total_expected_utility"]) for row in budget_rows]
        gap_values = [float(row["utility_gap_mckp_minus_equal"]) for row in budget_rows]
        aggregated.append(
            {
                "budget": budget,
                "n_seeds": len(budget_rows),
                "mean_equal_utility": float(mean(equal_values)),
                "std_equal_utility": _std(equal_values),
                "mean_mckp_utility": float(mean(mckp_values)),
                "std_mckp_utility": _std(mckp_values),
                "mean_utility_gap": float(mean(gap_values)),
                "std_utility_gap": _std(gap_values),
                "fraction_mckp_beats_equal": (
                    sum(1 for value in gap_values if value > 0.0) / len(gap_values)
                ),
            }
        )
    return aggregated


def aggregate_noise_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate noise comparisons across seeds."""
    if not rows:
        raise ValueError("rows must be non-empty")

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["noise_name"]), float(row["noise_std"]))
        grouped[key].append(row)

    aggregated: list[dict[str, Any]] = []
    for noise_name, noise_std in sorted(grouped, key=lambda item: item[1]):
        noise_rows = grouped[(noise_name, noise_std)]
        equal_values = [
            float(row["equal_true_utility_achieved"]) for row in noise_rows
        ]
        mckp_values = [
            float(row["mckp_true_utility_achieved"]) for row in noise_rows
        ]
        gap_values = [float(row["utility_gap_mckp_minus_equal"]) for row in noise_rows]
        aggregated.append(
            {
                "noise_name": noise_name,
                "noise_std": noise_std,
                "n_seeds": len(noise_rows),
                "mean_equal_utility": float(mean(equal_values)),
                "std_equal_utility": _std(equal_values),
                "mean_mckp_utility": float(mean(mckp_values)),
                "std_mckp_utility": _std(mckp_values),
                "mean_utility_gap": float(mean(gap_values)),
                "std_utility_gap": _std(gap_values),
                "fraction_mckp_beats_equal": (
                    sum(1 for value in gap_values if value > 0.0) / len(gap_values)
                ),
            }
        )
    return aggregated


def compute_multi_seed_key_findings(
    budget_summary_rows: list[dict[str, Any]],
    noise_summary_rows: list[dict[str, Any]],
    small_gap_threshold: float = 0.5,
) -> dict[str, Any]:
    """Extract compact findings from aggregated multi-seed summaries."""
    if not budget_summary_rows:
        raise ValueError("budget_summary_rows must be non-empty")
    if not noise_summary_rows:
        raise ValueError("noise_summary_rows must be non-empty")

    peak_budget_row = max(
        budget_summary_rows,
        key=lambda row: float(row["mean_utility_gap"]),
    )
    small_gap_row = next(
        (
            row
            for row in budget_summary_rows
            if float(row["mean_utility_gap"]) <= small_gap_threshold
        ),
        None,
    )
    disappearing_noise_row = next(
        (
            row
            for row in sorted(
                noise_summary_rows,
                key=lambda item: float(item["noise_std"]),
            )
            if float(row["mean_utility_gap"]) <= 0.0
        ),
        None,
    )

    return {
        "budget_where_mean_gap_peaks": int(peak_budget_row["budget"]),
        "peak_mean_gap": float(peak_budget_row["mean_utility_gap"]),
        "budget_where_mean_gap_becomes_small": (
            None if small_gap_row is None else int(small_gap_row["budget"])
        ),
        "small_gap_threshold": float(small_gap_threshold),
        "first_noise_level_where_mean_mckp_advantage_disappears": (
            None
            if disappearing_noise_row is None
            else {
                "noise_name": str(disappearing_noise_row["noise_name"]),
                "noise_std": float(disappearing_noise_row["noise_std"]),
                "mean_utility_gap": float(disappearing_noise_row["mean_utility_gap"]),
            }
        ),
        "fraction_mckp_beats_equal_by_noise": {
            str(row["noise_name"]): float(row["fraction_mckp_beats_equal"])
            for row in noise_summary_rows
        },
    }


def aggregate_budget_comparisons(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compatibility wrapper for budget aggregation in tests and scripts."""
    return aggregate_budget_rows(rows)


def aggregate_noise_comparisons(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compatibility wrapper for noise aggregation in tests and scripts."""
    return aggregate_noise_rows(rows)


def aggregate_simulated_results(
    per_seed_results: list[dict[str, Any]],
    small_gap_threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate in-memory per-seed comparison results without writing files."""
    if not per_seed_results:
        raise ValueError("per_seed_results must be non-empty")

    budget_rows = flatten_per_seed_budget_rows(per_seed_results)
    noise_rows = flatten_per_seed_noise_rows(per_seed_results)
    budget_summary = aggregate_budget_rows(budget_rows)
    noise_summary = aggregate_noise_rows(noise_rows)
    key_findings = compute_multi_seed_key_findings(
        budget_summary_rows=budget_summary,
        noise_summary_rows=noise_summary,
        small_gap_threshold=small_gap_threshold,
    )

    return {
        "n_seeds": len(per_seed_results),
        "seeds": [int(result["seed"]) for result in per_seed_results],
        "budget_summary": budget_summary,
        "noise_summary": noise_summary,
        "key_findings": key_findings,
    }


def aggregate_multi_seed_results(
    seed_results: list[dict[str, Any]],
    output_dir: str | Path,
    small_gap_threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate per-seed sweep comparisons and write summary artifacts."""
    if not seed_results:
        raise ValueError("seed_results must be non-empty")

    base_dir = Path(output_dir)
    aggregate = aggregate_simulated_results(
        per_seed_results=seed_results,
        small_gap_threshold=small_gap_threshold,
    )
    per_seed_budget_rows = flatten_per_seed_budget_rows(seed_results)
    per_seed_noise_rows = flatten_per_seed_noise_rows(seed_results)
    aggregated_budget = aggregate["budget_summary"]
    aggregated_noise = aggregate["noise_summary"]
    key_findings = aggregate["key_findings"]

    per_seed_budget_path = _write_csv(
        per_seed_budget_rows,
        base_dir / "per_seed_budget_runs.csv",
    )
    per_seed_noise_path = _write_csv(
        per_seed_noise_rows,
        base_dir / "per_seed_noise_runs.csv",
    )
    budget_summary_path = _write_csv(
        aggregated_budget,
        base_dir / "aggregated_budget_summary.csv",
    )
    noise_summary_path = _write_csv(
        aggregated_noise,
        base_dir / "aggregated_noise_summary.csv",
    )
    findings_json_path = _write_json(
        {
            "n_seeds": len(seed_results),
            "seeds": [int(result["seed"]) for result in seed_results],
            "key_findings": key_findings,
        },
        base_dir / "aggregated_key_findings.json",
    )

    return {
        "n_seeds": aggregate["n_seeds"],
        "seeds": aggregate["seeds"],
        "per_seed_budget_rows": per_seed_budget_rows,
        "per_seed_noise_rows": per_seed_noise_rows,
        "aggregated_budget_summary": aggregated_budget,
        "aggregated_noise_summary": aggregated_noise,
        "key_findings": key_findings,
        "paths": {
            "per_seed_budget_runs": str(per_seed_budget_path),
            "per_seed_noise_runs": str(per_seed_noise_path),
            "aggregated_budget_summary": str(budget_summary_path),
            "aggregated_noise_summary": str(noise_summary_path),
            "aggregated_key_findings": str(findings_json_path),
        },
    }
