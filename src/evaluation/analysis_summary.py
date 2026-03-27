"""Lightweight summarization helpers for simulated sweep outputs.

This analysis layer sits after the synthetic allocation experiments finish. It
is intended to extract insights from allocator comparisons, highlight baseline
weaknesses, and prepare clean tables/figures for future paper drafts without
changing the experiment logic itself.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _require_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Required results file not found: {resolved}. "
            "Run the simulated sweep before summarizing results."
        )
    return resolved


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = _require_file(path)
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_budget_comparisons(path: str | Path) -> list[dict[str, float]]:
    """Load budget-comparison CSV rows with numeric fields converted."""
    rows = _read_csv_rows(path)
    parsed: list[dict[str, float]] = []
    for row in rows:
        parsed.append(
            {
                "budget": float(row["budget"]),
                "equal_total_expected_utility": float(row["equal_total_expected_utility"]),
                "mckp_total_expected_utility": float(row["mckp_total_expected_utility"]),
                "utility_gap_mckp_minus_equal": float(row["utility_gap_mckp_minus_equal"]),
                "relative_improvement_vs_equal": float(row["relative_improvement_vs_equal"]),
            }
        )
    return parsed


def load_noise_comparisons(path: str | Path) -> list[dict[str, float | str]]:
    """Load noise-comparison CSV rows with numeric fields converted."""
    rows = _read_csv_rows(path)
    parsed: list[dict[str, float | str]] = []
    for row in rows:
        parsed.append(
            {
                "noise_name": row["noise_name"],
                "noise_std": float(row["noise_std"]),
                "equal_true_utility_achieved": float(row["equal_true_utility_achieved"]),
                "mckp_true_utility_achieved": float(row["mckp_true_utility_achieved"]),
                "utility_gap_mckp_minus_equal": float(row["utility_gap_mckp_minus_equal"]),
                "relative_improvement_vs_equal": float(row["relative_improvement_vs_equal"]),
            }
        )
    return parsed


def build_budget_summary_table(
    budget_rows: list[dict[str, float]],
) -> list[dict[str, float | int]]:
    """Convert raw budget comparisons into a cleaner table for inspection."""
    table: list[dict[str, float | int]] = []
    for row in sorted(budget_rows, key=lambda item: item["budget"]):
        table.append(
            {
                "budget": int(row["budget"]),
                "equal_utility": row["equal_total_expected_utility"],
                "mckp_utility": row["mckp_total_expected_utility"],
                "absolute_gap": row["utility_gap_mckp_minus_equal"],
                "relative_improvement_percent": 100.0
                * row["relative_improvement_vs_equal"],
            }
        )
    return table


def build_noise_summary_table(
    noise_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Compute degradation summaries relative to the no-noise baseline."""
    if not noise_rows:
        return []

    sorted_rows = sorted(noise_rows, key=lambda item: float(item["noise_std"]))
    baseline = next(
        (row for row in sorted_rows if float(row["noise_std"]) == 0.0),
        sorted_rows[0],
    )
    baseline_equal = float(baseline["equal_true_utility_achieved"])
    baseline_mckp = float(baseline["mckp_true_utility_achieved"])

    table: list[dict[str, float | str]] = []
    for row in sorted_rows:
        equal_utility = float(row["equal_true_utility_achieved"])
        mckp_utility = float(row["mckp_true_utility_achieved"])
        table.append(
            {
                "noise_name": str(row["noise_name"]),
                "noise_std": float(row["noise_std"]),
                "equal_utility": equal_utility,
                "mckp_utility": mckp_utility,
                "absolute_gap": float(row["utility_gap_mckp_minus_equal"]),
                "relative_improvement_percent": 100.0
                * float(row["relative_improvement_vs_equal"]),
                "equal_degradation_vs_no_noise": baseline_equal - equal_utility,
                "mckp_degradation_vs_no_noise": baseline_mckp - mckp_utility,
            }
        )
    return table


def summarize_budget_metrics(
    budget_summary_rows: list[dict[str, float | int]],
    small_gap_threshold: float = 0.5,
) -> dict[str, Any]:
    """Extract concise budget-sweep metrics for quick reporting.

    We treat the gap as "small" once the absolute utility advantage falls to or
    below a fixed threshold. This keeps the rule simple and easy to read in
    quick diagnostics.
    """
    if not budget_summary_rows:
        raise ValueError("budget_summary_rows must be non-empty")

    peak_row = max(budget_summary_rows, key=lambda row: float(row["absolute_gap"]))
    average_gap = sum(float(row["absolute_gap"]) for row in budget_summary_rows) / len(
        budget_summary_rows
    )
    small_gap_row = next(
        (
            row
            for row in budget_summary_rows
            if float(row["absolute_gap"]) <= small_gap_threshold
        ),
        None,
    )

    return {
        "max_improvement_over_equal": float(peak_row["absolute_gap"]),
        "average_improvement_over_equal": float(average_gap),
        "budget_where_improvement_peaks": int(peak_row["budget"]),
        "budget_where_gap_becomes_small": (
            None if small_gap_row is None else int(small_gap_row["budget"])
        ),
        "small_gap_threshold": float(small_gap_threshold),
    }


def summarize_noise_metrics(
    noise_summary_rows: list[dict[str, float | str]],
) -> dict[str, Any]:
    """Extract concise noise-sensitivity metrics."""
    if not noise_summary_rows:
        raise ValueError("noise_summary_rows must be non-empty")

    rows = sorted(noise_summary_rows, key=lambda row: float(row["noise_std"]))
    advantage_disappears = next(
        (row for row in rows if float(row["absolute_gap"]) <= 0.0),
        None,
    )

    degradation_by_noise = [
        {
            "noise_name": str(row["noise_name"]),
            "noise_std": float(row["noise_std"]),
            "equal_degradation_vs_no_noise": float(row["equal_degradation_vs_no_noise"]),
            "mckp_degradation_vs_no_noise": float(row["mckp_degradation_vs_no_noise"]),
            "absolute_gap": float(row["absolute_gap"]),
        }
        for row in rows
    ]

    return {
        "degradation_by_noise": degradation_by_noise,
        "noise_where_mckp_advantage_disappears": (
            None
            if advantage_disappears is None
            else {
                "noise_name": str(advantage_disappears["noise_name"]),
                "noise_std": float(advantage_disappears["noise_std"]),
                "absolute_gap": float(advantage_disappears["absolute_gap"]),
            }
        ),
    }


def write_csv_table(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write a list of dict rows to CSV."""
    if not rows:
        raise ValueError("rows must be non-empty")

    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def generate_summary_plots(
    budget_summary_rows: list[dict[str, float | int]],
    noise_summary_rows: list[dict[str, float | str]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate simple summary plots when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "generated": False,
            "reason": "matplotlib is not installed; skipping plots",
            "paths": [],
        }

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    budgets = [int(row["budget"]) for row in budget_summary_rows]
    equal_utility = [float(row["equal_utility"]) for row in budget_summary_rows]
    mckp_utility = [float(row["mckp_utility"]) for row in budget_summary_rows]
    gaps = [float(row["absolute_gap"]) for row in budget_summary_rows]

    plt.figure()
    plt.plot(budgets, equal_utility, marker="o", label="equal")
    plt.plot(budgets, mckp_utility, marker="o", label="mckp")
    plt.xlabel("Budget")
    plt.ylabel("Expected utility")
    plt.title("Allocator utility vs budget")
    plt.legend()
    plt.tight_layout()
    utility_budget_path = plot_dir / "utility_vs_budget.png"
    plt.savefig(utility_budget_path)
    plt.close()
    saved_paths.append(str(utility_budget_path))

    plt.figure()
    plt.plot(budgets, gaps, marker="o")
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Budget")
    plt.ylabel("MCKP - equal utility")
    plt.title("Utility gap vs budget")
    plt.tight_layout()
    gap_budget_path = plot_dir / "gap_vs_budget.png"
    plt.savefig(gap_budget_path)
    plt.close()
    saved_paths.append(str(gap_budget_path))

    noise_labels = [str(row["noise_name"]) for row in noise_summary_rows]
    equal_noise_utility = [float(row["equal_utility"]) for row in noise_summary_rows]
    mckp_noise_utility = [float(row["mckp_utility"]) for row in noise_summary_rows]

    plt.figure()
    plt.plot(noise_labels, equal_noise_utility, marker="o", label="equal")
    plt.plot(noise_labels, mckp_noise_utility, marker="o", label="mckp")
    plt.xlabel("Noise level")
    plt.ylabel("True achieved utility")
    plt.title("Allocator utility vs utility-estimate noise")
    plt.legend()
    plt.tight_layout()
    utility_noise_path = plot_dir / "utility_vs_noise.png"
    plt.savefig(utility_noise_path)
    plt.close()
    saved_paths.append(str(utility_noise_path))

    return {
        "generated": True,
        "reason": None,
        "paths": saved_paths,
    }


def summarize_simulated_results(
    input_dir: str | Path = "outputs/simulated_sweep",
    small_gap_threshold: float = 0.5,
) -> dict[str, Any]:
    """Load simulated sweep outputs, build tables, save CSVs, and plot if possible."""
    base_dir = Path(input_dir)
    budget_rows = load_budget_comparisons(base_dir / "budget_sweep_comparisons.csv")
    noise_rows = load_noise_comparisons(base_dir / "noise_sensitivity_comparisons.csv")

    budget_summary_table = build_budget_summary_table(budget_rows)
    noise_summary_table = build_noise_summary_table(noise_rows)

    summary_table_path = write_csv_table(
        budget_summary_table,
        base_dir / "summary_table.csv",
    )
    noise_summary_path = write_csv_table(
        noise_summary_table,
        base_dir / "noise_summary_table.csv",
    )

    budget_metrics = summarize_budget_metrics(
        budget_summary_table,
        small_gap_threshold=small_gap_threshold,
    )
    noise_metrics = summarize_noise_metrics(noise_summary_table)
    plot_summary = generate_summary_plots(
        budget_summary_rows=budget_summary_table,
        noise_summary_rows=noise_summary_table,
        output_dir=base_dir / "plots",
    )

    return {
        "input_dir": str(base_dir),
        "summary_table_path": str(summary_table_path),
        "noise_summary_table_path": str(noise_summary_path),
        "budget_summary_table": budget_summary_table,
        "noise_summary_table": noise_summary_table,
        "budget_metrics": budget_metrics,
        "noise_metrics": noise_metrics,
        "plots": plot_summary,
    }


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_terminal_summary(summary: dict[str, Any]) -> str:
    """Render a concise text summary for quick terminal inspection."""
    budget_metrics = summary["budget_metrics"]
    noise_metrics = summary["noise_metrics"]
    disappearing_advantage = noise_metrics["noise_where_mckp_advantage_disappears"]
    plot_summary = summary["plots"]

    lines = [
        "--- Simulated Sweep Summary ---",
        f"summary_table:              {summary['summary_table_path']}",
        f"noise_summary_table:        {summary['noise_summary_table_path']}",
        "",
        "Budget sweep:",
        "  max improvement over equal: "
        f"{_format_float(budget_metrics['max_improvement_over_equal'])}",
        "  average improvement:       "
        f"{_format_float(budget_metrics['average_improvement_over_equal'])}",
        f"  budget where improvement peaks: {budget_metrics['budget_where_improvement_peaks']}",
        "  budget where gap becomes small: "
        f"{budget_metrics['budget_where_gap_becomes_small']}",
        "",
        "Noise sensitivity:",
    ]

    if disappearing_advantage is None:
        lines.append("  MCKP advantage disappearance point: not observed")
    else:
        lines.append(
            "  MCKP advantage disappears at: "
            f"{disappearing_advantage['noise_name']} "
            f"(std={_format_float(disappearing_advantage['noise_std'])}, "
            f"gap={_format_float(disappearing_advantage['absolute_gap'])})"
        )

    for row in noise_metrics["degradation_by_noise"]:
        lines.append(
            "  "
            f"{row['noise_name']}: "
            f"equal_deg={_format_float(row['equal_degradation_vs_no_noise'])}, "
            f"mckp_deg={_format_float(row['mckp_degradation_vs_no_noise'])}, "
            f"gap={_format_float(row['absolute_gap'])}"
        )

    lines.extend(
        [
            "",
            f"plots_generated:           {plot_summary['generated']}",
        ]
    )
    if plot_summary["generated"]:
        lines.append(f"plots_dir:                 {Path(summary['input_dir']) / 'plots'}")
    else:
        lines.append(f"plots_note:                {plot_summary['reason']}")

    return "\n".join(lines)

