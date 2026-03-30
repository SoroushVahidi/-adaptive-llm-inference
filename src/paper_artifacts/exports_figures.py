"""Build manuscript figures from existing CSV artifacts (matplotlib optional)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from .io_util import ensure_dir, load_json
from .paths import REAL_POLICY_SUMMARY_FILES, ArtifactPaths


class FigureBlocker(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise FigureBlocker(
            "BLOCKED figures: matplotlib is not installed. "
            "Install the project with dev extras or `pip install matplotlib`, then retry."
        ) from e


def figure_simulated_sweep(art: ArtifactPaths, out_dir: Path) -> list[Path]:
    plt = _require_matplotlib()
    base = art.simulated_sweep
    budget_csv = base / "budget_sweep_comparisons.csv"
    noise_csv = base / "noise_sensitivity_comparisons.csv"
    if not budget_csv.is_file():
        raise FigureBlocker(
            "BLOCKED simulated sweep figures: missing "
            f"{budget_csv.relative_to(art.root)}. "
            "Run: python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml"
        )
    if not noise_csv.is_file():
        raise FigureBlocker(
            "BLOCKED simulated sweep figures: missing "
            f"{noise_csv.relative_to(art.root)}."
        )

    from src.evaluation.analysis_summary import (
        build_budget_summary_table,
        build_noise_summary_table,
        load_budget_comparisons,
        load_noise_comparisons,
    )

    budget_summary = build_budget_summary_table(load_budget_comparisons(budget_csv))
    noise_summary = build_noise_summary_table(load_noise_comparisons(noise_csv))
    ensure_dir(out_dir)
    saved: list[Path] = []

    budgets = [int(r["budget"]) for r in budget_summary]
    eq = [float(r["equal_utility"]) for r in budget_summary]
    mckp = [float(r["mckp_utility"]) for r in budget_summary]
    gaps = [float(r["absolute_gap"]) for r in budget_summary]

    plt.figure(figsize=(6, 4))
    plt.plot(budgets, eq, marker="o", label="equal")
    plt.plot(budgets, mckp, marker="o", label="mckp")
    plt.xlabel("Budget")
    plt.ylabel("Expected utility")
    plt.title("Simulated allocation: utility vs budget")
    plt.legend()
    plt.tight_layout()
    p1 = out_dir / "simulated_sweep_utility_vs_budget.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    saved.append(p1)

    plt.figure(figsize=(6, 4))
    plt.plot(budgets, gaps, marker="o")
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Budget")
    plt.ylabel("MCKP − equal utility")
    plt.title("Simulated allocation: utility gap vs budget")
    plt.tight_layout()
    p2 = out_dir / "simulated_sweep_gap_vs_budget.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    saved.append(p2)

    labels = [str(r["noise_name"]) for r in noise_summary]
    eq_n = [float(r["equal_utility"]) for r in noise_summary]
    mckp_n = [float(r["mckp_utility"]) for r in noise_summary]
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(labels)), eq_n, marker="o", label="equal")
    plt.plot(range(len(labels)), mckp_n, marker="o", label="mckp")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Noise level")
    plt.ylabel("True achieved utility")
    plt.title("Simulated allocation: utility vs estimate noise")
    plt.legend()
    plt.tight_layout()
    p3 = out_dir / "simulated_sweep_utility_vs_noise.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    saved.append(p3)

    return saved


def figure_next_stage_budget_curves(art: ArtifactPaths, out_dir: Path) -> list[Path]:
    plt = _require_matplotlib()
    base = art.next_stage_eval
    if not base.is_dir():
        raise FigureBlocker(
            "BLOCKED next-stage budget figures: missing "
            f"{base.relative_to(art.root)}."
        )
    ensure_dir(out_dir)
    saved: list[Path] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        curve = sub / "budget_curve.csv"
        if not curve.is_file():
            continue
        rows = list(csv.DictReader(curve.open(encoding="utf-8")))
        if not rows:
            continue
        costs = [float(r["achieved_avg_cost"]) for r in rows]
        acc = [float(r["accuracy"]) for r in rows]
        plt.figure(figsize=(5, 3.5))
        plt.plot(costs, acc, marker="o")
        plt.xlabel("Achieved average cost")
        plt.ylabel("Accuracy")
        plt.title(f"Budget curve ({sub.name})")
        plt.tight_layout()
        safe = sub.name.replace("/", "_")
        p = out_dir / f"next_stage_budget_curve_{safe}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        saved.append(p)
    if not saved:
        raise FigureBlocker(
            "BLOCKED next-stage budget figures: no budget_curve.csv under "
            f"{base.relative_to(art.root)}/*/"
        )
    return saved


def figure_next_stage_cascade_curves(art: ArtifactPaths, out_dir: Path) -> list[Path]:
    plt = _require_matplotlib()
    base = art.next_stage_eval
    if not base.is_dir():
        raise FigureBlocker(
            f"BLOCKED cascade figures: missing {base.relative_to(art.root)}."
        )
    ensure_dir(out_dir)
    saved: list[Path] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        curve = sub / "cascade_curve.csv"
        if not curve.is_file():
            continue
        rows = list(csv.DictReader(curve.open(encoding="utf-8")))
        if not rows:
            continue
        thr = [float(r["threshold"]) for r in rows]
        acc = [float(r["accuracy"]) for r in rows]
        cost = [float(r["avg_cost"]) for r in rows]
        plt.figure(figsize=(5, 3.5))
        plt.plot(thr, acc, marker="o", label="accuracy")
        plt.plot(thr, cost, marker="s", label="avg_cost")
        plt.xlabel("unified_confidence_score threshold")
        plt.ylabel("value")
        plt.title(f"Cascade curve ({sub.name})")
        plt.legend()
        plt.tight_layout()
        safe = sub.name.replace("/", "_")
        p = out_dir / f"next_stage_cascade_curve_{safe}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        saved.append(p)
    if not saved:
        raise FigureBlocker(
            "BLOCKED cascade figures: no cascade_curve.csv under "
            f"{base.relative_to(art.root)}/*/"
        )
    return saved


def figure_real_policy_frontier(art: ArtifactPaths, out_dir: Path) -> list[Path]:
    """One Pareto-style scatter per policy eval summary (accuracy vs avg_cost)."""
    plt = _require_matplotlib()
    ensure_dir(out_dir)
    saved: list[Path] = []
    for rel in REAL_POLICY_SUMMARY_FILES:
        p = art.root / rel
        if not p.is_file():
            continue
        data = load_json(p)
        comp = data.get("comparison")
        if not isinstance(comp, list) or not comp:
            continue
        points = [
            r
            for r in comp
            if isinstance(r, dict)
            and "avg_cost" in r
            and "accuracy" in r
        ]
        if not points:
            continue
        costs = [float(r["avg_cost"]) for r in points]
        acc = [float(r["accuracy"]) for r in points]
        labels = [str(r.get("route", "")) for r in points]
        plt.figure(figsize=(6, 4))
        plt.scatter(costs, acc, s=40)
        for x, y, lab in zip(costs, acc, labels):
            plt.annotate(lab, (x, y), fontsize=7, alpha=0.85)
        plt.xlabel("Average cost")
        plt.ylabel("Accuracy")
        slug = rel.replace("outputs/", "").replace("/summary.json", "").replace("/", "_")
        plt.title(f"Policy routes: accuracy vs cost ({slug})")
        plt.tight_layout()
        out = out_dir / f"real_policy_accuracy_vs_cost_{slug}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        saved.append(out)
    if not saved:
        raise FigureBlocker(
            "BLOCKED real policy frontier figures: no policy eval summaries found. "
            "See docs/PAPER_ARTIFACT_GENERATION_STATUS.md for required paths."
        )
    return saved


def run_all_figure_exports(
    art: ArtifactPaths,
    out_dir: Path | None = None,
    *,
    only: set[str] | None = None,
) -> tuple[list[Path], list[tuple[str, str]]]:
    out = out_dir or art.paper_figures
    ensure_dir(out)
    written: list[Path] = []
    blockers: list[tuple[str, str]] = []

    def try_fig(name: str, fn: Any) -> None:
        if only is not None and name not in only:
            return
        try:
            paths = fn()
            written.extend(paths)
        except FigureBlocker as e:
            blockers.append((name, e.message))

    try_fig("simulated_sweep", lambda: figure_simulated_sweep(art, out / "simulated_sweep"))
    try_fig(
        "next_stage_budget",
        lambda: figure_next_stage_budget_curves(art, out / "next_stage"),
    )
    try_fig(
        "next_stage_cascade",
        lambda: figure_next_stage_cascade_curves(art, out / "next_stage"),
    )
    try_fig("real_policy", lambda: figure_real_policy_frontier(art, out / "real_routing"))

    manifest = {
        "written": [str(p.relative_to(art.root)) for p in written],
        "blockers": [{"name": n, "message": m} for n, m in blockers],
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(out / "export_manifest.json")
    return written, blockers


def print_blockers(blockers: list[tuple[str, str]], stream: Any = sys.stderr) -> None:
    for name, msg in blockers:
        print(f"[{name}] {msg}", file=stream)
