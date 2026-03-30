"""Build manuscript CSV tables from existing artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from src.evaluation.analysis_summary import (
    build_budget_summary_table,
    build_noise_summary_table,
    load_budget_comparisons,
    load_noise_comparisons,
    write_csv_table,
)

from .io_util import copy_file, ensure_dir, load_json, write_csv_rows
from .paths import REAL_POLICY_SUMMARY_FILES, ArtifactPaths


class Blocker(Exception):
    """Missing or invalid input; message is user-facing."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def export_simulated_sweep_tables(art: ArtifactPaths, out_dir: Path) -> list[Path]:
    base = art.simulated_sweep
    budget_csv = base / "budget_sweep_comparisons.csv"
    noise_csv = base / "noise_sensitivity_comparisons.csv"
    if not budget_csv.is_file():
        raise Blocker(
            "BLOCKED simulated allocation summary tables: missing "
            f"{budget_csv.relative_to(art.root)}. Regenerate with: "
            "python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml"
        )
    if not noise_csv.is_file():
        raise Blocker(
            "BLOCKED simulated allocation summary tables: missing "
            f"{noise_csv.relative_to(art.root)} (noise sweep disabled or incomplete). "
            "Regenerate with the same simulated sweep config ensuring noise.enabled is true."
        )
    budget_rows = load_budget_comparisons(budget_csv)
    noise_rows = load_noise_comparisons(noise_csv)
    budget_table = build_budget_summary_table(budget_rows)
    noise_table = build_noise_summary_table(noise_rows)
    ensure_dir(out_dir)
    p1 = write_csv_table(budget_table, out_dir / "simulated_sweep_budget_summary.csv")
    p2 = write_csv_table(noise_table, out_dir / "simulated_sweep_noise_summary.csv")
    return [p1, p2]


def export_baseline_json_rollups(art: ArtifactPaths, out_dir: Path) -> list[Path]:
    """One flattened CSV per *_{baseline_summary}.json when present."""
    base = art.baselines
    if not base.is_dir():
        raise Blocker(
            "BLOCKED baseline rollup tables: directory "
            f"{base.relative_to(art.root)} does not exist. Regenerate with: "
            "python3 scripts/run_strong_baselines.py --config configs/strong_baselines_real.yaml"
        )
    paths = sorted(base.glob("*_baseline_summary.json"))
    if not paths:
        raise Blocker(
            "BLOCKED baseline rollup tables: no *_baseline_summary.json under "
            f"{base.relative_to(art.root)}. Regenerate with: "
            "python3 scripts/run_strong_baselines.py --config configs/strong_baselines_dummy.yaml"
        )
    written: list[Path] = []
    ensure_dir(out_dir)
    for jp in paths:
        data = load_json(jp)
        rows: list[dict[str, str]] = []
        dataset = str(data.get("dataset", jp.stem.replace("_baseline_summary", "")))
        strategies = data.get("strategies") or {}
        if isinstance(strategies, dict):
            for strat_name, metrics in strategies.items():
                if not isinstance(metrics, dict):
                    continue
                row: dict[str, str] = {
                    "dataset": dataset,
                    "strategy": strat_name,
                }
                for mk, mv in metrics.items():
                    row[str(mk)] = str(mv)
                rows.append(row)
        if not rows:
            continue
        fieldnames = sorted({k for r in rows for k in r})
        out = out_dir / f"baselines_{dataset}_strategies.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        written.append(out)
    if not written:
        raise Blocker(
            "BLOCKED baseline rollup tables: JSON files present but no strategy rows "
            "could be parsed."
        )
    return written


def export_cross_regime_table(art: ArtifactPaths, out_dir: Path) -> Path:
    src_csv = art.cross_regime / "cross_regime_summary.csv"
    src_json = art.cross_regime / "cross_regime_summary.json"
    if src_csv.is_file():
        dest = out_dir / "cross_regime_summary.csv"
        copy_file(src_csv, dest)
        return dest
    if src_json.is_file():
        rows = load_json(src_json)
        if not isinstance(rows, list) or not rows:
            rel = src_json.relative_to(art.root)
            raise Blocker(
                f"BLOCKED cross-regime table: {rel} is not a non-empty list."
            )
        dest = out_dir / "cross_regime_summary.csv"
        write_csv_rows(rows, dest)
        return dest
    raise Blocker(
        "BLOCKED cross-regime table: missing "
        f"{src_csv.relative_to(art.root)} and {src_json.relative_to(art.root)}. "
        "Regenerate with: python3 scripts/run_cross_regime_comparison.py"
    )


def export_final_cross_regime_table(art: ArtifactPaths, out_dir: Path) -> Path:
    src = art.cross_regime / "final_cross_regime_summary.csv"
    if not src.is_file():
        raise Blocker(
            "BLOCKED final cross-regime summary: missing "
            f"{src.relative_to(art.root)}. "
            "Regenerate with: python3 scripts/run_final_cross_regime_summary.py"
        )
    dest = out_dir / "final_cross_regime_summary.csv"
    copy_file(src, dest)
    return dest


def export_oracle_routing_summaries(art: ArtifactPaths, out_dir: Path) -> list[Path]:
    d = art.oracle_routing_eval
    if not d.is_dir():
        raise Blocker(
            "BLOCKED oracle routing eval table: directory "
            f"{d.relative_to(art.root)} missing. "
            "Regenerate via scripts/run_next_stage_postprocess.py (writes oracle summaries)."
        )
    files = sorted(d.glob("*_oracle_summary.json"))
    if not files:
        rel_d = d.relative_to(art.root)
        raise Blocker(
            f"BLOCKED oracle routing eval table: no *_oracle_summary.json under {rel_d}."
        )
    rows: list[dict[str, Any]] = []
    for p in files:
        j = load_json(p)
        row = dict(j)
        row["source_file"] = str(p.relative_to(art.root))
        rows.append(row)
    ensure_dir(out_dir)
    dest = out_dir / "oracle_routing_eval_summaries.csv"
    write_csv_rows(rows, dest)
    return [dest]


def export_oracle_subset_table(art: ArtifactPaths, out_dir: Path) -> list[Path]:
    """Prefer summary.csv (one row per strategy); copy summary.json for traceability."""
    d = art.oracle_subset_eval
    summary_csv = d / "summary.csv"
    summary_json = d / "summary.json"
    if not summary_json.is_file():
        raise Blocker(
            "BLOCKED oracle subset table: missing "
            f"{summary_json.relative_to(art.root)}. Regenerate with: "
            "python3 scripts/run_oracle_subset_eval.py "
            "--config configs/oracle_subset_eval_gsm8k.yaml"
        )
    payload = load_json(summary_json)
    if payload.get("run_status") == "BLOCKED":
        bt = payload.get("blocker_type", "unknown")
        rel = summary_json.relative_to(art.root)
        raise Blocker(
            f"BLOCKED oracle subset table: {rel} has run_status=BLOCKED ({bt}). "
            "Complete a successful oracle subset run first."
        )
    written: list[Path] = []
    ensure_dir(out_dir)
    if summary_csv.is_file():
        dest = out_dir / "oracle_subset_strategy_accuracy.csv"
        copy_file(summary_csv, dest)
        written.append(dest)
    else:
        strat = payload.get("strategy_accuracy") or {}
        strat_rows: list[dict[str, str]] = []
        if isinstance(strat, dict):
            for name, metrics in strat.items():
                if isinstance(metrics, dict):
                    r = {"strategy": name, **{k: str(v) for k, v in metrics.items()}}
                    strat_rows.append(r)
        if strat_rows:
            write_csv_rows(strat_rows, out_dir / "oracle_subset_strategy_accuracy.csv")
            written.append(out_dir / "oracle_subset_strategy_accuracy.csv")
    meta = {
        "total_queries": payload.get("total_queries"),
        "oracle_accuracy": payload.get("oracle_accuracy"),
        "direct_accuracy": payload.get("direct_accuracy"),
        "oracle_minus_direct_gap": payload.get("oracle_minus_direct_gap"),
        "strategies_run": json.dumps(payload.get("strategies_run", [])),
    }
    headline_path = out_dir / "oracle_subset_headline_metrics.csv"
    write_csv_rows([{k: str(v) for k, v in meta.items()}], headline_path)
    written.append(out_dir / "oracle_subset_headline_metrics.csv")
    return written


def export_real_policy_comparison_table(art: ArtifactPaths, out_dir: Path) -> Path:
    """Long-form accuracy/cost rows from real policy eval summaries."""
    blocks: list[dict[str, Any]] = []
    for rel in REAL_POLICY_SUMMARY_FILES:
        p = art.root / rel
        if not p.is_file():
            continue
        data = load_json(p)
        comp = data.get("comparison")
        if not isinstance(comp, list):
            continue
        label = rel.replace("outputs/", "").replace("/summary.json", "")
        for row in comp:
            if not isinstance(row, dict):
                continue
            blocks.append(
                {
                    "eval_summary": label,
                    "route": row.get("route", ""),
                    "accuracy": row.get("accuracy", ""),
                    "avg_cost": row.get("avg_cost", ""),
                    "revise_rate": row.get("revise_rate", ""),
                }
            )
    if not blocks:
        missing = "\n  ".join(REAL_POLICY_SUMMARY_FILES)
        raise Blocker(
            "BLOCKED real policy comparison table: no readable policy eval summaries. "
            "Expected at least one of:\n  " + missing + "\n"
            "Regenerate e.g. with: python3 scripts/run_real_policy_eval.py "
            "--output-dir outputs/real_policy_eval"
        )
    dest = out_dir / "real_policy_eval_comparison_long.csv"
    write_csv_rows(blocks, dest)
    return dest


def export_next_stage_budget_curves_merged(art: ArtifactPaths, out_dir: Path) -> Path:
    """Single CSV with dataset_key column from outputs/next_stage_eval/*/budget_curve.csv."""
    base = art.next_stage_eval
    if not base.is_dir():
        raise Blocker(
            "BLOCKED merged budget curves: missing "
            f"{base.relative_to(art.root)}. "
            "Regenerate with scripts/run_next_stage_postprocess.py per dataset."
        )
    all_rows: list[dict[str, str]] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        curve = sub / "budget_curve.csv"
        if not curve.is_file():
            continue
        with curve.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                r = dict(row)
                r["dataset_key"] = sub.name
                all_rows.append(r)
    if not all_rows:
        raise Blocker(
            "BLOCKED merged budget curves: no budget_curve.csv under "
            f"{base.relative_to(art.root)}/*/ . "
            "Run run_next_stage_postprocess.py for each dataset."
        )
    keys0 = list(all_rows[0].keys())
    fieldnames = ["dataset_key"] + [k for k in keys0 if k != "dataset_key"]
    ensure_dir(out_dir)
    dest = out_dir / "next_stage_budget_curves_all_datasets.csv"
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    return dest


def run_all_table_exports(
    art: ArtifactPaths,
    out_dir: Path | None = None,
    *,
    only: set[str] | None = None,
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Returns (written paths, blockers as (name, message))."""
    out = out_dir or art.paper_tables
    ensure_dir(out)
    written: list[Path] = []
    blockers: list[tuple[str, str]] = []

    def try_export(name: str, fn: Any) -> None:
        if only is not None and name not in only:
            return
        try:
            result = fn()
            if isinstance(result, list):
                written.extend(result)
            else:
                written.append(result)
        except Blocker as e:
            blockers.append((name, e.message))
        except FileNotFoundError as e:
            blockers.append((name, f"BLOCKED {name}: {e}"))

    try_export(
        "simulated_sweep",
        lambda: export_simulated_sweep_tables(art, out / "simulated_sweep"),
    )
    try_export(
        "baselines",
        lambda: export_baseline_json_rollups(art, out / "baselines"),
    )
    try_export(
        "cross_regime",
        lambda: export_cross_regime_table(art, out / "cross_regime"),
    )
    try_export(
        "final_cross_regime",
        lambda: export_final_cross_regime_table(art, out / "cross_regime"),
    )
    try_export(
        "oracle_routing",
        lambda: export_oracle_routing_summaries(art, out / "oracle_routing"),
    )
    try_export(
        "oracle_subset",
        lambda: export_oracle_subset_table(art, out / "oracle_subset"),
    )
    try_export(
        "real_policy_comparison",
        lambda: export_real_policy_comparison_table(art, out / "real_routing"),
    )
    try_export(
        "next_stage_budget_curves",
        lambda: export_next_stage_budget_curves_merged(art, out / "next_stage"),
    )

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
