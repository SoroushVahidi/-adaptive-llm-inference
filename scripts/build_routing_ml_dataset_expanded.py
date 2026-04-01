#!/usr/bin/env python3
"""Build expanded routing-ML dataset (same protocol, larger if artifacts exist)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_routing_ml_dataset import build_dataset

# Priority order: expanded files if present, else canonical committed files.
REGIME_SOURCE_CANDIDATES: dict[str, list[str]] = {
    "gsm8k_random_100": [
        "data/real_gsm8k_routing_dataset_expanded_enriched.csv",
        "data/real_gsm8k_routing_dataset_expanded.csv",
        "data/real_gsm8k_routing_dataset_enriched.csv",
    ],
    "hard_gsm8k_100": [
        "data/real_hard_gsm8k_routing_dataset_expanded_enriched.csv",
        "data/real_hard_gsm8k_routing_dataset_expanded.csv",
        "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    ],
    "hard_gsm8k_b2": [
        "data/real_hard_gsm8k_b2_routing_dataset_expanded_enriched.csv",
        "data/real_hard_gsm8k_b2_routing_dataset_expanded.csv",
        "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    ],
    "math500_100": [
        "data/real_math500_routing_dataset_expanded_enriched.csv",
        "data/real_math500_routing_dataset_expanded.csv",
        "data/real_math500_routing_dataset_enriched.csv",
    ],
}


def _resolve_sources() -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    blockers: list[str] = []
    for regime, candidates in REGIME_SOURCE_CANDIDATES.items():
        chosen = None
        for c in candidates:
            if Path(c).exists():
                chosen = c
                break
        if chosen is None:
            blockers.append(f"{regime}: no source CSV found in candidates {candidates}")
        else:
            resolved[regime] = chosen
    return resolved, blockers


def _check_complete_action_outcomes(df: pd.DataFrame, actions: list[str]) -> list[str]:
    issues: list[str] = []
    for action in actions:
        cc = f"{action}__correct"
        kc = f"{action}__cost"
        if cc not in df.columns:
            issues.append(f"missing column {cc}")
            continue
        if df[cc].isna().any():
            issues.append(f"NaN values in {cc}")
        if not set(pd.unique(df[cc])).issubset({0, 1}):
            issues.append(f"non-binary values in {cc}")
        if kc not in df.columns:
            issues.append(f"missing column {kc}")
            continue
        if df[kc].isna().any():
            issues.append(f"NaN values in {kc}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", default="data/routing_ml_dataset_expanded.csv")
    parser.add_argument("--dataset-jsonl", default="data/routing_ml_dataset_expanded.jsonl")
    parser.add_argument("--splits-csv", default="data/routing_ml_splits_expanded.csv")
    parser.add_argument("--summary-json", default="data/routing_ml_dataset_expanded_summary.json")
    parser.add_argument("--old-summary", default="data/routing_ml_dataset_summary.json")
    parser.add_argument("--target-per-regime", type=int, default=300)
    args = parser.parse_args()

    regime_files, blockers = _resolve_sources()
    if len(regime_files) != 4:
        raise RuntimeError(f"Cannot resolve all 4 regimes. Blockers: {blockers}")

    result = build_dataset(regime_files)
    df = result.dataset.copy()

    actions = result.report["actions_included"]
    completeness_issues = _check_complete_action_outcomes(df, actions)
    if completeness_issues:
        raise ValueError("; ".join(completeness_issues))

    Path(args.dataset_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.dataset_csv, index=False)
    Path(args.dataset_jsonl).write_text(df.to_json(orient="records", lines=True) + "\n")
    result.splits.to_csv(args.splits_csv, index=False)

    old_size = None
    old_path = Path(args.old_summary)
    if old_path.exists():
        old = json.loads(old_path.read_text())
        old_size = int(old.get("num_prompts_total", 0))

    by_regime = df.groupby("regime").size().to_dict()
    expansion_blockers: list[str] = []
    for regime, n in by_regime.items():
        if n < args.target_per_regime:
            expansion_blockers.append(
                f"{regime}: only {n} prompts with complete 3-action outcomes available from resolved sources"
            )

    summary = {
        "old_num_prompts_total": old_size,
        "new_num_prompts_total": int(len(df)),
        "delta_prompts_total": None if old_size is None else int(len(df) - old_size),
        "counts_per_regime": by_regime,
        "label_distribution": df["best_action_label"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "actions_included": actions,
        "action_coverage_complete": len(completeness_issues) == 0,
        "splits_regenerated": True,
        "source_files_by_regime": regime_files,
        "source_resolution_blockers": blockers,
        "expansion_blockers": expansion_blockers,
        "target_per_regime": args.target_per_regime,
        "enough_for_meaningful_learned_router": bool(len(df) >= 800),
        "notes": [
            "To exceed current size, new routing outcomes must be generated with repository build scripts requiring model/API access.",
            "This run preserves the same 3-action set and label policy as the baseline builder.",
        ],
    }

    Path(args.summary_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
