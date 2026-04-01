#!/usr/bin/env python3
"""Build a labeled routing dataset for learned method selection.

Repository-grounded source artifacts:
- data/real_*_routing_dataset_enriched.csv (four canonical regimes)

Label policy (default):
1) choose cheapest correct action among supported actions
2) if no action is correct, choose cheapest action

Supported actions are inferred from committed per-prompt outcome columns:
- reasoning_greedy            -> reasoning_correct, cost reasoning_cost (fallback 1)
- direct_plus_revise          -> revise_correct,    cost revise_cost (fallback 2)
- reasoning_then_revise       -> reasoning_then_revise_correct, cost fixed 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REGIME_FILES: dict[str, str] = {
    "gsm8k_random_100": "data/real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": "data/real_math500_routing_dataset_enriched.csv",
}

ACTION_SPECS = {
    "reasoning_greedy": {"correct_col": "reasoning_correct", "cost_col": "reasoning_cost", "default_cost": 1.0},
    "direct_plus_revise": {"correct_col": "revise_correct", "cost_col": "revise_cost", "default_cost": 2.0},
    "reasoning_then_revise": {
        "correct_col": "reasoning_then_revise_correct",
        "cost_col": None,
        "default_cost": 2.0,
    },
}

FEATURE_PREFIXES = (
    "q_",
    "tq_",
    "cons_",
    "role_",
    "cal_",
    "self_",
    "step_",
    "fp_",
    "unified_",
    "v6_",
    "v7_",
)

BASE_COLUMNS = ["prompt_id", "question", "regime", "split", "best_action_label"]


@dataclass
class BuildResult:
    dataset: pd.DataFrame
    splits: pd.DataFrame
    report: dict


def _infer_supported_actions(df: pd.DataFrame) -> list[str]:
    actions: list[str] = []
    for action, spec in ACTION_SPECS.items():
        if spec["correct_col"] in df.columns:
            actions.append(action)
    return actions


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _label_row(row: pd.Series, actions: list[str]) -> str:
    correct_actions: list[str] = []
    for action in actions:
        if int(row[f"{action}__correct"]) == 1:
            correct_actions.append(action)
    if correct_actions:
        return min(correct_actions, key=lambda a: (float(row[f"{a}__cost"]), a))
    return min(actions, key=lambda a: (float(row[f"{a}__cost"]), a))


def _split_df(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype="object")

    for _, gidx in df.groupby(["regime", "best_action_label"]).groups.items():
        idx = np.array(list(gidx))
        idx = idx.copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(np.floor(0.70 * n))
        n_val = int(np.floor(0.15 * n))
        n_test = n - n_train - n_val
        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
                n_train = max(1, n_train - 1)
            if n_test == 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val = max(1, n_val - 1)

        split.loc[idx[:n_train]] = "train"
        split.loc[idx[n_train : n_train + n_val]] = "validation"
        split.loc[idx[n_train + n_val :]] = "test"

    return split


def build_dataset(regime_files: dict[str, str]) -> BuildResult:
    frames: list[pd.DataFrame] = []
    actions_global: list[str] | None = None

    for regime, path_str in regime_files.items():
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Missing regime CSV: {path}")
        df = pd.read_csv(path)
        actions = _infer_supported_actions(df)
        if actions_global is None:
            actions_global = actions
        if actions != actions_global:
            raise ValueError(
                f"Action mismatch for {regime}: {actions} vs {actions_global}"
            )

        out = pd.DataFrame()
        out["prompt_id"] = df["question_id"].astype(str)
        out["question"] = df["question"].astype(str)
        out["regime"] = regime

        feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
        for c in feature_cols:
            out[c] = df[c]

        for action in actions:
            spec = ACTION_SPECS[action]
            out[f"{action}__correct"] = _to_numeric(df[spec["correct_col"]], 0).astype(int)
            if spec["cost_col"] is not None and spec["cost_col"] in df.columns:
                out[f"{action}__cost"] = _to_numeric(df[spec["cost_col"]], spec["default_cost"])
            else:
                out[f"{action}__cost"] = float(spec["default_cost"])

        out["best_action_label"] = out.apply(lambda r: _label_row(r, actions), axis=1)
        frames.append(out)

    dataset = pd.concat(frames, ignore_index=True)
    dataset["split"] = _split_df(dataset)

    required = set(BASE_COLUMNS)
    if not required.issubset(dataset.columns):
        raise ValueError(f"Required columns missing: {required - set(dataset.columns)}")

    if dataset["prompt_id"].isna().any() or (dataset["prompt_id"].str.len() == 0).any():
        raise ValueError("Empty prompt_id found")

    if dataset.duplicated(subset=["regime", "prompt_id"]).any():
        raise ValueError("Found duplicate prompt_id within a regime")

    if not set(dataset["best_action_label"]).issubset(set(actions_global or [])):
        raise ValueError("best_action_label has values outside supported action set")

    prompt_split_counts = dataset.groupby(["regime", "prompt_id"])["split"].nunique()
    if (prompt_split_counts > 1).any():
        raise ValueError("Split leakage detected: prompt appears in multiple splits")

    splits = dataset[["prompt_id", "regime", "split", "best_action_label"]].copy()

    report = {
        "actions_included": actions_global,
        "regimes_included": list(regime_files.keys()),
        "num_prompts_total": int(len(dataset)),
        "num_prompts_per_regime": dataset.groupby("regime").size().to_dict(),
        "label_distribution": dataset["best_action_label"].value_counts().to_dict(),
        "split_counts": dataset["split"].value_counts().to_dict(),
        "split_counts_by_regime": dataset.groupby(["regime", "split"]).size().unstack(fill_value=0).to_dict(),
        "label_policy": {
            "primary": "cheapest_correct_action",
            "fallback": "if no action correct, choose cheapest action",
            "utility_fallback_used": False,
            "notes": "Costs are available or defaulted from repo-defined call counts (reasoning=1, revise=2, reasoning_then_revise=2).",
        },
        "feature_policy": {
            "included_prefixes": list(FEATURE_PREFIXES),
            "excluded": [
                "oracle-style outcomes/correctness labels",
                "raw answer text columns",
                "gold answer column",
            ],
        },
    }

    return BuildResult(dataset=dataset, splits=splits, report=report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build routing ML dataset from committed routing artifacts")
    parser.add_argument("--dataset-csv", default="data/routing_ml_dataset.csv")
    parser.add_argument("--dataset-jsonl", default="data/routing_ml_dataset.jsonl")
    parser.add_argument("--splits-csv", default="data/routing_ml_splits.csv")
    parser.add_argument("--report-json", default="data/routing_ml_dataset_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_dataset(REGIME_FILES)
    if args.seed != 42:
        result.dataset["split"] = _split_df(result.dataset.drop(columns=["split"]), seed=args.seed)
        result.splits = result.dataset[["prompt_id", "regime", "split", "best_action_label"]].copy()
        result.report["split_counts"] = result.dataset["split"].value_counts().to_dict()

    dataset_csv = Path(args.dataset_csv)
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    result.dataset.to_csv(dataset_csv, index=False)

    dataset_jsonl = Path(args.dataset_jsonl)
    dataset_jsonl.write_text("\n".join(result.dataset.to_json(orient="records", lines=True).splitlines()) + "\n")

    splits_csv = Path(args.splits_csv)
    result.splits.to_csv(splits_csv, index=False)

    report_json = Path(args.report_json)
    report_json.write_text(json.dumps(result.report, indent=2))

    print(json.dumps(result.report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
