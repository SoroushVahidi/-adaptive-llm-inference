#!/usr/bin/env python3
"""Build a per-prompt learned-routing dataset from committed enriched CSVs.

No API calls. Uses canonical four main regimes.
Label policy: cheapest correct action among {reasoning_greedy, direct_plus_revise}.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


REGIME_TO_CSV = {
    "gsm8k_random_100": "data/real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": "data/real_math500_routing_dataset_enriched.csv",
}

ACTION_SPACE = ["reasoning_greedy", "direct_plus_revise"]

NON_FEATURE_COLS = {
    "question_id",
    "question",
    "gold_answer",
    "status",
    "reasoning_raw",
    "reasoning_answer",
    "revise_answer",
    "reasoning_then_revise_answer",
    "reasoning_correct",
    "revise_correct",
    "reasoning_then_revise_correct",
    "revise_helpful",
    "reasoning_then_revise_helpful",
    "reasoning_cost",
    "revise_cost",
    "reasoning_raw_chars",
    "revise_num_calls",
}

# Explicit leakage guard: never train on oracle/action outcome columns.
LEAKAGE_PREFIXES = ("correct", "revise_helpful", "oracle", "cost_", "policy_")
LEAKAGE_CONTAINS = (
    "reasoning_correct",
    "revise_correct",
    "reasoning_then_revise_correct",
    "revise_helpful",
    "cost",
    "policy_",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_int(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    s = str(v).strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _to_float(v: Any) -> float:
    s = str(v).strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _is_feature_col(col: str) -> bool:
    if col in NON_FEATURE_COLS:
        return False
    c = col.lower()
    if c.startswith(LEAKAGE_PREFIXES):
        return False
    if any(t in c for t in LEAKAGE_CONTAINS):
        return False
    return True


def _label_row(row: dict[str, str]) -> tuple[str, str]:
    rg = _to_int(row.get("reasoning_correct", 0))
    rv = _to_int(row.get("revise_correct", 0))
    if rg == 1:
        return "reasoning_greedy", "cheapest_correct"
    if rv == 1:
        return "direct_plus_revise", "cheapest_correct"
    return "reasoning_greedy", "fallback_no_correct"


def _assign_splits(rows: list[dict[str, Any]], seed: int) -> None:
    """Assign split in place with prompt-level stratification by regime+label."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(rows):
        key = (str(r["regime"]), str(r["best_action_label"]))
        buckets.setdefault(key, []).append(i)

    for key, idxs in buckets.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(0.7 * n))
        n_val = int(round(0.15 * n))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        if n >= 3 and n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
            n_test = n - n_train - n_val
        if n >= 3 and n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)
            n_val = n - n_train - n_test

        for j, idx in enumerate(idxs):
            if j < n_train:
                rows[idx]["split"] = "train"
            elif j < n_train + n_val:
                rows[idx]["split"] = "validation"
            else:
                rows[idx]["split"] = "test"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-csv", default="data/routing_ml_dataset.csv")
    p.add_argument("--metadata-json", default="data/routing_ml_dataset_metadata.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_csv = Path(args.output_csv)
    out_meta = Path(args.metadata_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    feature_cols: list[str] = []
    seen_ids: set[tuple[str, str]] = set()

    for regime, rel in REGIME_TO_CSV.items():
        path = Path(rel)
        if not path.is_file():
            raise FileNotFoundError(f"Missing regime CSV: {rel}")
        rows = _read_csv(path)
        if not rows:
            raise RuntimeError(f"No rows in {rel}")
        if not feature_cols:
            feature_cols = [c for c in rows[0].keys() if _is_feature_col(c)]

        for r in rows:
            qid = str(r.get("question_id", "")).strip()
            if not qid:
                raise ValueError(f"Missing question_id in {rel}")
            key = (regime, qid)
            if key in seen_ids:
                raise ValueError(f"Duplicate prompt key: {key}")
            seen_ids.add(key)

            label, label_source = _label_row(r)
            row: dict[str, Any] = {
                "question_id": qid,
                "question": r.get("question", ""),
                "regime": regime,
                "action_reasoning_greedy_correct": _to_int(r.get("reasoning_correct", 0)),
                "action_direct_plus_revise_correct": _to_int(r.get("revise_correct", 0)),
                "action_reasoning_greedy_cost": _to_float(r.get("reasoning_cost", 1.0)),
                "action_direct_plus_revise_cost": _to_float(r.get("revise_cost", 2.0)),
                "best_action_label": label,
                "label_policy": "cheapest_correct_then_fallback_reasoning",
                "label_source": label_source,
            }
            for c in feature_cols:
                row[f"feat_{c}"] = _to_float(r.get(c, 0.0))
            all_rows.append(row)

    _assign_splits(all_rows, seed=args.seed)

    # Validation checks
    for r in all_rows:
        if r["best_action_label"] not in ACTION_SPACE:
            raise ValueError(f"Invalid label: {r['best_action_label']}")
        if r.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Missing/invalid split assignment")

    fieldnames = list(all_rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    split_counts = {"train": 0, "validation": 0, "test": 0}
    label_counts: dict[str, int] = {}
    for r in all_rows:
        split_counts[str(r["split"])] += 1
        label_counts[str(r["best_action_label"])] = label_counts.get(str(r["best_action_label"]), 0) + 1

    meta = {
        "run_status": "OK",
        "evidence_status": "measured_now",
        "num_rows": len(all_rows),
        "num_feature_cols": len([c for c in fieldnames if c.startswith("feat_")]),
        "action_space": ACTION_SPACE,
        "label_policy": "cheapest_correct_then_fallback_reasoning",
        "canonical_regimes": list(REGIME_TO_CSV.keys()),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "output_csv": str(out_csv),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
