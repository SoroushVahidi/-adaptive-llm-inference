#!/usr/bin/env python3
"""Evaluate trained learned router on held-out test split vs baselines."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


REGIME_ORDER = ["gsm8k_random_100", "hard_gsm8k_100", "hard_gsm8k_b2", "math500_100"]
POLICY_DECISION_PATHS = {
    "gsm8k_random_100": "outputs/real_policy_eval/per_query_policy_decisions.csv",
    "hard_gsm8k_100": "outputs/real_hard_gsm8k_policy_eval/per_query_policy_decisions.csv",
    "hard_gsm8k_b2": "outputs/real_hard_gsm8k_b2_policy_eval/per_query_policy_decisions.csv",
    "math500_100": "outputs/real_math500_policy_eval/per_query_policy_decisions.csv",
}
ORACLE_PATHS = {
    "gsm8k_random_100": "outputs/oracle_routing_eval/gsm8k_random100_oracle_summary.json",
    "hard_gsm8k_100": "outputs/oracle_routing_eval/hard_gsm8k_100_oracle_summary.json",
    "hard_gsm8k_b2": "outputs/oracle_routing_eval/hard_gsm8k_b2_oracle_summary.json",
    "math500_100": "outputs/oracle_routing_eval/math500_100_oracle_summary.json",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _routing_metrics(rows: list[dict[str, str]], preds: list[str]) -> dict[str, float]:
    correct = 0
    total_cost = 0.0
    revise = 0
    for r, p in zip(rows, preds, strict=True):
        if p == "direct_plus_revise":
            correct += int(float(r["action_direct_plus_revise_correct"]))
            total_cost += float(r["action_direct_plus_revise_cost"])
            revise += 1
        else:
            correct += int(float(r["action_reasoning_greedy_correct"]))
            total_cost += float(r["action_reasoning_greedy_cost"])
    n = max(1, len(rows))
    return {
        "routing_accuracy": correct / n,
        "avg_cost": total_cost / n,
        "revise_rate": revise / n,
    }


def _oracle_action(r: dict[str, str]) -> str:
    rg = int(float(r["action_reasoning_greedy_correct"]))
    rv = int(float(r["action_direct_plus_revise_correct"]))
    if rg == 1:
        return "reasoning_greedy"
    if rv == 1:
        return "direct_plus_revise"
    return "reasoning_greedy"


def _load_policy_v5_actions() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for regime, rel in POLICY_DECISION_PATHS.items():
        rows = _read_csv(Path(rel))
        for r in rows:
            out[(regime, r["question_id"])] = r["policy_v5"]
    return out


def _load_confidence_thresholds(path: Path) -> dict[str, float]:
    rows = _read_csv(path)
    return {r["regime"]: float(r["threshold"]) for r in rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-csv", default="data/routing_ml_dataset.csv")
    p.add_argument("--models-dir", default="outputs/routing_ml_models")
    p.add_argument("--output-dir", default="outputs/routing_ml_eval")
    args = p.parse_args()

    dataset = Path(args.dataset_csv)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(dataset)
    test_rows = [r for r in rows if r["split"] == "test"]
    if not test_rows:
        raise RuntimeError("No test rows found.")

    train_summary = json.loads((models_dir / "training_summary.json").read_text(encoding="utf-8"))
    best_model_name = train_summary["best_model"]
    model_blob = pickle.loads((models_dir / f"{best_model_name}.pkl").read_bytes())
    model = model_blob["model"]
    feature_cols = model_blob["feature_cols"]
    id_to_label = {int(k): v for k, v in model_blob["id_to_label"].items()} if isinstance(next(iter(model_blob["id_to_label"].keys())), str) else model_blob["id_to_label"]

    X_test = np.array([[float(r[c]) for c in feature_cols] for r in test_rows], dtype=float)
    y_true = [r["best_action_label"] for r in test_rows]
    pred_ids = model.predict(X_test)
    y_pred = [id_to_label[int(i)] for i in pred_ids]

    learned_cls_acc = float(accuracy_score(y_true, y_pred))
    learned_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=["reasoning_greedy", "direct_plus_revise"]).tolist()
    learned_metrics = _routing_metrics(test_rows, y_pred)

    # Baselines
    policy_v5 = _load_policy_v5_actions()
    conf_thr = _load_confidence_thresholds(Path("outputs/baselines/confidence_threshold/confidence_threshold_summary.csv"))

    baseline_rows: list[dict[str, Any]] = []
    baseline_preds: dict[str, list[str]] = {
        "always_cheap": ["reasoning_greedy"] * len(test_rows),
        "always_revise": ["direct_plus_revise"] * len(test_rows),
        "oracle": [_oracle_action(r) for r in test_rows],
        "adaptive_policy_v5": [
            policy_v5.get((r["regime"], r["question_id"]), "reasoning_greedy")
            for r in test_rows
        ],
        "confidence_threshold": [
            "direct_plus_revise"
            if float(r.get("feat_unified_confidence_score", 1.0)) < conf_thr[r["regime"]]
            else "reasoning_greedy"
            for r in test_rows
        ],
    }

    for name, preds in baseline_preds.items():
        cls_acc = float(accuracy_score(y_true, preds))
        macro_f1 = float(f1_score(y_true, preds, average="macro"))
        rm = _routing_metrics(test_rows, preds)
        baseline_rows.append(
            {
                "model_or_policy": name,
                "classification_accuracy": cls_acc,
                "macro_f1": macro_f1,
                "routing_accuracy": rm["routing_accuracy"],
                "avg_cost": rm["avg_cost"],
                "revise_rate": rm["revise_rate"],
                "oracle_gap": baseline_rows[0]["routing_accuracy"] if False else "",
            }
        )

    # Add previous learned-router baseline summary as reference (not held-out comparable)
    prior_path = Path("outputs/baselines/learned_router/learned_router_summary.csv")
    prior_rows = _read_csv(prior_path) if prior_path.is_file() else []
    prior_note = "Unavailable"
    if prior_rows:
        prior_note = "Available (cross-val on full regimes; not directly comparable to held-out split)"

    # Oracle gap field now that oracle routing accuracy is known on this test split
    oracle_acc = next(r["routing_accuracy"] for r in baseline_rows if r["model_or_policy"] == "oracle")
    for r in baseline_rows:
        r["oracle_gap"] = float(oracle_acc) - float(r["routing_accuracy"])

    learned_row = {
        "model_or_policy": f"learned_router_{best_model_name}",
        "classification_accuracy": learned_cls_acc,
        "macro_f1": learned_macro_f1,
        "routing_accuracy": learned_metrics["routing_accuracy"],
        "avg_cost": learned_metrics["avg_cost"],
        "revise_rate": learned_metrics["revise_rate"],
        "oracle_gap": float(oracle_acc) - float(learned_metrics["routing_accuracy"]),
    }

    # Save per-query decisions
    per_query_path = out_dir / "per_query_predictions_test.csv"
    with per_query_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "regime",
            "question_id",
            "split",
            "best_action_label",
            "predicted_action",
            "learned_correct_action_match",
            "action_reasoning_greedy_correct",
            "action_direct_plus_revise_correct",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, p_ in zip(test_rows, y_pred, strict=True):
            w.writerow(
                {
                    "regime": r["regime"],
                    "question_id": r["question_id"],
                    "split": r["split"],
                    "best_action_label": r["best_action_label"],
                    "predicted_action": p_,
                    "learned_correct_action_match": int(p_ == r["best_action_label"]),
                    "action_reasoning_greedy_correct": r["action_reasoning_greedy_correct"],
                    "action_direct_plus_revise_correct": r["action_direct_plus_revise_correct"],
                }
            )

    # Save comparison table
    comp_path = out_dir / "routing_model_comparison_test.csv"
    rows_out = [learned_row] + baseline_rows
    with comp_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "run_status": "OK",
        "evidence_status": "measured_now",
        "dataset_csv": str(dataset),
        "split_sizes": {
            "train": train_summary["split_counts"]["train"],
            "validation": train_summary["split_counts"]["validation"],
            "test": len(test_rows),
        },
        "best_model": best_model_name,
        "best_model_test": learned_row,
        "confusion_matrix_labels": ["reasoning_greedy", "direct_plus_revise"],
        "confusion_matrix": cm,
        "baselines_test": baseline_rows,
        "prior_learned_router_baseline_note": prior_note,
        "per_query_predictions_test_csv": str(per_query_path),
        "comparison_csv": str(comp_path),
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
