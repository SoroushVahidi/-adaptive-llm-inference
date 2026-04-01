#!/usr/bin/env python3
"""Train lightweight learned routers on routing ML dataset splits."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


ACTIONS = ["reasoning_greedy", "direct_plus_revise"]


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


def _to_matrix(rows: list[dict[str, str]], feature_cols: list[str]) -> np.ndarray:
    return np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=float)


def _pick_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    # maximize val routing accuracy, then minimize val avg_cost
    return sorted(
        results,
        key=lambda r: (-float(r["val_routing_accuracy"]), float(r["val_avg_cost"])),
    )[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-csv", default="data/routing_ml_dataset.csv")
    p.add_argument("--output-dir", default="outputs/routing_ml_models")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dataset = Path(args.dataset_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(dataset)
    if not rows:
        raise RuntimeError("Routing ML dataset is empty.")

    feature_cols = [c for c in rows[0].keys() if c.startswith("feat_")]
    if not feature_cols:
        raise RuntimeError("No feature columns found (expected feat_*).")

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError("Missing one or more splits (train/validation/test).")

    label_to_id = {a: i for i, a in enumerate(ACTIONS)}
    id_to_label = {i: a for a, i in label_to_id.items()}

    X_train = _to_matrix(train_rows, feature_cols)
    y_train = np.array([label_to_id[r["best_action_label"]] for r in train_rows], dtype=int)
    X_val = _to_matrix(val_rows, feature_cols)
    y_val = np.array([label_to_id[r["best_action_label"]] for r in val_rows], dtype=int)

    candidates: list[tuple[str, Any, dict[str, Any]]] = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
            {"max_iter": 2000, "class_weight": "balanced"},
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=args.seed),
            {"max_depth": 5, "min_samples_leaf": 3, "class_weight": "balanced"},
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=args.seed,
            ),
            {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2, "class_weight": "balanced"},
        ),
    ]

    results: list[dict[str, Any]] = []
    model_paths: dict[str, str] = {}
    for name, model, cfg in candidates:
        model.fit(X_train, y_train)
        val_pred_id = model.predict(X_val)
        val_pred = [id_to_label[int(i)] for i in val_pred_id]
        val_true = [r["best_action_label"] for r in val_rows]
        cls_acc = float(accuracy_score(val_true, val_pred))
        macro_f1 = float(f1_score(val_true, val_pred, average="macro"))
        rmetrics = _routing_metrics(val_rows, val_pred)

        model_file = out_dir / f"{name}.pkl"
        with model_file.open("wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols, "id_to_label": id_to_label}, f)
        model_paths[name] = str(model_file)

        results.append(
            {
                "model_name": name,
                "config": cfg,
                "val_classification_accuracy": cls_acc,
                "val_macro_f1": macro_f1,
                "val_routing_accuracy": rmetrics["routing_accuracy"],
                "val_avg_cost": rmetrics["avg_cost"],
                "val_revise_rate": rmetrics["revise_rate"],
                "model_path": str(model_file),
            }
        )

    best = _pick_best(results)

    # Save validation model comparison
    val_csv = out_dir / "model_validation_results.csv"
    with val_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    summary = {
        "run_status": "OK",
        "dataset_csv": str(dataset),
        "num_rows": len(rows),
        "feature_count": len(feature_cols),
        "split_counts": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "models_trained": [r["model_name"] for r in results],
        "selection_rule": "maximize val_routing_accuracy then minimize val_avg_cost",
        "best_model": best["model_name"],
        "best_model_path": best["model_path"],
        "validation_results_csv": str(val_csv),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
