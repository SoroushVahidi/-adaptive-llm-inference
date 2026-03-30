"""Initial learned routing model eval on real GSM8K routing rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.models.revise_helpful_classifier import (
    compute_binary_metrics,
    detect_sklearn_support,
    metrics_to_dict,
)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_int(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    text = str(v).strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _to_float(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    text = str(v).strip()
    if not text:
        return 0.0
    return float(text)


def _feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    protected = {
        "question_id",
        "question",
        "gold_answer",
        "reasoning_answer",
        "revise_answer",
        "reasoning_correct",
        "revise_correct",
        "revise_helpful",
        "reasoning_cost",
        "revise_cost",
    }
    sample = rows[0] if rows else {}
    out: list[str] = []
    for key in sample:
        if key in protected:
            continue
        try:
            _ = _to_float(sample[key])
        except Exception:
            continue
        out.append(key)
    return out


def _simulate_routing(rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"accuracy": 0.0, "avg_cost": 0.0, "revise_rate": 0.0}

    correct = 0
    total_cost = 0.0
    revise_count = 0
    for row, pred in zip(rows, predictions, strict=True):
        if pred == 1:
            revise_count += 1
            total_cost += _to_float(row.get("revise_cost", 2))
            correct += _to_int(row["revise_correct"])
        else:
            total_cost += _to_float(row.get("reasoning_cost", 1))
            correct += _to_int(row["reasoning_correct"])

    return {
        "accuracy": correct / total,
        "avg_cost": total_cost / total,
        "revise_rate": revise_count / total,
    }


def run_real_routing_model_eval(
    dataset_csv: str | Path = "data/real_gsm8k_routing_dataset.csv",
    output_dir: str | Path = "outputs/real_routing_model",
) -> dict[str, Any]:
    dataset_path = Path(dataset_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    model_metrics_csv = out_dir / "model_metrics.csv"
    per_query_predictions_csv = out_dir / "per_query_predictions.csv"
    routing_simulation_csv = out_dir / "routing_simulation.csv"
    feature_importance_csv = out_dir / "feature_importance.csv"

    if not dataset_path.exists():
        summary = {
            "run_status": "BLOCKED",
            "evidence_status": "blocked",
            "reason": f"Missing dataset CSV: {dataset_csv}",
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        return {"summary": summary, "summary_path": str(summary_path)}

    rows = _read_csv_rows(dataset_path)
    if not rows:
        summary = {
            "run_status": "BLOCKED",
            "evidence_status": "blocked",
            "reason": "Dataset CSV has no rows.",
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        return {"summary": summary, "summary_path": str(summary_path)}

    sklearn = detect_sklearn_support()
    if not sklearn.available:
        summary = {
            "run_status": "BLOCKED",
            "evidence_status": "blocked",
            "reason": sklearn.reason,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        return {"summary": summary, "summary_path": str(summary_path)}

    y_true = [_to_int(r["revise_helpful"]) for r in rows]
    if len(set(y_true)) < 2:
        summary = {
            "run_status": "BLOCKED",
            "evidence_status": "blocked",
            "reason": "Need both classes in revise_helpful for training.",
            "num_rows": len(rows),
            "num_positive": sum(y_true),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        return {"summary": summary, "summary_path": str(summary_path)}

    feature_columns = _feature_columns(rows)
    X = [[_to_float(r.get(col, 0.0)) for col in feature_columns] for r in rows]

    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    positive = sum(y_true)
    negative = len(y_true) - positive
    n_splits = min(5, positive, negative)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "bagging_trees": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=25,
            random_state=42,
        ),
        "boosting_shallow_trees": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=60,
            random_state=42,
        ),
    }

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        preds = cross_val_predict(model, X, y_true, cv=cv, method="predict")
        metrics = compute_binary_metrics(y_true, [int(p) for p in preds])
        mdict = metrics_to_dict(metrics)
        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": mdict["accuracy"],
                "precision": mdict["precision"],
                "recall": mdict["recall"],
                "f1": mdict["f1"],
                "false_positive_rate": mdict["false_positive_rate"],
            }
        )
        for row, pred in zip(rows, preds, strict=True):
            prediction_rows.append(
                {
                    "model": model_name,
                    "question_id": row["question_id"],
                    "y_true": row["revise_helpful"],
                    "y_pred": int(pred),
                }
            )

        sim = _simulate_routing(rows, [int(p) for p in preds])
        routing_rows.append({"route": model_name, **sim})

    baseline_preds: dict[str, list[int]] = {
        "reasoning_greedy": [0] * len(rows),
        "direct_plus_revise": [1] * len(rows),
        "heuristic_calibrated_role": [
            _to_int(r.get("calibrated_strong_escalation_candidate", 0)) for r in rows
        ],
        "heuristic_unified_error": [
            int(_to_float(r.get("unified_error_score", 0.0)) >= 0.5) for r in rows
        ],
    }
    if rows and "v6_revise_recommended" in rows[0]:
        baseline_preds["heuristic_v6_revise_column"] = [
            _to_int(r.get("v6_revise_recommended", 0)) for r in rows
        ]
    if rows and "v7_revise_recommended" in rows[0]:
        baseline_preds["heuristic_v7_revise_column"] = [
            _to_int(r.get("v7_revise_recommended", 0)) for r in rows
        ]
    for name, preds in baseline_preds.items():
        routing_rows.append({"route": name, **_simulate_routing(rows, preds)})

    best_model_name = max(metrics_rows, key=lambda r: float(r["f1"]))["model"]
    # fit one final model for feature importances
    final_model = models[best_model_name]
    final_model.fit(X, y_true)
    fi = getattr(final_model, "feature_importances_", None)
    fi_rows: list[dict[str, Any]] = []
    if fi is not None:
        fi_rows = [
            {"feature": feature, "importance": float(score)}
            for feature, score in sorted(zip(feature_columns, fi, strict=True), key=lambda x: -x[1])
        ]

    _write_csv(model_metrics_csv, metrics_rows, list(metrics_rows[0].keys()))
    _write_csv(per_query_predictions_csv, prediction_rows, list(prediction_rows[0].keys()))
    _write_csv(routing_simulation_csv, routing_rows, list(routing_rows[0].keys()))
    if fi_rows:
        _write_csv(feature_importance_csv, fi_rows, list(fi_rows[0].keys()))

    summary = {
        "run_status": "OK",
        "evidence_status": "measured_now",
        "num_rows": len(rows),
        "num_positive": sum(y_true),
        "models": metrics_rows,
        "best_model": best_model_name,
        "outputs": {
            "model_metrics_csv": str(model_metrics_csv),
            "per_query_predictions_csv": str(per_query_predictions_csv),
            "routing_simulation_csv": str(routing_simulation_csv),
            "feature_importance_csv": str(feature_importance_csv),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "summary_path": str(summary_path)}
