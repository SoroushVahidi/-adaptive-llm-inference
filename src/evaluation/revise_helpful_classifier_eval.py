"""Offline evaluation for learned revise-helpful routing classifiers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.analysis.consistency_benchmark import (
    evaluate_benchmark,
    load_benchmark,
)
from src.features.calibration_features import extract_calibration_features
from src.features.constraint_violation_features import extract_constraint_violation_features
from src.features.number_role_features import (
    compute_calibrated_role_decision,
    compute_role_coverage_features,
)
from src.features.precompute_features import extract_query_features
from src.features.selective_prediction_features import extract_selective_prediction_features
from src.features.self_verification_features import extract_self_verification_features
from src.features.step_verification_features import extract_step_verification_features
from src.features.target_quantity_features import extract_target_quantity_features
from src.features.unified_error_signal import compute_unified_error_signal
from src.models.revise_helpful_classifier import (
    compute_binary_metrics,
    detect_sklearn_support,
    metrics_to_dict,
)


def _candidate_features(question: str, answer: str) -> dict[str, Any]:
    reasoning_text = f"Final answer: {answer}"
    parsed_answer = answer.strip()

    query_feats = extract_query_features(question)
    target_feats = extract_target_quantity_features(question)
    constraint_feats = extract_constraint_violation_features(
        question_text=question,
        reasoning_output=reasoning_text,
        predicted_answer=parsed_answer,
    )
    role_feats = compute_role_coverage_features(
        question_text=question,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    calibrated = compute_calibrated_role_decision(
        question_text=question,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    self_verify = extract_self_verification_features(
        question_text=question,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    selective = extract_selective_prediction_features(
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    calibration = extract_calibration_features(
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    step = extract_step_verification_features(
        question_text=question,
        reasoning_text=reasoning_text,
    )
    unified = compute_unified_error_signal(
        question_text=question,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )

    calibration_bin = calibration["calibration_bin"]
    calibrated_decision = calibrated["calibrated_decision"]

    return {
        **query_feats,
        **target_feats,
        **constraint_feats,
        "role_coverage_score": role_feats["role_coverage_score"],
        "missing_required_number_count": role_feats["missing_required_number_count"],
        "missing_strong_required_number_count": role_feats[
            "missing_strong_required_number_count"
        ],
        "possible_intermediate_stop_suspected": role_feats[
            "possible_intermediate_stop_suspected"
        ],
        "required_subtractive_number_missing": role_feats[
            "required_subtractive_number_missing"
        ],
        "required_rate_number_missing": role_feats["required_rate_number_missing"],
        "required_capacity_number_missing": role_feats[
            "required_capacity_number_missing"
        ],
        "role_warning_score": calibrated["role_warning_score"],
        "role_strong_error_score": calibrated["role_strong_error_score"],
        "calibrated_no_escalation": int(calibrated_decision == "no_escalation"),
        "calibrated_maybe_escalate": int(calibrated_decision == "maybe_escalate"),
        "calibrated_strong_escalation_candidate": int(
            calibrated_decision == "strong_escalation_candidate"
        ),
        **self_verify,
        **selective,
        "predicted_answer_format_confidence": calibration[
            "predicted_answer_format_confidence"
        ],
        "calibration_bin_high": int(calibration_bin == "high_confidence"),
        "calibration_bin_medium": int(calibration_bin == "medium_confidence"),
        "calibration_bin_low": int(calibration_bin == "low_confidence"),
        **step,
        "unified_error_score": unified["unified_error_score"],
        "unified_confidence_score": unified["unified_confidence_score"],
    }


def build_training_rows_from_benchmark(benchmark_path: str | Path) -> list[dict[str, Any]]:
    """Build fallback training rows from consistency benchmark candidates.

    Evidence-status: exploratory_only. We treat each candidate answer as a
    simulated reasoning_greedy first pass and the benchmark's known-correct
    answer as a simulated direct_plus_revise success.
    """
    rows = load_benchmark(benchmark_path)
    out: list[dict[str, Any]] = []

    for qrow in rows:
        question = str(qrow["question"])
        qid = str(qrow["question_id"])
        gold = str(qrow["gold_answer"])
        for idx, cand in enumerate(qrow["candidate_answers"]):
            answer = str(cand["answer"])
            is_correct = bool(cand["is_correct"])
            revise_helpful = int((not is_correct) and bool(gold))

            feats = _candidate_features(question=question, answer=answer)
            out.append(
                {
                    "question_id": qid,
                    "scenario_id": f"{qid}_cand_{idx}",
                    "question": question,
                    "reasoning_greedy_answer": answer,
                    "direct_plus_revise_answer": gold,
                    "reasoning_greedy_correct": int(is_correct),
                    "direct_plus_revise_correct": 1,
                    "revise_helpful": revise_helpful,
                    **feats,
                }
            )

    return out


def _to_float_feature(v: Any) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    text = str(v).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def _feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    protected = {
        "question_id",
        "scenario_id",
        "question",
        "reasoning_greedy_answer",
        "direct_plus_revise_answer",
        "reasoning_greedy_correct",
        "direct_plus_revise_correct",
        "revise_helpful",
    }
    sample = rows[0] if rows else {}
    out: list[str] = []
    for c in sample:
        if c in protected:
            continue
        if _to_float_feature(sample[c]) is None:
            continue
        out.append(c)
    return out


def _heuristic_predictions(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    benchmark_eval = evaluate_benchmark(load_benchmark("data/consistency_benchmark.json"))
    indexed = {
        (r["question_id"], r["candidate_answer"]): r
        for r in benchmark_eval["per_candidate_results"]
    }

    calibrated_preds: list[int] = []
    unified_preds: list[int] = []
    for row in rows:
        key = (row["question_id"], row["reasoning_greedy_answer"])
        meta = indexed.get(key)
        if meta is None:
            calibrated_preds.append(0)
            unified_preds.append(0)
            continue
        calibrated_preds.append(int(meta["calibrated_role_flagged"] == 1))
        unified_preds.append(int(meta["unified_error_flagged"] == 1))
    return calibrated_preds, unified_preds


def _simulate_routing(
    rows: list[dict[str, Any]],
    predictions: list[int],
) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "accuracy": 0.0,
            "avg_cost": 0.0,
            "revise_trigger_count": 0,
            "revise_trigger_fraction": 0.0,
        }

    correct = 0
    total_cost = 0.0
    revise_count = 0

    for row, pred in zip(rows, predictions, strict=True):
        if pred == 1:
            revise_count += 1
            total_cost += 2.0
            correct += int(row["direct_plus_revise_correct"])
        else:
            total_cost += 1.0
            correct += int(row["reasoning_greedy_correct"])

    return {
        "accuracy": correct / total,
        "avg_cost": total_cost / total,
        "revise_trigger_count": revise_count,
        "revise_trigger_fraction": revise_count / total,
    }


def _routing_baselines(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    n = len(rows)
    rg_preds = [0] * n
    dpr_preds = [1] * n
    cal_preds, uni_preds = _heuristic_predictions(rows)
    return {
        "reasoning_greedy": _simulate_routing(rows, rg_preds),
        "direct_plus_revise": _simulate_routing(rows, dpr_preds),
        "heuristic_calibrated_role": _simulate_routing(rows, cal_preds),
        "heuristic_unified_error": _simulate_routing(rows, uni_preds),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_with_sklearn(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> dict[str, Any]:
    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X = [
        [_to_float_feature(row.get(col, 0.0)) or 0.0 for col in feature_columns]
        for row in rows
    ]
    y = [int(row["revise_helpful"]) for row in rows]

    class_count = len(set(y))
    if class_count < 2:
        return {
            "run_status": "BLOCKED",
            "reason": "Need at least two classes to train classifiers.",
        }

    positive = sum(y)
    negative = len(y) - positive
    min_class = min(positive, negative)
    if min_class < 2:
        return {
            "run_status": "BLOCKED",
            "reason": "Need at least two samples per class for CV.",
        }

    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "bagging_tree": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=25,
            random_state=42,
        ),
        "adaboost_tree": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=40,
            learning_rate=0.7,
            random_state=42,
        ),
    }

    metrics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, list[int]] = {}
    fi_rows: list[dict[str, Any]] = []

    for name, model in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        pred = cross_val_predict(pipe, X, y, cv=cv)
        pred_list = [int(v) for v in pred]
        predictions[name] = pred_list
        metrics[name] = metrics_to_dict(compute_binary_metrics(y, pred_list))

        pipe.fit(X, y)
        fitted_model = pipe.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            importances = list(fitted_model.feature_importances_)
            for col, imp in sorted(
                zip(feature_columns, importances, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )[:20]:
                fi_rows.append(
                    {
                        "model": name,
                        "feature": col,
                        "importance": float(imp),
                    }
                )

    best_model = max(metrics.items(), key=lambda kv: kv[1]["f1"])[0]

    return {
        "run_status": "OK",
        "cv_splits": n_splits,
        "metrics": metrics,
        "predictions": predictions,
        "best_model": best_model,
        "feature_importance_rows": fi_rows,
    }


def run_revise_helpful_classifier_eval(
    benchmark_path: str | Path = "data/consistency_benchmark.json",
    output_dir: str | Path = "outputs/revise_helpful_classifier",
) -> dict[str, Any]:
    """Run offline learned-router evaluation and write standard artifacts."""
    rows = build_training_rows_from_benchmark(benchmark_path)
    feature_cols = _feature_columns(rows)
    y_true = [int(r["revise_helpful"]) for r in rows]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    support = detect_sklearn_support()
    routing_baselines = _routing_baselines(rows)

    per_query_rows: list[dict[str, Any]] = []
    model_metric_rows: list[dict[str, Any]] = []
    feature_importance_rows: list[dict[str, Any]] = []

    if not support.available:
        for row in rows:
            per_query_rows.append(
                {
                    "scenario_id": row["scenario_id"],
                    "question_id": row["question_id"],
                    "revise_helpful": row["revise_helpful"],
                    "decision_tree_pred": "",
                    "bagging_tree_pred": "",
                    "adaboost_tree_pred": "",
                }
            )

        summary = {
            "run_status": "BLOCKED",
            "block_reason": support.reason,
            "dataset": {
                "num_rows": len(rows),
                "num_features": len(feature_cols),
                "num_positive_labels": int(sum(y_true)),
                "num_negative_labels": int(len(y_true) - sum(y_true)),
            },
            "models": {
                "decision_tree": "blocked",
                "bagging_tree": "blocked",
                "adaboost_tree": "blocked",
            },
            "routing_simulation": {
                "reasoning_greedy": routing_baselines["reasoning_greedy"],
                "direct_plus_revise": routing_baselines["direct_plus_revise"],
                "heuristic_calibrated_role": routing_baselines[
                    "heuristic_calibrated_role"
                ],
                "heuristic_unified_error": routing_baselines["heuristic_unified_error"],
            },
            "evidence_status": "blocked",
            "claim_status": "exploratory_only",
            "dataset_evidence": "measured_now",
            "label_definition": (
                "revise_helpful=1 when simulated reasoning_greedy candidate is wrong "
                "and direct_plus_revise (gold) is correct"
            ),
        }
    else:
        trained = _evaluate_with_sklearn(rows, feature_cols)
        if trained["run_status"] != "OK":
            summary = {
                "run_status": "BLOCKED",
                "block_reason": trained["reason"],
                "dataset": {
                    "num_rows": len(rows),
                    "num_features": len(feature_cols),
                    "num_positive_labels": int(sum(y_true)),
                    "num_negative_labels": int(len(y_true) - sum(y_true)),
                },
                "models": {
                    "decision_tree": "blocked",
                    "bagging_tree": "blocked",
                    "adaboost_tree": "blocked",
                },
                "routing_simulation": routing_baselines,
                "evidence_status": "blocked",
                "claim_status": "exploratory_only",
                "dataset_evidence": "measured_now",
            }
            for row in rows:
                per_query_rows.append(
                    {
                        "scenario_id": row["scenario_id"],
                        "question_id": row["question_id"],
                        "revise_helpful": row["revise_helpful"],
                        "decision_tree_pred": "",
                        "bagging_tree_pred": "",
                        "adaboost_tree_pred": "",
                    }
                )
        else:
            metrics = trained["metrics"]
            best_model = trained["best_model"]
            preds = trained["predictions"]
            feature_importance_rows = trained["feature_importance_rows"]

            for model_name, model_metrics in metrics.items():
                model_metric_rows.append(
                    {
                        "model": model_name,
                        "accuracy": model_metrics["accuracy"],
                        "precision": model_metrics["precision"],
                        "recall": model_metrics["recall"],
                        "f1": model_metrics["f1"],
                        "false_positive_rate": model_metrics["false_positive_rate"],
                        "tp": model_metrics["confusion_matrix"]["tp"],
                        "fp": model_metrics["confusion_matrix"]["fp"],
                        "tn": model_metrics["confusion_matrix"]["tn"],
                        "fn": model_metrics["confusion_matrix"]["fn"],
                    }
                )

            for idx, row in enumerate(rows):
                per_query_rows.append(
                    {
                        "scenario_id": row["scenario_id"],
                        "question_id": row["question_id"],
                        "revise_helpful": row["revise_helpful"],
                        "decision_tree_pred": preds["decision_tree"][idx],
                        "bagging_tree_pred": preds["bagging_tree"][idx],
                        "adaboost_tree_pred": preds["adaboost_tree"][idx],
                    }
                )

            learned_routing = {
                m: _simulate_routing(rows, preds[m])
                for m in ("decision_tree", "bagging_tree", "adaboost_tree")
            }

            summary = {
                "run_status": "OK",
                "dataset": {
                    "num_rows": len(rows),
                    "num_features": len(feature_cols),
                    "num_positive_labels": int(sum(y_true)),
                    "num_negative_labels": int(len(y_true) - sum(y_true)),
                    "cv_splits": trained["cv_splits"],
                },
                "models": metrics,
                "best_model": best_model,
                "routing_simulation": {
                    **routing_baselines,
                    **learned_routing,
                },
                "evidence_status": "measured_now",
                "claim_status": "exploratory_only",
                "dataset_evidence": "measured_now",
            }

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    model_metrics_path = out / "model_metrics.csv"
    if model_metric_rows:
        _write_csv(
            model_metrics_path,
            model_metric_rows,
            fieldnames=list(model_metric_rows[0].keys()),
        )
    else:
        _write_csv(
            model_metrics_path,
            [],
            fieldnames=[
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "tp",
                "fp",
                "tn",
                "fn",
            ],
        )

    per_query_path = out / "per_query_predictions.csv"
    _write_csv(
        per_query_path,
        per_query_rows,
        fieldnames=list(per_query_rows[0].keys()) if per_query_rows else ["scenario_id"],
    )

    routing_rows = [
        {"router": k, **v} for k, v in summary["routing_simulation"].items()
    ]
    routing_path = out / "routing_simulation.csv"
    _write_csv(
        routing_path,
        routing_rows,
        fieldnames=list(routing_rows[0].keys()),
    )

    feature_importance_path = out / "feature_importance.csv"
    if feature_importance_rows:
        _write_csv(
            feature_importance_path,
            feature_importance_rows,
            fieldnames=list(feature_importance_rows[0].keys()),
        )
    else:
        _write_csv(
            feature_importance_path,
            [],
            fieldnames=["model", "feature", "importance"],
        )

    return {
        "summary_json": str(summary_path),
        "model_metrics_csv": str(model_metrics_path),
        "per_query_predictions_csv": str(per_query_path),
        "routing_simulation_csv": str(routing_path),
        "feature_importance_csv": str(feature_importance_path),
        "summary": summary,
    }
