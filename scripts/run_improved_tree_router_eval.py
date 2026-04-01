#!/usr/bin/env python3
"""Train/evaluate improved tree-based router models on hybrid candidate data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.routing_hybrid.calibration import ProbabilityCalibrator  # noqa: E402
from src.routing_hybrid.dataset_builder import _read_csv as _read_prompt_csv  # noqa: E402
from src.routing_hybrid.dataset_builder import build_candidate_rows, write_candidate_artifacts  # noqa: E402
from src.routing_hybrid.eval import compute_simple_baselines, evaluate_chosen_actions  # noqa: E402
from src.routing_hybrid.reporting import write_csv, write_json  # noqa: E402
from src.routing_hybrid.tree_router.data import (  # noqa: E402
    build_feature_matrix,
    build_targets,
    filter_rows,
    read_candidate_rows,
    transform_feature_matrix,
)
from src.routing_hybrid.tree_router.metrics import compute_prediction_metrics  # noqa: E402
from src.routing_hybrid.tree_router.models import feature_importance, make_tree_model, predict_score  # noqa: E402
from src.routing_hybrid.tree_router.selectors import assign_predicted_utility, select_actions  # noqa: E402
from src.routing_hybrid.tree_router.tuning import hyperparameter_search  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _load_or_build_candidate_csv(cfg: dict[str, Any]) -> Path:
    cand = Path(cfg["data"]["candidate_rows_csv"])
    if cand.is_file():
        return cand
    # fallback: build from prompt-level routing_ml_dataset.csv
    prompt_csv = Path(cfg["data"].get("prompt_level_dataset_csv", "data/routing_ml_dataset.csv"))
    rows = _read_prompt_csv(prompt_csv)
    candidate_rows = build_candidate_rows(rows, utility_lambdas=[float(cfg["utility"].get("lambda", 1.0))])
    out_dir = cand.parent
    write_candidate_artifacts(candidate_rows, out_dir)
    return cand


def _default_grid(model_type: str) -> dict[str, list[Any]]:
    common = {
        "max_depth": [3, 6, 10],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5],
    }
    if model_type == "decision_tree":
        return common
    if model_type in {"bagging_tree", "random_forest"}:
        g = dict(common)
        g["n_estimators"] = [100, 300]
        g["max_features"] = ["sqrt"] if model_type == "random_forest" else []
        return {k: v for k, v in g.items() if v}
    if model_type in {"gradient_boosting", "adaboost"}:
        g = dict(common)
        g["n_estimators"] = [100, 300]
        g["learning_rate"] = [0.03, 0.1]
        if model_type == "adaboost":
            g["base_max_depth"] = [1, 2]
        return g
    return {}


def _read_existing_learned_router_baseline() -> dict[str, float] | None:
    p = Path("outputs/routing_ml_eval/routing_model_comparison_test.csv")
    if not p.is_file():
        return None
    with p.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lr = next((r for r in rows if r.get("model_or_policy", "").startswith("learned_router_")), None)
    if lr is None:
        return None
    return {
        "accuracy": float(lr.get("routing_accuracy", 0.0)),
        "avg_cost": float(lr.get("avg_cost", 0.0)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = _load_or_build_candidate_csv(cfg)
    all_rows = read_candidate_rows(cand_path)
    regimes = cfg["data"].get("regimes")
    rows = filter_rows(all_rows, regimes=regimes)

    scenarios = cfg.get("training", {}).get("scenarios", ["pooled"])
    model_types = cfg.get(
        "models",
        ["decision_tree", "bagging_tree", "random_forest", "gradient_boosting", "adaboost"],
    )
    target_type = str(cfg.get("target", {}).get("type", "success_binary"))
    lambda_cost = float(cfg["utility"].get("lambda", 1.0))
    selector = str(cfg.get("downstream", {}).get("selector", "per_prompt_argmax"))
    optimizer_params = cfg.get("downstream", {}).get("optimizer_params", {})
    beta_unc = float(cfg["utility"].get("beta_uncertainty", 0.0))
    utility_name = str(cfg["utility"]["name"])
    seed = int(cfg.get("seed", 42))

    # class-balance summary
    cb_rows: list[dict[str, Any]] = []
    for reg in sorted({r["regime"] for r in rows}):
        rr = [x for x in rows if x["regime"] == reg]
        y = build_targets(rr, target_type=target_type, lambda_cost=lambda_cost)
        if target_type == "utility_regression":
            cb_rows.append({"regime": reg, "rows": len(rr), "target_mean": float(np.mean(y))})
        else:
            cb_rows.append(
                {
                    "regime": reg,
                    "rows": len(rr),
                    "pos_rate": float(np.mean(y)),
                    "num_classes": len(set(y.tolist())),
                }
            )
    write_csv(out_dir / "class_balance_summary.csv", cb_rows)

    comparison_rows: list[dict[str, Any]] = []
    hp_rows: list[dict[str, Any]] = []
    calib_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    chosen_rows_all: list[dict[str, Any]] = []
    feature_imp_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    unique_regimes = sorted({r["regime"] for r in rows})
    for scenario in scenarios:
        if scenario == "pooled":
            scenario_sets = [("pooled_main", unique_regimes)]
        elif scenario == "per_regime":
            scenario_sets = [(reg, [reg]) for reg in unique_regimes]
        elif scenario == "leave_one_regime_out":
            scenario_sets = [(f"loro_test_{reg}", [x for x in unique_regimes if x != reg]) for reg in unique_regimes]
        else:
            raise ValueError(f"Unknown training scenario '{scenario}'")

        for scenario_name, train_regimes in scenario_sets:
            if scenario == "leave_one_regime_out":
                test_regime = scenario_name.replace("loro_test_", "")
                train_rows = [r for r in rows if r["regime"] in train_regimes and r["split"] != "test"]
                val_rows = [r for r in rows if r["regime"] in train_regimes and r["split"] == "test"]
                test_rows = [r for r in rows if r["regime"] == test_regime]
            else:
                scoped = [r for r in rows if r["regime"] in train_regimes]
                train_rows = [r for r in scoped if r.get("split") == "train"]
                val_rows = [r for r in scoped if r.get("split") == "validation"]
                test_rows = [r for r in scoped if r.get("split") == "test"]
                if not val_rows:
                    val_rows = train_rows
                if not test_rows:
                    test_rows = scoped

            if not train_rows or not test_rows:
                skipped.append({"scenario": scenario_name, "reason": "empty_train_or_test"})
                continue

            X_train, feature_names, fmeta = build_feature_matrix(
                train_rows,
                include_regime_indicator=bool(cfg["features"].get("include_regime_indicator", True)),
                include_action_indicator=bool(cfg["features"].get("include_action_indicator", True)),
            )
            y_train = build_targets(train_rows, target_type=target_type, lambda_cost=lambda_cost)
            X_val = transform_feature_matrix(val_rows, fmeta)
            y_val = build_targets(val_rows, target_type=target_type, lambda_cost=lambda_cost)
            X_test = transform_feature_matrix(test_rows, fmeta)
            y_test = build_targets(test_rows, target_type=target_type, lambda_cost=lambda_cost)

            for model_type in model_types:
                if target_type != "utility_regression" and len(set(y_train.tolist())) < 2:
                    skipped.append(
                        {
                            "scenario": scenario_name,
                            "model_type": model_type,
                            "reason": "label_degeneracy_train",
                        }
                    )
                    continue

                model = None
                best_params: dict[str, Any] = {}
                if cfg.get("tuning", {}).get("enabled", True):
                    grid = cfg.get("tuning", {}).get("grid", {}).get(model_type, _default_grid(model_type))
                    model, best_params, hp = hyperparameter_search(
                        model_type=model_type,
                        task_type=target_type,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        seed=seed,
                        grid=grid,
                    )
                    for h in hp:
                        hp_rows.append({"scenario": scenario_name, **h})
                else:
                    model = make_tree_model(
                        model_type=model_type,
                        task_type=target_type,
                        seed=seed,
                        params=cfg.get("model_params", {}).get(model_type, {}),
                    )
                    model.fit(X_train, y_train)

                val_score = predict_score(model, X_val, task_type=target_type)
                test_score_raw = predict_score(model, X_test, task_type=target_type)
                calib_method = str(cfg.get("calibration", {}).get("method", "none"))
                if target_type == "utility_regression":
                    test_score = test_score_raw
                    calib_used = "none"
                else:
                    calibrator = ProbabilityCalibrator(method=calib_method)
                    calibrator.fit(np.clip(val_score, 0.0, 1.0), y_val.astype(int))
                    test_score = calibrator.transform(np.clip(test_score_raw, 0.0, 1.0))
                    calib_used = calib_method
                    val_metrics_uncal = compute_prediction_metrics(y_val.astype(int), np.clip(val_score, 0.0, 1.0), task_type=target_type)
                    val_metrics_cal = compute_prediction_metrics(y_val.astype(int), np.clip(calibrator.transform(np.clip(val_score, 0.0, 1.0)), 0.0, 1.0), task_type=target_type)
                    calib_rows.append(
                        {
                            "scenario": scenario_name,
                            "model_type": model_type,
                            "calibration": calib_used,
                            "val_brier_uncal": val_metrics_uncal.get("brier", 0.0),
                            "val_brier_cal": val_metrics_cal.get("brier", 0.0),
                            "val_logloss_uncal": val_metrics_uncal.get("log_loss", 0.0),
                            "val_logloss_cal": val_metrics_cal.get("log_loss", 0.0),
                        }
                    )

                pred_metrics = compute_prediction_metrics(y_test, test_score, task_type=target_type)
                scored_rows = assign_predicted_utility(
                    test_rows,
                    pred_score_by_index=[float(x) for x in test_score.tolist()],
                    utility_name=utility_name,
                    lambda_cost=lambda_cost,
                    beta_uncertainty=beta_unc,
                )
                num_prompts = len({r["prompt_id"] for r in scored_rows})
                budget = float(
                    cfg.get("downstream", {}).get("total_budget", cfg.get("downstream", {}).get("avg_budget_per_prompt", 1.2) * num_prompts)
                )
                sel = select_actions(scored_rows, selector=selector, budget=budget, optimizer_params=optimizer_params)
                eval_sel = evaluate_chosen_actions(scored_rows, sel["chosen_by_prompt"])
                baselines = compute_simple_baselines(scored_rows)
                existing_lr = _read_existing_learned_router_baseline()
                comparison_rows.append(
                    {
                        "scenario": scenario_name,
                        "model_type": model_type,
                        "target_type": target_type,
                        "calibration": calib_used,
                        "selector": selector,
                        "train_rows": len(train_rows),
                        "test_rows": len(test_rows),
                        "prediction_metric_primary": pred_metrics.get("roc_auc", pred_metrics.get("rmse", 0.0)),
                        "final_accuracy": eval_sel["final_accuracy"],
                        "avg_cost": eval_sel["average_cost"],
                        "avg_utility": eval_sel["average_utility"],
                        "baseline_cheapest_accuracy": baselines["cheapest_only"]["accuracy"],
                        "baseline_oracle_accuracy": baselines["oracle_upper_bound"]["accuracy"],
                        "existing_learned_router_accuracy": existing_lr["accuracy"] if existing_lr else "",
                        "existing_learned_router_avg_cost": existing_lr["avg_cost"] if existing_lr else "",
                        "best_params_json": json.dumps(best_params, sort_keys=True),
                    }
                )

                for i, (r, score) in enumerate(zip(test_rows, test_score.tolist(), strict=True)):
                    prediction_rows.append(
                        {
                            "scenario": scenario_name,
                            "model_type": model_type,
                            "prompt_id": r["prompt_id"],
                            "regime": r["regime"],
                            "action_name": r["action_name"],
                            "split": r.get("split", ""),
                            "target": float(y_test[i]) if len(y_test) > i else "",
                            "pred_score": float(score),
                        }
                    )
                idx = {(str(r["prompt_id"]), str(r["action_name"])): r for r in scored_rows}
                for pid, action in sel["chosen_by_prompt"].items():
                    rr = idx[(pid, action)]
                    chosen_rows_all.append(
                        {
                            "scenario": scenario_name,
                            "model_type": model_type,
                            "prompt_id": pid,
                            "regime": rr["regime"],
                            "chosen_action": action,
                            "correctness_label": rr["correctness_label"],
                            "action_cost": rr["action_cost"],
                            "final_utility": rr["final_utility"],
                        }
                    )
                fi = feature_importance(model, feature_names)
                feature_imp_rows.extend(
                    {
                        "scenario": scenario_name,
                        "model_type": model_type,
                        "feature": k,
                        "importance": v,
                    }
                    for k, v in sorted(fi.items(), key=lambda x: -x[1])[:100]
                )

    # write outputs
    write_csv(out_dir / "model_comparison.csv", comparison_rows)
    write_csv(out_dir / "hyperparameter_search_results.csv", hp_rows)
    write_csv(out_dir / "calibration_summary.csv", calib_rows)
    write_csv(out_dir / "per_candidate_predictions.csv", prediction_rows)
    write_csv(out_dir / "chosen_actions.csv", chosen_rows_all)
    write_csv(out_dir / "feature_importance.csv", feature_imp_rows)
    write_csv(out_dir / "metrics.csv", comparison_rows)
    summary = {
        "run_status": "OK",
        "candidate_rows_csv": str(cand_path),
        "num_rows": len(rows),
        "regimes_used": sorted({r["regime"] for r in rows}),
        "target_type": target_type,
        "models_requested": model_types,
        "num_comparisons": len(comparison_rows),
        "num_skipped": len(skipped),
        "skipped": skipped[:50],
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

