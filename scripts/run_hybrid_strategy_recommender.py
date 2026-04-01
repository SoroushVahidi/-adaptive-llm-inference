#!/usr/bin/env python3
"""Train/evaluate hybrid strategy recommender (ML + heuristics + optimizer)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.routing_hybrid.calibration import ProbabilityCalibrator  # noqa: E402
from src.routing_hybrid.eval import compute_simple_baselines, evaluate_chosen_actions  # noqa: E402
from src.routing_hybrid.features.registry import apply_feature_families  # noqa: E402
from src.routing_hybrid.heuristics.registry import apply_heuristics  # noqa: E402
from src.routing_hybrid.models.registry import make_model  # noqa: E402
from src.routing_hybrid.optimizers.registry import make_optimizer  # noqa: E402
from src.routing_hybrid.reporting import write_csv, write_json  # noqa: E402
from src.routing_hybrid.utility import compute_candidate_utility  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _select_feature_cols(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    out = [c for c in rows[0].keys() if c.startswith("hfeat_") or c.startswith("feat_")]
    return sorted(out)


def _matrix(rows: list[dict[str, Any]], feature_cols: list[str]) -> np.ndarray:
    return np.array([[float(r.get(c, 0.0)) for c in feature_cols] for r in rows], dtype=float)


def _run_variant(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_dir: Path,
    variant_name: str,
    model_name: str,
    optimizer_name: str,
    use_heuristics: bool,
) -> dict[str, Any]:
    working = [dict(r) for r in rows]
    working = apply_feature_families(working, cfg["features"]["families"])
    if use_heuristics:
        working = apply_heuristics(working, cfg["heuristics"]["rules"])

    train_rows = [r for r in working if str(r.get("split", "train")) == "train"]
    val_rows = [r for r in working if str(r.get("split", "train")) == "validation"]
    test_rows = [r for r in working if str(r.get("split", "train")) == "test"]
    if not val_rows:
        val_rows = train_rows
    feature_cols = _select_feature_cols(working)
    X_train = _matrix(train_rows, feature_cols)
    y_train = np.array([int(float(r["correctness_label"])) for r in train_rows], dtype=int)
    X_val = _matrix(val_rows, feature_cols)
    y_val = np.array([int(float(r["correctness_label"])) for r in val_rows], dtype=int)

    model = make_model(model_name, seed=int(cfg["split"].get("seed", 42)), model_params=cfg["model"].get("params", {}))
    model.fit(X_train, y_train)
    val_scores = model.predict_proba(X_val)
    calibrator = ProbabilityCalibrator(method=str(cfg["model"].get("calibration", "none")))
    calibrator.fit(val_scores, y_val)

    # Predict all rows
    X_all = _matrix(working, feature_cols)
    raw_scores = model.predict_proba(X_all)
    cal_scores = calibrator.transform(raw_scores)
    for i, r in enumerate(working):
        r["pred_raw_score"] = float(raw_scores[i])
        r["pred_p_success"] = float(cal_scores[i])
        r["pred_utility"] = float(cal_scores[i])
        r["pred_gain"] = float(cal_scores[i] - 0.5)
        r["pred_reward"] = 1.0
        r["pred_uncertainty"] = abs(0.5 - float(cal_scores[i]))

    # baseline cost per prompt for utility formulas that need deltas
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for r in working:
        by_prompt.setdefault(str(r["prompt_id"]), []).append(r)
    for pid, rows_pid in by_prompt.items():
        baseline = min(rows_pid, key=lambda x: (float(x["action_cost"]), str(x["action_name"])))
        base_cost = float(baseline["action_cost"])
        for rr in rows_pid:
            rr["baseline_cost"] = base_cost
            rr["final_utility"] = compute_candidate_utility(
                rr,
                utility_name=str(cfg["utility"]["name"]),
                lambda_cost=float(cfg["utility"].get("lambda", 1.0)),
                beta_uncertainty=float(cfg["utility"].get("beta_uncertainty", 0.0)),
            )
            if float(rr.get("heur_forbidden", 0.0)) > 0.5:
                rr["final_utility"] = -1e9

    test_candidates = [r for r in working if str(r.get("split", "train")) == "test"]
    if not test_candidates:
        test_candidates = working
    num_prompts_test = len({str(r["prompt_id"]) for r in test_candidates})
    budget = float(cfg["budget"].get("total_budget", cfg["budget"].get("avg_budget_per_prompt", 1.0) * num_prompts_test))
    optimizer = make_optimizer(optimizer_name, optimizer_params=cfg.get("optimizer", {}).get("params", {}))
    opt_result = optimizer.solve(test_candidates, budget=budget)

    chosen_eval = evaluate_chosen_actions(test_candidates, opt_result["chosen_by_prompt"])
    baselines = compute_simple_baselines(test_candidates)

    # outputs
    variant_dir = out_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    for r in test_candidates:
        out = {
            "prompt_id": r["prompt_id"],
            "regime": r["regime"],
            "action_name": r["action_name"],
            "split": r["split"],
            "correctness_label": r["correctness_label"],
            "action_cost": r["action_cost"],
            "pred_p_success": r["pred_p_success"],
            "pred_uncertainty": r["pred_uncertainty"],
            "final_utility": r["final_utility"],
        }
        pred_rows.append(out)
    write_csv(variant_dir / "predictions_per_candidate.csv", pred_rows)
    chosen_rows = []
    index = {(str(r["prompt_id"]), str(r["action_name"])): r for r in test_candidates}
    for pid, action in opt_result["chosen_by_prompt"].items():
        rr = index[(pid, action)]
        chosen_rows.append(
            {
                "prompt_id": pid,
                "regime": rr["regime"],
                "chosen_action": action,
                "correctness_label": rr["correctness_label"],
                "action_cost": rr["action_cost"],
                "final_utility": rr["final_utility"],
            }
        )
    write_csv(variant_dir / "chosen_actions.csv", chosen_rows)
    feature_importance = model.feature_importance(feature_cols)
    write_csv(
        variant_dir / "feature_importance.csv",
        [{"feature": k, "importance": v} for k, v in sorted(feature_importance.items(), key=lambda x: -x[1])[:100]],
    )
    heur_summary = {
        "rows": len(test_candidates),
        "forbidden_rows": sum(1 for r in test_candidates if float(r.get("heur_forbidden", 0.0)) > 0.5),
        "dominated_rows": sum(1 for r in test_candidates if float(r.get("heur_dominated_action", 0.0)) > 0.5),
        "avg_heuristic_adjustment": float(np.mean([float(r.get("heur_utility_adjustment", 0.0)) for r in test_candidates])),
    }
    write_csv(variant_dir / "heuristic_usage_summary.csv", [heur_summary])
    write_csv(
        variant_dir / "budget_usage_summary.csv",
        [
            {
                "optimizer": opt_result["optimizer_name"],
                "budget": opt_result["budget"],
                "total_cost": opt_result["total_cost"],
                "utilization": float(opt_result["total_cost"]) / max(1e-9, float(opt_result["budget"])),
                "objective_value": opt_result["objective_value"],
            }
        ],
    )
    metrics_row = {
        "variant": variant_name,
        "final_accuracy": chosen_eval["final_accuracy"],
        "average_cost": chosen_eval["average_cost"],
        "average_utility": chosen_eval["average_utility"],
        "budget_utilization": float(opt_result["total_cost"]) / max(1e-9, float(opt_result["budget"])),
        "optimizer": opt_result["optimizer_name"],
        "model": model_name,
        "heuristics_enabled": int(use_heuristics),
        "baseline_cheapest_accuracy": baselines["cheapest_only"]["accuracy"],
        "baseline_oracle_accuracy": baselines["oracle_upper_bound"]["accuracy"],
    }
    write_csv(variant_dir / "metrics.csv", [metrics_row])
    summary = {
        "variant": variant_name,
        "metrics": metrics_row,
        "action_distribution": chosen_eval["action_distribution"],
        "baselines": baselines,
    }
    write_json(variant_dir / "summary.json", summary)
    summary["variant_dir"] = str(variant_dir)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)

    in_csv = Path(cfg["data"]["candidate_rows_csv"])
    rows = _read_csv(in_csv)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, str, str, bool]] = []
    if cfg.get("ablation", {}).get("run_all", False):
        variants = [
            ("heuristics_only_argmax", "dummy", "per_prompt_argmax", True),
            ("ml_argmax", str(cfg["model"]["type"]), "per_prompt_argmax", False),
            ("ml_greedy_upgrade", str(cfg["model"]["type"]), "greedy_upgrade", False),
            ("ml_mckp_exact", str(cfg["model"]["type"]), "mckp_exact", False),
            ("hybrid_argmax", str(cfg["model"]["type"]), "per_prompt_argmax", True),
            ("hybrid_greedy_upgrade", str(cfg["model"]["type"]), "greedy_upgrade", True),
            ("hybrid_mckp_exact", str(cfg["model"]["type"]), "mckp_exact", True),
        ]
    else:
        variants = [
            (
                "default",
                str(cfg["model"]["type"]),
                str(cfg["optimizer"]["type"]),
                bool(cfg.get("heuristics", {}).get("enabled", True)),
            )
        ]

    summaries = []
    for name, model_name, optimizer_name, use_heur in variants:
        summaries.append(_run_variant(rows, cfg, out_dir, name, model_name, optimizer_name, use_heur))

    # Provide canonical top-level outputs for the default/first variant.
    first_dir = Path(str(summaries[0]["variant_dir"]))
    for fname in [
        "predictions_per_candidate.csv",
        "chosen_actions.csv",
        "metrics.csv",
        "feature_importance.csv",
        "heuristic_usage_summary.csv",
        "budget_usage_summary.csv",
    ]:
        src = first_dir / fname
        if src.is_file():
            shutil.copy2(src, out_dir / fname)

    write_csv(
        out_dir / "ablation_summary.csv",
        [s["metrics"] for s in summaries],
    )
    write_json(out_dir / "summary.json", {"run_status": "OK", "variants": summaries})
    print(json.dumps({"run_status": "OK", "output_dir": str(out_dir), "num_variants": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()

