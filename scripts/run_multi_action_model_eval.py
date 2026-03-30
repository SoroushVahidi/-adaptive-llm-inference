#!/usr/bin/env python3
"""Train multi-action routers and simulate deployment policies.

Reads: data/multi_action_routing_<dataset>.csv (or --csv)
Writes:
  outputs/multi_action_models/<dataset>_model_results.json
  outputs/multi_action_models/<dataset>_policy_simulation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    BaggingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.evaluation.multi_action_routing import LAMBDA_VALUES, MULTI_ACTION_ORDER  # noqa: E402

ACTIONS = list(MULTI_ACTION_ORDER)
LABEL_TARGETS = (
    "best_accuracy_action",
    "best_utility_action_lambda_0_10",
    "best_utility_action_lambda_0_25",
)


def _feature_columns(fieldnames: list[str]) -> list[str]:
    return [c for c in fieldnames if c.startswith(("qf__", "fp__"))]


def _load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames


def _to_float_matrix(rows: list[dict], cols: list[str]) -> np.ndarray:
    x = np.zeros((len(rows), len(cols)), dtype=float)
    for i, row in enumerate(rows):
        for j, c in enumerate(cols):
            v = row.get(c, "")
            if v is True or v == "True":
                x[i, j] = 1.0
            elif v is False or v == "False":
                x[i, j] = 0.0
            else:
                try:
                    x[i, j] = float(v) if v != "" else 0.0
                except ValueError:
                    x[i, j] = 0.0
    return x


def _policy_metrics(chosen: str, row: dict[str, str]) -> dict[str, float]:
    c = int(row[f"{chosen}__correct"])
    cost = float(row[f"{chosen}__cost"])
    out: dict[str, float] = {"correct": float(c), "cost": cost}
    for lam in LAMBDA_VALUES:
        suffix = f"{lam:.2f}".replace(".", "_")
        out[f"utility_{suffix}"] = float(c) - lam * cost
    return out


def _oracle_regret(
    row: dict[str, str],
    policy_correct: int,
    policy_util: dict[str, float],
) -> dict[str, float]:
    oa = row["best_accuracy_action"]
    oc = int(row[f"{oa}__correct"])
    reg_acc = float(oc - policy_correct)
    out: dict[str, float] = {"regret_best_accuracy": reg_acc}
    for lam in LAMBDA_VALUES:
        if lam == 0.0:
            continue
        suffix = f"{lam:.2f}".replace(".", "_")
        ok = row[f"best_utility_action_lambda_{suffix}"]
        ou = float(row[f"{ok}__correct"]) - lam * float(row[f"{ok}__cost"])
        out[f"regret_utility_{suffix}"] = ou - policy_util[f"utility_{suffix}"]
    return out


def _simulate_block(
    label_target: str,
    test_rows: list[dict[str, str]],
    pred_by_model: dict[str, np.ndarray],
    le: LabelEncoder,
) -> list[dict[str, Any]]:
    """Policies for one label target on a fixed test set."""
    n = len(test_rows)
    sim: list[dict[str, Any]] = []

    def append_row(policy: str, actions: list[str]) -> None:
        for i, row in enumerate(test_rows):
            a = actions[i]
            m = _policy_metrics(a, row)
            reg = _oracle_regret(row, int(m["correct"]), m)
            sim.append({
                "label_target": label_target,
                "query_id": row["query_id"],
                "policy": policy,
                "chosen_action": a,
                **{k: float(v) for k, v in m.items()},
                **{k: float(v) for k, v in reg.items()},
            })

    append_row("oracle_best_accuracy", [r["best_accuracy_action"] for r in test_rows])
    append_row(
        "oracle_best_utility_lambda_0_10",
        [r["best_utility_action_lambda_0_10"] for r in test_rows],
    )
    for action in ACTIONS:
        append_row(f"always_{action}", [action] * n)
    for mname, preds in pred_by_model.items():
        chosen = [le.classes_[int(p)] for p in preds]
        append_row(f"clf__{mname}", chosen)
    return sim


def _aggregate_policy(sim: list[dict[str, Any]]) -> dict[str, Any]:
    by_pol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sim:
        by_pol[str(r["policy"])].append(r)
    out: dict[str, Any] = {}
    for pol, rs in by_pol.items():
        out[pol] = {
            "n": len(rs),
            "mean_accuracy": float(np.mean([r["correct"] for r in rs])),
            "mean_cost": float(np.mean([r["cost"] for r in rs])),
            "mean_utility_lambda_0_00": float(np.mean([r["utility_0_00"] for r in rs])),
            "mean_utility_lambda_0_10": float(np.mean([r["utility_0_10"] for r in rs])),
            "mean_utility_lambda_0_25": float(np.mean([r["utility_0_25"] for r in rs])),
            "mean_regret_best_accuracy": float(np.mean([r["regret_best_accuracy"] for r in rs])),
            "mean_regret_utility_0_10": float(np.mean([r["regret_utility_0_10"] for r in rs])),
            "mean_regret_utility_0_25": float(np.mean([r["regret_utility_0_25"] for r in rs])),
            "action_counts": {a: sum(1 for r in rs if r["chosen_action"] == a) for a in ACTIONS},
        }
    return out


def _build_models(random_state: int) -> dict[str, Any]:
    return {
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
        ),
        "bagging_tree": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=6,
                class_weight="balanced",
                random_state=random_state,
            ),
            n_estimators=15,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=80,
            random_state=random_state,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def _classifier_metrics(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    feature_names: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    bacc = balanced_accuracy_score(y_test, pred)
    labels = np.arange(len(le.classes_))
    cm = confusion_matrix(y_test, pred, labels=labels)
    report = classification_report(
        y_test,
        pred,
        labels=labels,
        target_names=list(le.classes_),
        zero_division=0,
        output_dict=True,
    )
    importances: dict[str, Any] | None = None
    top_splits: list[dict[str, Any]] = []
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        pairs = sorted(
            zip(feature_names, imps, strict=False),
            key=lambda t: t[1],
            reverse=True,
        )
        importances = {
            "feature": [p[0] for p in pairs[:30]],
            "importance": [float(p[1]) for p in pairs[:30]],
        }
    if hasattr(model, "tree_"):
        tree = model.tree_
        fn = feature_names

        def walk(node: int, depth: int) -> None:
            if depth > 3 or tree.children_left[node] == tree.children_right[node]:
                return
            fid = int(tree.feature[node])
            thr = float(tree.threshold[node])
            fname = fn[fid] if 0 <= fid < len(fn) else str(fid)
            top_splits.append({"feature": fname, "threshold": thr, "depth": depth})
            walk(int(tree.children_left[node]), depth + 1)
            walk(int(tree.children_right[node]), depth + 1)

        walk(0, 0)

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(bacc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importance_top": importances,
        "tree_top_splits": top_splits[:12],
    }, pred


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-samples-for-holdout", type=int, default=40)
    args = parser.parse_args(argv)

    path = Path(args.csv)
    if not path.exists():
        print(f"[BLOCKED] CSV not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows, fieldnames = _load_csv(path)
    feat_cols = _feature_columns(fieldnames)
    if not feat_cols:
        print("[BLOCKED] No qf__ / fp__ feature columns in CSV.", file=sys.stderr)
        sys.exit(1)

    slug = args.dataset_name or path.stem.replace("multi_action_routing_", "")
    x_all = _to_float_matrix(rows, feat_cols)
    idx_all = np.arange(len(rows))

    use_holdout = len(rows) >= args.min_samples_for_holdout
    results_root: dict[str, Any] = {
        "dataset": slug,
        "csv": str(path.resolve()),
        "num_samples": len(rows),
        "feature_columns": feat_cols,
        "evaluation_mode": (
            "train_test_split" if use_holdout else f"stratified_cv_{args.cv_folds}_folds"
        ),
        "targets": {},
    }

    models_template = _build_models(args.random_state)
    out_dir = _REPO_ROOT / "outputs" / "multi_action_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_policy_rows: list[dict[str, Any]] = []

    for target in LABEL_TARGETS:
        y_raw = [row.get(target, "") for row in rows]
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        target_entry: dict[str, Any] = {
            "label": target,
            "classes": list(le.classes_),
        }
        if len(le.classes_) < 2:
            target_entry["run_status"] = "SKIPPED"
            target_entry["skip_reason"] = (
                "Single class only (oracle tie-break is degenerate on this slice). "
                "Increase sample size or use queries where actions disagree."
            )
            test_rows = rows
            sim = _simulate_block(target, test_rows, {}, le)
            all_policy_rows.extend(sim)
            target_entry["policy_aggregate_full_data"] = _aggregate_policy(sim)
            results_root["targets"][target] = target_entry
            print(
                f"[WARN] {target}: only one class {list(le.classes_)} — "
                "skipping classifiers; baseline policies still logged.",
                file=sys.stderr,
            )
            continue

        if use_holdout:
            strat_ok = len(np.unique(y)) > 1
            _, class_counts = np.unique(y, return_counts=True)
            # Stratified split requires ≥2 samples per class in the full set.
            if strat_ok and int(class_counts.min()) < 2:
                strat_ok = False
            x_tr, x_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
                x_all,
                y,
                idx_all,
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=y if strat_ok else None,
            )
            test_rows = [rows[int(i)] for i in idx_te]
            per_model: dict[str, Any] = {}
            pred_by_model: dict[str, np.ndarray] = {}
            for mname, tmpl in models_template.items():
                clf = clone(tmpl)
                metrics, pred = _classifier_metrics(
                    clf, x_tr, y_tr, x_te, y_te, le, feat_cols
                )
                per_model[mname] = metrics
                pred_by_model[mname] = pred
            target_entry["models"] = per_model
            bc = np.bincount(y, minlength=len(le.classes_))
            target_entry["class_counts_full_dataset"] = {
                str(le.classes_[i]): int(bc[i]) for i in range(len(le.classes_))
            }
            target_entry["train_test_stratified"] = strat_ok
            target_entry["test_size"] = len(test_rows)
            sim = _simulate_block(target, test_rows, pred_by_model, le)
            all_policy_rows.extend(sim)
            target_entry["policy_aggregate_test"] = _aggregate_policy(sim)
        else:
            n_splits = min(args.cv_folds, max(2, len(rows) // 2))
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=args.random_state,
            )
            fold_summaries: list[dict[str, Any]] = []
            agg_policy_counters: dict[str, list[float]] = defaultdict(list)

            for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(x_all, y)):
                x_tr, x_te = x_all[tr_idx], x_all[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                test_rows = [rows[int(i)] for i in te_idx]
                per_model: dict[str, Any] = {}
                pred_by_model: dict[str, np.ndarray] = {}
                for mname, tmpl in models_template.items():
                    clf = clone(tmpl)
                    if hasattr(clf, "random_state"):
                        clf.set_params(random_state=args.random_state + fold_idx)
                    if mname == "bagging_tree":
                        est = DecisionTreeClassifier(
                            max_depth=6,
                            class_weight="balanced",
                            random_state=args.random_state + fold_idx,
                        )
                        clf.set_params(estimator=est)
                    metrics, pred = _classifier_metrics(
                        clf, x_tr, y_tr, x_te, y_te, le, feat_cols
                    )
                    per_model[mname] = metrics
                    pred_by_model[mname] = pred
                sim = _simulate_block(target, test_rows, pred_by_model, le)
                all_policy_rows.extend(sim)
                pol_agg = _aggregate_policy(sim)
                fold_summaries.append(
                    {"fold": fold_idx, "policy_aggregate": pol_agg, "models": per_model}
                )
                for pol, stats in pol_agg.items():
                    agg_policy_counters[f"{pol}__acc"].append(stats["mean_accuracy"])
                    agg_policy_counters[f"{pol}__cost"].append(stats["mean_cost"])

            target_entry["cv_folds"] = fold_summaries
            target_entry["policy_aggregate_cv_mean"] = {
                k.replace("__acc", ""): float(np.mean(v))
                for k, v in agg_policy_counters.items()
                if k.endswith("__acc")
            }

        results_root["targets"][target] = target_entry

    pol_path = out_dir / f"{slug}_policy_simulation.csv"
    if all_policy_rows:
        keys = list(all_policy_rows[0].keys())
        with pol_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(all_policy_rows)

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    json_path = out_dir / f"{slug}_model_results.json"
    json_path.write_text(json.dumps(_clean(results_root), indent=2))
    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {pol_path}")


if __name__ == "__main__":
    main()
