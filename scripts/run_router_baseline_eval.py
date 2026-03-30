#!/usr/bin/env python3
"""Evaluate sklearn router baselines on an oracle-labelled routing dataset CSV.

This is the follow-up to ``run_build_real_routing_dataset.py`` when that build
produces ``routing_dataset.csv`` with oracle columns (``oracle_label_available``,
``direct_already_optimal``, ``best_accuracy_strategy``).

For **revise_helpful** tree ensembles and ``routing_simulation.csv`` on a
**paired-outcomes** CSV (e.g. ``data/real_gsm8k_routing_dataset.csv``), use
``scripts/run_real_routing_model_eval.py`` instead.

Prerequisites
-------------
- Routing CSV from ``run_build_real_routing_dataset.py`` (default path below).
- Oracle labels on rows (``oracle_label_available = True``).
- ``scikit-learn`` recommended for tree/logistic models; majority baseline runs without it.

Usage
-----
    python3 scripts/run_router_baseline_eval.py

    python3 scripts/run_router_baseline_eval.py \\
        --routing-csv outputs/real_routing_dataset/routing_dataset.csv

    python3 scripts/run_router_baseline_eval.py \\
        --output-dir outputs/real_router_eval

Outputs
-------
``outputs/real_router_eval/`` (default)
    ├── router_eval_results.json
    ├── binary_results.csv
    └── multiclass_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.policies.router_baseline import (  # noqa: E402
    QUERY_FEATURE_COLS,
    EvalResult,
    _check_sklearn,
    fit_and_evaluate,
    load_routing_csv,
    prepare_features,
    save_router_outputs,
)

_DEFAULT_ROUTING_CSV = Path("outputs/real_routing_dataset/routing_dataset.csv")
_DEFAULT_OUTPUT_DIR = Path("outputs/real_router_eval")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sklearn router baselines (oracle labels) on routing_dataset.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--routing-csv",
        default=None,
        metavar="PATH",
        help=(
            f"Path to routing dataset CSV with oracle columns "
            f"(default: {_DEFAULT_ROUTING_CSV})"
        ),
    )
    parser.add_argument(
        "--dataset-csv",
        default=None,
        metavar="PATH",
        help="Alias for --routing-csv (backward compatibility).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    raw_csv = args.routing_csv or args.dataset_csv or str(_DEFAULT_ROUTING_CSV)
    routing_csv_path = Path(raw_csv)
    output_dir = Path(args.output_dir)

    if not routing_csv_path.exists():
        print(
            f"\n[BLOCKED] Routing dataset CSV not found:\n"
            f"  {routing_csv_path}\n"
            "\n"
            "  Build with:\n"
            "    python3 scripts/run_build_real_routing_dataset.py --subset-size 100\n"
            "\n"
            "  Requires OPENAI_API_KEY.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Loading routing dataset: {routing_csv_path}")
    try:
        rows, oracle_available = load_routing_csv(routing_csv_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(rows)} rows. Oracle labels available: {oracle_available}")

    if not oracle_available:
        print(
            "\n[BLOCKED] Oracle labels are not available on this CSV.\n"
            "  For paired-outcomes / revise_helpful evaluation use:\n"
            "    python3 scripts/run_real_routing_model_eval.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    labeled_rows = [
        r for r in rows
        if str(r.get("oracle_label_available", "")).lower() in ("true", "1")
    ]
    print(f"[INFO] Rows with oracle labels: {len(labeled_rows)} / {len(rows)}")

    if not labeled_rows:
        print(
            "\n[BLOCKED] No rows have oracle_label_available = True.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] sklearn available: {_check_sklearn()}")

    X, feature_names = prepare_features(labeled_rows, QUERY_FEATURE_COLS)
    question_ids = [r.get("question_id", "") for r in labeled_rows]

    output_dir.mkdir(parents=True, exist_ok=True)

    y_binary = [str(r.get("direct_already_optimal", "")) for r in labeled_rows]
    valid_binary_mask = [v in ("0", "1") for v in y_binary]
    y_binary_clean = [v for v, ok in zip(y_binary, valid_binary_mask) if ok]
    X_binary = [X[i] for i, ok in enumerate(valid_binary_mask) if ok]
    ids_binary = [question_ids[i] for i, ok in enumerate(valid_binary_mask) if ok]

    binary_results: list[EvalResult] = []
    if X_binary and y_binary_clean:
        print("\n[Task: binary — direct_already_optimal]")
        binary_results = fit_and_evaluate(
            X_binary, y_binary_clean, "binary", feature_names, ids_binary
        )
        for r in binary_results:
            print(f"  {r.model_name}: accuracy={r.accuracy:.4f}  n_test={r.n_test}")
    else:
        print("[WARN] No valid rows for binary task (direct_already_optimal).")

    y_multi = [r.get("best_accuracy_strategy", "") for r in labeled_rows]
    valid_multi_mask = [bool(v) for v in y_multi]
    y_multi_clean = [y_multi[i] for i, ok in enumerate(valid_multi_mask) if ok]
    X_multi = [X[i] for i, ok in enumerate(valid_multi_mask) if ok]
    ids_multi = [question_ids[i] for i, ok in enumerate(valid_multi_mask) if ok]

    multi_results: list[EvalResult] = []
    if X_multi and y_multi_clean:
        print("\n[Task: multiclass — best_accuracy_strategy]")
        multi_results = fit_and_evaluate(
            X_multi, y_multi_clean, "multiclass", feature_names, ids_multi
        )
        for r in multi_results:
            print(f"  {r.model_name}: accuracy={r.accuracy:.4f}  n_test={r.n_test}")
    else:
        print("[WARN] No valid rows for multiclass task (best_accuracy_strategy).")

    paths = save_router_outputs(binary_results, multi_results, str(output_dir))

    for results in (binary_results, multi_results):
        for r in results:
            if r.feature_importances:
                top = sorted(
                    r.feature_importances.items(), key=lambda kv: kv[1], reverse=True
                )[:5]
                print(f"\n  Top features ({r.model_name}, {r.task}):")
                for feat, imp in top:
                    print(f"    {feat}: {imp:.4f}")

    all_results: dict[str, Any] = {
        "routing_csv": str(routing_csv_path),
        "total_rows": len(rows),
        "labeled_rows": len(labeled_rows),
        "binary_task": {
            "label": "direct_already_optimal",
            "n_labeled": len(X_binary),
            "results": [
                {
                    "model": r.model_name,
                    "accuracy": r.accuracy,
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                    "feature_importances": r.feature_importances or {},
                }
                for r in binary_results
            ],
        },
        "multiclass_task": {
            "label": "best_accuracy_strategy",
            "n_labeled": len(X_multi),
            "results": [
                {
                    "model": r.model_name,
                    "accuracy": r.accuracy,
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                }
                for r in multi_results
            ],
        },
        "router_baseline_paths": paths,
    }
    results_path = output_dir / "router_eval_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))

    summary_data: dict[str, Any] = json.loads(Path(paths["summary"]).read_text())
    print("\n=== Router baseline eval complete ===")
    print(f"  Routing CSV         : {routing_csv_path}")
    print(f"  Labeled rows        : {len(labeled_rows)}")
    print(f"  sklearn available   : {summary_data['sklearn_available']}")
    print(f"  Output directory    : {output_dir}")
    print(f"  Results JSON        : {results_path}")
    print(f"  Router summary      : {paths['summary']}")


if __name__ == "__main__":
    main()
