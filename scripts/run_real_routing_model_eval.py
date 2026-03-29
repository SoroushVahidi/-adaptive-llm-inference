#!/usr/bin/env python3
"""Evaluate routing models on the real GSM8K routing dataset.

This is the follow-up script to ``run_build_real_routing_dataset.py``.
It loads the routing dataset CSV produced by that script and trains/evaluates
the router baseline models (majority baseline, decision tree, logistic
regression) on the oracle-labelled rows.

Prerequisites
-------------
- The real routing dataset CSV must exist (produced by
  ``run_build_real_routing_dataset.py --subset-size 100``).
- Oracle labels must be available (``oracle_label_available = True`` for at
  least some rows in the CSV).
- ``scikit-learn`` is recommended but not required — the majority-class
  baseline runs without it.

Usage
-----
    # Default paths (output of run_build_real_routing_dataset.py)
    python3 scripts/run_real_routing_model_eval.py

    # Custom routing CSV (--routing-csv or the legacy --dataset-csv alias)
    python3 scripts/run_real_routing_model_eval.py \\
        --routing-csv outputs/real_routing_dataset/routing_dataset.csv

    # Custom output directory
    python3 scripts/run_real_routing_model_eval.py \\
        --output-dir outputs/real_router_eval

Outputs
-------
``outputs/real_router_eval/``
    ├── router_eval_results.json  — all model results and accuracies
    ├── binary_results.csv        — per-model binary task results
    └── multiclass_results.csv    — per-model multiclass task results
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ROUTING_CSV = Path("outputs/real_routing_dataset/routing_dataset.csv")
_DEFAULT_OUTPUT_DIR = Path("outputs/real_router_eval")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate routing models (decision tree, logistic regression) on "
            "the real GSM8K routing dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--routing-csv",
        default=None,
        metavar="PATH",
        help=(
            f"Path to the real routing dataset CSV "
            f"(default: {_DEFAULT_ROUTING_CSV})"
        ),
    )
    # Legacy alias from origin/main — accepted for backward compatibility.
    parser.add_argument(
        "--dataset-csv",
        default=None,
        metavar="PATH",
        help=(
            "Alias for --routing-csv (for backward compatibility). "
            "If both are provided, --routing-csv takes precedence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help=f"Output directory for evaluation results (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        metavar="N",
        help="Max depth for the decision tree (default: 3)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Resolve CSV path: --routing-csv wins; fall back to --dataset-csv, then default.
    raw_csv = args.routing_csv or args.dataset_csv or str(_DEFAULT_ROUTING_CSV)
    routing_csv_path = Path(raw_csv)
    output_dir = Path(args.output_dir)

    # --- Guard: routing CSV must exist ---
    if not routing_csv_path.exists():
        print(
            f"\n[BLOCKED] Routing dataset CSV not found:\n"
            f"  {routing_csv_path}\n"
            "\n"
            "  To build the real routing dataset, run:\n"
            "    python3 scripts/run_build_real_routing_dataset.py --subset-size 100\n"
            "\n"
            "  That script requires OPENAI_API_KEY to be set.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load routing CSV ---
    print(f"[INFO] Loading routing dataset: {routing_csv_path}")
    try:
        rows, oracle_available = load_routing_csv(routing_csv_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(rows)} rows. Oracle labels available: {oracle_available}")

    if not oracle_available:
        print(
            "\n[BLOCKED] Routing dataset loaded but oracle labels are not available.\n"
            "  Every row has oracle_label_available = False.\n"
            "\n"
            "  Oracle labels are required to train and evaluate the routing model.\n"
            "  To produce oracle labels, run:\n"
            "    python3 scripts/run_build_real_routing_dataset.py --subset-size 100\n"
            "  (requires OPENAI_API_KEY)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Filter to rows with oracle labels
    labeled_rows = [
        r for r in rows
        if str(r.get("oracle_label_available", "")).lower() in ("true", "1")
    ]
    print(f"[INFO] Rows with oracle labels: {len(labeled_rows)} / {len(rows)}")

    if not labeled_rows:
        print(
            "\n[BLOCKED] No rows have oracle_label_available = True.\n"
            "  Rebuild the routing dataset with:\n"
            "    python3 scripts/run_build_real_routing_dataset.py --subset-size 100\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] sklearn available: {_check_sklearn()}")

    X, feature_names = prepare_features(labeled_rows, QUERY_FEATURE_COLS)
    question_ids = [r.get("question_id", "") for r in labeled_rows]

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Binary task: direct_already_optimal ---
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

    # --- Multiclass task: best_accuracy_strategy ---
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

    # --- Save outputs ---
    paths = save_router_outputs(binary_results, multi_results, str(output_dir))

    # --- Feature importance summary ---
    for results in (binary_results, multi_results):
        for r in results:
            if r.feature_importances:
                top = sorted(
                    r.feature_importances.items(), key=lambda kv: kv[1], reverse=True
                )[:5]
                print(f"\n  Top features ({r.model_name}, {r.task}):")
                for feat, imp in top:
                    print(f"    {feat}: {imp:.4f}")

    # --- Save extended results JSON ---
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

    # --- Print summary ---
    summary_data: dict[str, Any] = json.loads(Path(paths["summary"]).read_text())
    print("\n=== Real Routing Model Eval Complete ===")
    print(f"  Routing CSV         : {routing_csv_path}")
    print(f"  Labeled rows        : {len(labeled_rows)}")
    print(f"  sklearn available   : {summary_data['sklearn_available']}")
    print(f"  Output directory    : {output_dir}")
    print(f"  Results JSON        : {results_path}")
    print(f"  Router summary      : {paths['summary']}")


if __name__ == "__main__":
    main()
