"""CLI script: run the router baseline on the routing dataset.

Usage
-----
    python3 scripts/run_router_baseline.py [OPTIONS]

Options
-------
--routing-csv PATH
    Path to routing_dataset.csv.
    Defaults to outputs/routing_dataset/routing_dataset.csv.

--oracle-dir PATH
    Oracle eval output directory, used to (re)build the routing dataset
    when the CSV is missing.
    Defaults to outputs/oracle_subset_eval.

--output-dir PATH
    Directory where baseline outputs will be written.
    Defaults to outputs/router_baseline.

--max-depth INT
    Max depth for decision tree (default: 3).

Examples
--------
# Run with existing routing dataset
python3 scripts/run_router_baseline.py

# Custom paths
python3 scripts/run_router_baseline.py \\
    --routing-csv outputs/routing_dataset/routing_dataset.csv \\
    --output-dir outputs/router_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.routing_dataset import (  # noqa: E402
    DEFAULT_ORACLE_DIR,
    assemble_routing_dataset,
    load_oracle_files,
    save_routing_dataset,
)
from src.policies.router_baseline import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROUTING_CSV,
    QUERY_FEATURE_COLS,
    EvalResult,
    _check_sklearn,
    fit_and_evaluate,
    load_routing_csv,
    prepare_features,
    save_router_outputs,
)

_BUNDLED = _REPO_ROOT / "src" / "datasets" / "bundled" / "gsm8k_test_sample.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run router baseline on the routing dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--routing-csv",
        default=str(DEFAULT_ROUTING_CSV),
        help=f"Path to routing_dataset.csv (default: {DEFAULT_ROUTING_CSV})",
    )
    parser.add_argument(
        "--oracle-dir",
        default=str(DEFAULT_ORACLE_DIR),
        help=(
            f"Oracle eval directory used when routing CSV is missing "
            f"(default: {DEFAULT_ORACLE_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max depth for decision tree (default: 3)",
    )
    return parser.parse_args(argv)


def _try_build_routing_csv(oracle_dir: str, routing_csv_path: Path) -> bool:
    """Try to build routing_dataset.csv from oracle outputs.

    Returns True if successfully built, False otherwise.
    """
    oracle_data = load_oracle_files(oracle_dir)
    if not oracle_data.available:
        return False

    # Load queries from bundled sample
    try:
        from src.datasets.gsm8k import load_gsm8k  # noqa: PLC0415

        try:
            queries = load_gsm8k(split="test")
        except Exception:  # noqa: BLE001
            if _BUNDLED.exists():
                queries = load_gsm8k(split="test", data_file=_BUNDLED)
            else:
                return False
    except Exception:  # noqa: BLE001
        return False

    rows = assemble_routing_dataset(queries, oracle_data=oracle_data)
    save_routing_dataset(rows, routing_csv_path.parent, oracle_data=oracle_data)
    return True


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    routing_csv_path = Path(args.routing_csv)

    # --- Ensure routing CSV exists ---
    if not routing_csv_path.exists():
        print(
            f"[INFO] {routing_csv_path} not found — attempting to build from oracle outputs…"
        )
        built = _try_build_routing_csv(args.oracle_dir, routing_csv_path)
        if not built:
            print(
                "\n[BLOCKED] Cannot run router baseline.\n"
                "  The routing dataset CSV does not exist and oracle labels are "
                "unavailable.\n"
                "  To unblock:\n"
                "    1. Run the oracle evaluation:\n"
                "         python3 scripts/run_oracle_subset_eval.py "
                "--config configs/oracle_subset_eval_gsm8k.yaml\n"
                "    2. Then build the routing dataset:\n"
                "         python3 scripts/build_routing_dataset.py\n"
                "    3. Then re-run this script.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[INFO] Routing dataset built: {routing_csv_path}")

    # --- Load routing dataset ---
    try:
        rows, oracle_available = load_routing_csv(routing_csv_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if not oracle_available:
        print(
            "\n[BLOCKED] Routing dataset loaded but oracle labels are not available.\n"
            "  Every row has oracle_label_available = False.\n"
            "  To unblock, run the oracle evaluation first and rebuild the "
            "routing dataset.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Loaded {len(rows)} rows from {routing_csv_path}")
    print(f"[INFO] sklearn available: {_check_sklearn()}")

    # Filter rows with oracle labels
    labeled_rows = [
        r for r in rows
        if str(r.get("oracle_label_available", "")).lower() in ("true", "1")
    ]
    print(f"[INFO] Rows with oracle labels: {len(labeled_rows)}")

    X, feature_names = prepare_features(labeled_rows, QUERY_FEATURE_COLS)
    question_ids = [r.get("question_id", "") for r in labeled_rows]

    # --- Binary task: direct_already_optimal ---
    y_binary = [str(r.get("direct_already_optimal", "")) for r in labeled_rows]
    y_binary_clean = [v for v in y_binary if v in ("0", "1")]
    if len(y_binary_clean) < len(y_binary):
        print(
            f"[WARN] {len(y_binary) - len(y_binary_clean)} rows have invalid "
            "direct_already_optimal values and will be excluded."
        )
    valid_binary_mask = [v in ("0", "1") for v in y_binary]
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
        print("[WARN] No valid rows for binary task.")

    # --- Multiclass task: best_accuracy_strategy ---
    y_multi = [r.get("best_accuracy_strategy", "") for r in labeled_rows]
    valid_multi_mask = [bool(v) for v in y_multi]
    X_multi = [X[i] for i, ok in enumerate(valid_multi_mask) if ok]
    y_multi_clean = [y_multi[i] for i, ok in enumerate(valid_multi_mask) if ok]
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
        print("[WARN] No valid rows for multiclass task.")

    # --- Save outputs ---
    paths = save_router_outputs(binary_results, multi_results, args.output_dir)

    # Print feature importance summary
    for results in (binary_results, multi_results):
        for r in results:
            if r.feature_importances:
                top = sorted(
                    r.feature_importances.items(), key=lambda kv: kv[1], reverse=True
                )[:5]
                print(f"\n  Top features ({r.model_name}, {r.task}):")
                for feat, imp in top:
                    print(f"    {feat}: {imp:.4f}")

    # Print summary
    summary = json.loads(Path(paths["summary"]).read_text())
    print("\n=== Router Baseline Complete ===")
    print(f"  Outputs: {args.output_dir}")
    print(f"  Summary: {paths['summary']}")
    print(f"  sklearn available: {summary['sklearn_available']}")


if __name__ == "__main__":
    main()
