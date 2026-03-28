"""CLI script: build the routing dataset from queries and optional oracle outputs.

Usage
-----
    python3 scripts/build_routing_dataset.py [OPTIONS]

Options
-------
--oracle-dir PATH
    Directory containing oracle_assignments.csv and per_query_matrix.csv.
    Defaults to outputs/oracle_subset_eval.

--output-dir PATH
    Directory where routing_dataset.csv and routing_dataset_summary.json
    will be written.  Defaults to outputs/routing_dataset.

--max-queries INT
    Maximum number of queries to include (default: all bundled queries).

--dry-run
    Print the schema and summary to stdout without writing files.

Examples
--------
# schema-only mode (no oracle outputs needed)
python3 scripts/build_routing_dataset.py --dry-run

# full mode, using default oracle output directory
python3 scripts/build_routing_dataset.py

# full mode, custom paths
python3 scripts/build_routing_dataset.py \\
    --oracle-dir outputs/oracle_subset_eval \\
    --output-dir outputs/routing_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.gsm8k import load_gsm8k  # noqa: E402
from src.datasets.routing_dataset import (  # noqa: E402
    DEFAULT_ORACLE_DIR,
    DEFAULT_OUTPUT_DIR,
    assemble_routing_dataset,
    load_oracle_files,
    save_routing_dataset,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the routing dataset from queries and optional oracle outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--oracle-dir",
        default=str(DEFAULT_ORACLE_DIR),
        help=f"Oracle eval output directory (default: {DEFAULT_ORACLE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schema/summary to stdout; do not write files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # --- Load queries (falls back to bundled sample when offline) ---
    _BUNDLED = _REPO_ROOT / "src" / "datasets" / "bundled" / "gsm8k_test_sample.json"
    try:
        queries = load_gsm8k(split="test", max_samples=args.max_queries)
    except Exception:  # noqa: BLE001
        if _BUNDLED.exists():
            print(f"[INFO] HuggingFace unavailable — using bundled sample: {_BUNDLED}")
            queries = load_gsm8k(
                split="test",
                max_samples=args.max_queries,
                data_file=_BUNDLED,
            )
        else:
            print("[ERROR] Could not load queries and no bundled sample found.", file=sys.stderr)
            sys.exit(1)

    if not queries:
        print("[BLOCKED] No queries loaded.  Cannot build routing dataset.", file=sys.stderr)
        sys.exit(1)

    # --- Load oracle data (graceful if missing) ---
    oracle_data = load_oracle_files(args.oracle_dir)

    if oracle_data.available:
        print(f"[INFO] Oracle labels loaded from: {oracle_data.source_files}")
    else:
        print(
            "[INFO] Oracle output files not found — running in schema-only mode.\n"
            f"       Missing: {oracle_data.missing_files}\n"
            "       Run the oracle evaluation to get full labels:\n"
            "         python3 scripts/run_oracle_subset_eval.py "
            "--config configs/oracle_subset_eval_gsm8k.yaml"
        )

    # --- Assemble rows ---
    rows = assemble_routing_dataset(queries, oracle_data=oracle_data)

    if args.dry_run:
        sample = rows[0] if rows else {}
        print("\n=== Routing Dataset Schema ===")
        print(f"Columns ({len(sample)}):")
        for col, val in sample.items():
            print(f"  {col}: {val!r}")
        print(f"\nTotal rows: {len(rows)}")
        print(f"Oracle labels available: {oracle_data.available}")
        return

    # --- Save outputs ---
    paths = save_routing_dataset(rows, args.output_dir, oracle_data=oracle_data)

    # Print summary
    summary = json.loads(Path(paths["summary_path"]).read_text())
    print("\n=== Routing Dataset Build Complete ===")
    print(f"  Queries processed  : {summary['num_queries']}")
    print(f"  Oracle labels      : {summary['oracle_labels_available']}")
    print(f"  Feature columns    : {summary['num_feature_columns']}")
    print(f"  Label columns      : {summary['num_label_columns']}")
    print(f"  CSV output         : {paths['csv_path']}")
    print(f"  Summary JSON       : {paths['summary_path']}")
    if summary["missing_optional_inputs"]:
        print("  Missing optional inputs:")
        for m in summary["missing_optional_inputs"]:
            print(f"    - {m}")


if __name__ == "__main__":
    main()
