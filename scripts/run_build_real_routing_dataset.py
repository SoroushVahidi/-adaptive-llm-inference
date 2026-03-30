#!/usr/bin/env python3
"""Build the real GSM8K routing dataset by running oracle strategies on a query subset.

This script is the **main offline-preparation entry point** for the 100-query
real-data routing experiment.  When API access is available, run:

    python3 scripts/run_build_real_routing_dataset.py --subset-size 100

What it does
------------
1. Checks that OPENAI_API_KEY is set (exits with a clear blocker message if not).
2. Loads *subset-size* queries from the real GSM8K test split (via HuggingFace;
   falls back to ``--gsm8k-data-file`` if provided, or the bundled sample when
   offline).
3. Initialises the primary LLM model.
4. Runs all core oracle strategies on every query (same strategies as the oracle
   subset evaluation).
5. Assembles oracle label summaries and writes them in the same format as
   ``run_oracle_subset_eval.py`` so that ``src.datasets.routing_dataset`` can
   read them directly.
6. Builds the flat routing dataset CSV with query features + oracle labels.
7. Writes all outputs to ``outputs/real_routing_dataset/`` (and optionally to
   ``--output-dataset-csv`` for downstream compatibility).

Outputs
-------
``outputs/real_routing_dataset/``
    ├── oracle_assignments.csv       — per-query oracle label columns
    ├── per_query_matrix.csv         — per-strategy per-query correctness matrix
    ├── oracle_summary.json          — oracle-level accuracy summary
    ├── routing_dataset.csv          — flat feature + label CSV for ML
    └── routing_dataset_summary.json — column inventory and metadata

Prerequisites
-------------
- OPENAI_API_KEY environment variable set
- Network access to HuggingFace (or --gsm8k-data-file / bundled sample)
- pip install -e ".[dev]" already run

Usage
-----
    # 100-query experiment (main target)
    python3 scripts/run_build_real_routing_dataset.py --subset-size 100

    # Quick smoke-test with a smaller subset
    python3 scripts/run_build_real_routing_dataset.py --subset-size 5

    # Use a local normalized JSONL file instead of HuggingFace
    python3 scripts/run_build_real_routing_dataset.py \\
        --subset-size 100 \\
        --gsm8k-data-file data/gsm8k_uploaded_normalized.jsonl

    # Custom model / output directory
    python3 scripts/run_build_real_routing_dataset.py \\
        --subset-size 100 \\
        --model gpt-4o-mini \\
        --output-dir outputs/real_routing_dataset_v2

    # Paired outcomes (reasoning + direct_plus_revise + features) for policy eval
    python3 scripts/run_build_real_routing_dataset.py \\
        --paired-outcomes --subset-size 100 \\
        --output-dataset-csv data/real_gsm8k_routing_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.build_real_routing_dataset import (  # noqa: E402
    BuildConfig,
    build_real_routing_dataset,
)
from src.datasets.gsm8k import load_gsm8k  # noqa: E402
from src.datasets.routing_dataset import (  # noqa: E402
    assemble_routing_dataset,
    load_oracle_files,
    save_routing_dataset,
)
from src.evaluation.oracle_subset_eval import (  # noqa: E402
    CORE_ORACLE_STRATEGIES,
    compute_oracle_summaries,
    compute_pairwise_win_matrix,
    run_oracle_subset_eval,
    write_oracle_outputs,
)
from src.models.openai_llm import OpenAILLMModel  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path("outputs/real_routing_dataset")
_DEFAULT_MODEL = "gpt-4o-mini"
_BUNDLED = _REPO_ROOT / "src" / "datasets" / "bundled" / "gsm8k_test_sample.json"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the real GSM8K routing dataset by running oracle strategies "
            "on a query subset.  Requires OPENAI_API_KEY."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        metavar="N",
        help="Number of GSM8K test queries to process (default: 100)",
    )
    parser.add_argument(
        "--gsm8k-data-file",
        default=None,
        metavar="PATH",
        help=(
            "Optional path to a local GSM8K JSONL/JSON file "
            "(e.g. data/gsm8k_uploaded_normalized.jsonl). "
            "When provided, skips HuggingFace download and uses this file directly."
        ),
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        metavar="MODEL_NAME",
        help=f"OpenAI model name (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help=(
            "Optional base URL for a compatible API endpoint "
            "(defaults to OPENAI_BASE_URL env var or https://api.openai.com/v1)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-dataset-csv",
        default=None,
        metavar="PATH",
        help=(
            "Optional additional output path for the routing dataset CSV "
            "(e.g. data/real_gsm8k_routing_dataset.csv). "
            "When provided, the CSV is copied to this path after writing."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        metavar="N",
        help="Max tokens per model call (default: 512)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Per-call timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        metavar="STRATEGY",
        help=(
            "Override oracle strategies to run.  Defaults to all core oracle "
            "strategies: " + ", ".join(CORE_ORACLE_STRATEGIES)
        ),
    )
    parser.add_argument(
        "--paired-outcomes",
        action="store_true",
        help=(
            "Run reasoning_greedy + direct_plus_revise paired pipeline with "
            "engineered features (src.data.build_real_routing_dataset) instead "
            "of the full multi-strategy oracle matrix."
        ),
    )
    parser.add_argument(
        "--bundled-fallback",
        default=str(_REPO_ROOT / "src" / "datasets" / "bundled" / "gsm8k_test_sample.json"),
        metavar="PATH",
        help="Bundled GSM8K JSON when HF/local file insufficient (paired mode)",
    )
    parser.add_argument(
        "--include-reasoning-then-revise",
        action="store_true",
        help="Third model stage: review reasoning trace (paired mode only)",
    )
    parser.add_argument(
        "--regime-label",
        default="gsm8k_baseline",
        help="Tag written into paired-mode run summary JSON",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_api_key() -> str:
    """Return API key or exit with a clear blocker message."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "\n[BLOCKED] OPENAI_API_KEY is not set.\n"
            "\n"
            "  This script requires real model calls and cannot run without\n"
            "  a valid OpenAI API key (or compatible endpoint key).\n"
            "\n"
            "  To unblock:\n"
            "    export OPENAI_API_KEY=sk-...\n"
            "    python3 scripts/run_build_real_routing_dataset.py "
            "--subset-size 100\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def _load_queries(subset_size: int, gsm8k_data_file: str | None) -> list[Any]:
    """Load queries from a local file, HuggingFace, or the bundled sample.

    Priority order:
    1. ``gsm8k_data_file`` (explicit local path, e.g. normalized JSONL)
    2. HuggingFace download
    3. Bundled sample fallback
    """
    # --- Explicit local file takes priority ---
    if gsm8k_data_file is not None:
        data_path = Path(gsm8k_data_file)
        if not data_path.exists():
            print(
                f"\n[BLOCKED] --gsm8k-data-file not found: {data_path}\n"
                "  Provide a valid path or omit the flag to use HuggingFace.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[INFO] Loading {subset_size} queries from local file: {data_path}")
        queries = load_gsm8k(split="test", max_samples=subset_size, data_file=data_path)
        print(f"[INFO] Loaded {len(queries)} queries from local file.")
        return queries

    # --- HuggingFace with bundled fallback ---
    print(f"[INFO] Loading {subset_size} GSM8K test queries…")
    try:
        queries = load_gsm8k(split="test", max_samples=subset_size)
        print(f"[INFO] Loaded {len(queries)} queries from HuggingFace.")
        return queries
    except Exception as exc:  # noqa: BLE001
        if _BUNDLED.exists():
            print(
                f"[WARN] HuggingFace unavailable ({exc}); "
                f"falling back to bundled sample: {_BUNDLED}"
            )
            queries = load_gsm8k(
                split="test", max_samples=subset_size, data_file=_BUNDLED
            )
            print(f"[INFO] Loaded {len(queries)} queries from bundled sample.")
            return queries
        print(
            "\n[BLOCKED] Cannot load GSM8K queries.\n"
            f"  HuggingFace error: {exc}\n"
            f"  Bundled sample not found at: {_BUNDLED}\n"
            "\n"
            "  Ensure network access is available for the first run, or place\n"
            "  a local JSON file at:\n"
            f"    {_BUNDLED}\n"
            "  or pass --gsm8k-data-file <path> to use a local file.\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_paired_outcomes(args: argparse.Namespace, timestamp: str) -> None:
    """Reasoning + direct_plus_revise (+ optional RTR) with feature columns."""
    output_dir = Path(args.output_dir)
    data_file = args.gsm8k_data_file
    if data_file is not None and not Path(data_file).exists():
        data_file = None

    out_csv = args.output_dataset_csv
    if out_csv is None:
        out_csv = "data/real_gsm8k_routing_dataset.csv"

    result = build_real_routing_dataset(
        BuildConfig(
            gsm8k_data_file=Path(data_file) if data_file else None,
            subset_size=args.subset_size,
            output_dir=output_dir,
            output_dataset_csv=Path(out_csv),
            model_name=args.model,
            max_tokens=int(args.max_tokens),
            timeout=int(args.timeout),
            bundled_fallback=Path(args.bundled_fallback),
            dataset="gsm8k",
            summary_filename="gsm8k_subset_run_summary.json",
            per_query_csv_filename="gsm8k_per_query_outputs.csv",
            regime_label=args.regime_label,
            include_reasoning_then_revise=args.include_reasoning_then_revise,
        )
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"summary_json={result['summary_path']}")
    print(f"per_query_csv={result['per_query_csv']}")
    print(f"dataset_csv={result['dataset_csv']}")
    print(f"timestamp_utc={timestamp}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    timestamp = _now_utc()
    output_dir = Path(args.output_dir)

    # --- Guard: API key ---
    _check_api_key()

    if args.paired_outcomes:
        _run_paired_outcomes(args, timestamp)
        return

    # --- Load queries ---
    queries = _load_queries(args.subset_size, args.gsm8k_data_file)
    if not queries:
        print("[BLOCKED] No queries loaded. Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # --- Initialise model ---
    print(f"[INFO] Initialising model '{args.model}'…")
    try:
        model = OpenAILLMModel(
            model_name=args.model,
            base_url=args.base_url,
            greedy_temperature=0.0,
            sample_temperature=0.7,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout,
        )
    except ValueError as exc:
        print(
            f"\n[BLOCKED] Cannot initialise model '{args.model}'.\n"
            f"  Error: {exc}\n"
            "\n"
            "  Ensure OPENAI_API_KEY is set and correct.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    strategies: list[str] = args.strategies or list(CORE_ORACLE_STRATEGIES)
    print(f"[INFO] Strategies: {strategies}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Starting oracle evaluation on {len(queries)} queries…\n")

    # --- Run oracle evaluation ---
    try:
        eval_result = run_oracle_subset_eval(
            model=model,
            queries=queries,
            strategies=strategies,
        )
    except RuntimeError as exc:
        print(
            f"\n[BLOCKED] Oracle evaluation failed during execution.\n"
            f"  Error: {exc}\n"
            "\n"
            "  Check API connectivity, key validity, and model availability.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Compute summaries ---
    oracle_summaries = compute_oracle_summaries(
        eval_result["per_query_rows"],
        eval_result["strategies_run"],
    )
    pairwise = compute_pairwise_win_matrix(
        eval_result["per_query_rows"],
        eval_result["strategies_run"],
    )

    # --- Write oracle output artefacts ---
    oracle_paths = write_oracle_outputs(
        eval_result, oracle_summaries, pairwise, output_dir
    )

    # Write a human-readable oracle summary JSON alongside the oracle outputs.
    oracle_summary_path = output_dir / "oracle_summary.json"
    oracle_summary_path.write_text(
        json.dumps(
            {
                "run_type": "real_routing_oracle_eval",
                "created_at_utc": timestamp,
                "model": args.model,
                "total_queries": len(queries),
                "subset_size_requested": args.subset_size,
                "strategies_run": eval_result["strategies_run"],
                "oracle_summaries": oracle_summaries,
            },
            indent=2,
        )
    )

    print(f"[INFO] Oracle artefacts written to: {output_dir}")
    for name, path in oracle_paths.items():
        print(f"         {name}: {path}")

    # --- Assemble routing dataset from oracle outputs ---
    print("\n[INFO] Assembling routing dataset…")
    oracle_data = load_oracle_files(output_dir)

    if not oracle_data.available:
        print(
            "[WARN] Oracle data not found after writing artefacts — "
            "routing dataset will be in schema-only mode.\n"
            f"       Missing: {oracle_data.missing_files}",
            file=sys.stderr,
        )

    rows = assemble_routing_dataset(queries, oracle_data=oracle_data)
    routing_paths = save_routing_dataset(rows, output_dir, oracle_data=oracle_data)

    # --- Optional: copy CSV to secondary output path ---
    if args.output_dataset_csv is not None:
        dest = Path(args.output_dataset_csv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(routing_paths["csv_path"], dest)
        print(f"[INFO] Routing dataset also written to: {dest}")

    # --- Print final summary ---
    summary: dict[str, Any] = json.loads(
        Path(routing_paths["summary_path"]).read_text()
    )

    print("\n=== Real Routing Dataset Build Complete ===")
    print(f"  Timestamp          : {timestamp}")
    print(f"  Model              : {args.model}")
    print(f"  Queries processed  : {summary['num_queries']}")
    print(f"  Oracle labels      : {summary['oracle_labels_available']}")
    print(f"  Feature columns    : {summary['num_feature_columns']}")
    print(f"  Label columns      : {summary['num_label_columns']}")
    print(f"  CSV output         : {routing_paths['csv_path']}")
    print(f"  Summary JSON       : {routing_paths['summary_path']}")
    print(f"  Oracle summary     : {oracle_summary_path}")
    if args.output_dataset_csv is not None:
        print(f"  Dataset CSV copy   : {args.output_dataset_csv}")
    print()
    print("Next step: sklearn router baselines on oracle-labelled rows:")
    print(
        "  python3 scripts/run_router_baseline_eval.py "
        f"--routing-csv {routing_paths['csv_path']}"
    )


if __name__ == "__main__":
    main()
