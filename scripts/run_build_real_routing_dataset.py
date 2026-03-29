#!/usr/bin/env python3
"""Build the real GSM8K routing dataset by running oracle strategies on a query subset.

This script is the **main offline-preparation entry point** for the 100-query
real-data routing experiment.  When API access is available, run:

    python3 scripts/run_build_real_routing_dataset.py --subset-size 100

What it does
------------
1. Checks that OPENAI_API_KEY is set (exits with a clear blocker message if not).
2. Loads *subset-size* queries from the real GSM8K test split (via HuggingFace;
   falls back to the bundled sample when offline).
3. Initialises the primary LLM model.
4. Runs all core oracle strategies on every query (same strategies as the oracle
   subset evaluation).
5. Assembles oracle label summaries and writes them in the same format as
   ``run_oracle_subset_eval.py`` so that ``src.datasets.routing_dataset`` can
   read them directly.
6. Builds the flat routing dataset CSV with query features + oracle labels.
7. Writes all outputs to ``outputs/real_routing_dataset/``.

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
- Network access to HuggingFace (or bundled sample present at
  src/datasets/bundled/gsm8k_test_sample.json)
- pip install -e ".[dev]" already run

Usage
-----
    # 100-query experiment (main target)
    python3 scripts/run_build_real_routing_dataset.py --subset-size 100

    # Quick smoke-test with a smaller subset
    python3 scripts/run_build_real_routing_dataset.py --subset-size 5

    # Custom model / output directory
    python3 scripts/run_build_real_routing_dataset.py \\
        --subset-size 100 \\
        --model gpt-4o-mini \\
        --output-dir outputs/real_routing_dataset_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def _load_queries(subset_size: int) -> list[Any]:
    """Load queries from HuggingFace GSM8K, falling back to bundled sample."""
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
            "  with at least one record containing 'question' and 'answer' keys.\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    timestamp = _now_utc()
    output_dir = Path(args.output_dir)

    # --- Guard: API key ---
    _check_api_key()

    # --- Load queries ---
    queries = _load_queries(args.subset_size)
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
    print()
    print("Next step: evaluate a routing model on this dataset:")
    print(
        "  python3 scripts/run_real_routing_model_eval.py "
        f"--routing-csv {routing_paths['csv_path']}"
    )


if __name__ == "__main__":
    main()
