#!/usr/bin/env python3
"""Build small AIME 2024 routing dataset (HF HuggingFaceH4/aime_2024)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_real_routing_dataset import (  # noqa: E402
    BuildConfig,
    build_real_routing_dataset,
)
from src.datasets.aime2024 import load_aime2024_hf  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--output-dir", default="outputs/real_aime2024_routing")
    p.add_argument("--output-dataset-csv", default="data/real_aime2024_routing_dataset.csv")
    p.add_argument("--include-reasoning-then-revise", action="store_true")
    p.add_argument("--model-name", default="gpt-4o-mini")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=float, default=150.0)
    args = p.parse_args()

    try:
        qs = load_aime2024_hf(max_samples=args.limit, cache_dir="data")
    except Exception as exc:
        err = {"status": "BLOCKED", "reason": str(exc), "error_type": type(exc).__name__}
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 2

    if not qs:
        print(json.dumps({"status": "BLOCKED", "reason": "empty_aime"}, indent=2), file=sys.stderr)
        return 2

    result = build_real_routing_dataset(
        BuildConfig(
            dataset="custom",
            queries_override=qs,
            subset_size=len(qs),
            output_dir=args.output_dir,
            output_dataset_csv=args.output_dataset_csv,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            timeout=int(args.timeout),
            answer_match_mode="math",
            include_reasoning_then_revise=args.include_reasoning_then_revise,
            summary_filename="aime_run_summary.json",
            per_query_csv_filename="per_query_outputs.csv",
            regime_label="aime2024",
        )
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
