#!/usr/bin/env python3
"""Build multi-action oracle CSV + summary for supervised routing (real API).

Writes:
  data/multi_action_routing_<dataset>.csv
  outputs/multi_action_oracle/<dataset>_oracle_summary.json

Datasets:
  gsm8k_hard — last N GSM8K test problems (tail slice as a harder proxy)
  math500    — first N MATH-500 problems

Requires OPENAI_API_KEY unless using a compatible local endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.gsm8k import load_gsm8k  # noqa: E402
from src.datasets.math500 import load_math500  # noqa: E402
from src.evaluation.multi_action_routing import (  # noqa: E402
    MULTI_ACTION_ORDER,
    build_multi_action_rows,
    compute_oracle_summary_json,
    write_multi_action_csv,
    write_oracle_summary,
)
from src.evaluation.oracle_subset_eval import (  # noqa: E402
    MULTI_ACTION_ORACLE_STRATEGIES,
    run_oracle_subset_eval,
)
from src.models.openai_llm import OpenAILLMModel  # noqa: E402

_DEFAULT_MODEL = "gpt-4o-mini"
_BUNDLED_GSM8K = _REPO_ROOT / "src" / "datasets" / "bundled" / "gsm8k_test_sample.json"


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "\n[BLOCKED] OPENAI_API_KEY is not set.\n\n"
            "  Export a key or use an OpenAI-compatible endpoint with key in "
            "OPENAI_API_KEY.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def _load_queries(
    dataset: str,
    subset_size: int,
    gsm8k_data_file: str | None,
    math500_data_file: str | None,
) -> tuple[list, str]:
    if dataset == "gsm8k_hard":
        if gsm8k_data_file:
            path = Path(gsm8k_data_file)
            if not path.exists():
                print(f"[BLOCKED] File not found: {path}", file=sys.stderr)
                sys.exit(1)
            queries = load_gsm8k(
                split="test",
                data_file=path,
                tail_max_samples=subset_size,
            )
            return queries, "gsm8k_hard"
        try:
            queries = load_gsm8k(split="test", tail_max_samples=subset_size)
        except Exception as exc:
            if _BUNDLED_GSM8K.exists():
                print(
                    f"[WARN] HuggingFace failed ({exc}); using bundled GSM8K sample.",
                    file=sys.stderr,
                )
                queries = load_gsm8k(
                    split="test",
                    data_file=_BUNDLED_GSM8K,
                    tail_max_samples=min(subset_size, 20),
                )
            else:
                print(
                    f"\n[BLOCKED] Cannot load GSM8K.\n  Error: {exc}\n",
                    file=sys.stderr,
                )
                sys.exit(1)
        return queries, "gsm8k_hard"

    if dataset == "math500":
        if math500_data_file:
            path = Path(math500_data_file)
            if not path.exists():
                print(f"[BLOCKED] File not found: {path}", file=sys.stderr)
                sys.exit(1)
            queries = load_math500(split="test", max_samples=subset_size, data_file=path)
        else:
            try:
                queries = load_math500(split="test", max_samples=subset_size)
            except Exception as exc:
                print(
                    f"\n[BLOCKED] Cannot load MATH500 (network or cache).\n"
                    f"  Error: {exc}\n"
                    "  Fix: ensure HuggingFace access or pass --math500-data-file.\n",
                    file=sys.stderr,
                )
                sys.exit(1)
        return queries, "math500"

    if dataset == "gsm8k100":
        if gsm8k_data_file:
            queries = load_gsm8k(
                split="test",
                max_samples=subset_size,
                data_file=gsm8k_data_file,
            )
        else:
            try:
                queries = load_gsm8k(split="test", max_samples=subset_size)
            except Exception as exc:
                if _BUNDLED_GSM8K.exists():
                    queries = load_gsm8k(
                        split="test",
                        max_samples=min(subset_size, 20),
                        data_file=_BUNDLED_GSM8K,
                    )
                    print(f"[WARN] HF failed ({exc}); bundled sample.", file=sys.stderr)
                else:
                    print(f"[BLOCKED] Cannot load GSM8K: {exc}", file=sys.stderr)
                    sys.exit(1)
        return queries, "gsm8k100"

    print(f"[BLOCKED] Unknown dataset: {dataset}", file=sys.stderr)
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build multi-action oracle dataset.")
    parser.add_argument(
        "--dataset",
        choices=("gsm8k_hard", "math500", "gsm8k100"),
        default="gsm8k_hard",
    )
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--gsm8k-data-file", default=None)
    parser.add_argument("--math500-data-file", default=None)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Override CSV path (default: data/multi_action_routing_<dataset>.csv)",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Override summary JSON path",
    )
    args = parser.parse_args(argv)

    _check_api_key()

    queries, slug = _load_queries(
        args.dataset,
        args.subset_size,
        args.gsm8k_data_file,
        args.math500_data_file,
    )
    if not queries:
        print("[BLOCKED] No queries loaded.", file=sys.stderr)
        sys.exit(1)

    strategies = list(MULTI_ACTION_ORACLE_STRATEGIES)
    for name in strategies:
        if name not in MULTI_ACTION_ORDER:
            print(f"[BLOCKED] Strategy {name} not in MULTI_ACTION_ORDER.", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Model={args.model} dataset={slug} n={len(queries)} strategies={strategies}")

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
        print(f"[BLOCKED] Model init failed: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        eval_result = run_oracle_subset_eval(
            model=model,
            queries=queries,
            strategies=strategies,
        )
    except RuntimeError as exc:
        print(f"[BLOCKED] Oracle eval failed: {exc}", file=sys.stderr)
        sys.exit(1)

    rows = build_multi_action_rows(
        queries,
        eval_result["per_query_rows"],
        strategies,
        dataset_name=slug,
    )

    csv_path = Path(
        args.output_csv or _REPO_ROOT / "data" / f"multi_action_routing_{slug}.csv"
    )
    write_multi_action_csv(rows, csv_path)

    summary = compute_oracle_summary_json(
        rows,
        strategies,
        model=args.model,
        dataset_name=slug,
    )
    summary["created_at_utc"] = _now_utc()
    summary["subset_size_requested"] = args.subset_size

    summ_path = Path(
        args.output_summary
        or _REPO_ROOT / "outputs" / "multi_action_oracle" / f"{slug}_oracle_summary.json"
    )
    write_oracle_summary(summ_path, summary)

    print(json.dumps(summary, indent=2))
    print(f"\n[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {summ_path}")


if __name__ == "__main__":
    main()
