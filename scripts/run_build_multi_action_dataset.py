#!/usr/bin/env python3
"""Build multi-action oracle CSV + summaries (real API).

Datasets:
  gsm8k_hard, math500, gsm8k100 — numeric answers
  aime2024 — numeric (Hugging Face AIME 2024)
  gpqa — multiple-choice A–D (public HF gpqa_diamond mirrors)

Writes CSV, oracle_summary.json, and disagreement_analysis.json under
outputs/multi_action_oracle/ unless overridden.
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

from src.datasets.aime2024 import (  # noqa: E402
    try_load_aime2024_hf,
    write_aime2024_jsonl,
)
from src.datasets.gpqa_diamond import (  # noqa: E402
    try_load_gpqa_diamond_hf,
    write_gpqa_normalized_jsonl,
)
from src.datasets.gsm8k import load_gsm8k  # noqa: E402
from src.datasets.math500 import load_math500  # noqa: E402
from src.evaluation.multi_action_routing import (  # noqa: E402
    MULTI_ACTION_ORDER,
    build_multi_action_rows,
    compute_oracle_summary_json,
    write_disagreement_analysis,
    write_multi_action_csv,
    write_oracle_summary,
)
from src.evaluation.oracle_subset_eval import (  # noqa: E402
    MULTI_ACTION_ORACLE_STRATEGIES,
    run_oracle_subset_eval,
)
from src.models.openai_llm import OpenAILLMModel  # noqa: E402
from src.utils.answer_extraction import normalize_math_answer  # noqa: E402

_DEFAULT_MODEL = "gpt-4o-mini"
_BUNDLED_GSM8K = _REPO_ROOT / "src" / "datasets" / "bundled" / "gsm8k_test_sample.json"


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY", ""):
        print(
            "\n[BLOCKED] OPENAI_API_KEY is not set.\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_blocker(
    dataset: str,
    source: str,
    error: str,
    cause: str,
    fix: str,
) -> None:
    print(
        f"\nDATASET BLOCKED: {dataset}\n"
        f"Source: {source}\n"
        f"Error: {error}\n"
        f"Cause: {cause}\n"
        f"Fix: {fix}\n",
        file=sys.stderr,
    )


def _load_queries_numeric(
    dataset: str,
    subset_size: int,
    gsm8k_data_file: str | None,
    math500_data_file: str | None,
) -> tuple[list, str, str]:
    """Return queries, slug, answer_mode."""
    if dataset == "gsm8k_hard":
        if gsm8k_data_file:
            path = Path(gsm8k_data_file)
            if not path.exists():
                _print_blocker(
                    "gsm8k_hard",
                    str(path),
                    f"File not found: {path}",
                    "dataset not found",
                    "Provide a valid --gsm8k-data-file or omit for HuggingFace.",
                )
                sys.exit(1)
            queries = load_gsm8k(
                split="test",
                data_file=path,
                tail_max_samples=subset_size,
            )
            return queries, "hard_gsm8k_large" if subset_size >= 100 else "gsm8k_hard", "numeric"
        try:
            queries = load_gsm8k(split="test", tail_max_samples=subset_size)
        except Exception as exc:
            if _BUNDLED_GSM8K.exists():
                print(
                    f"[WARN] HuggingFace failed ({exc}); bundled sample.",
                    file=sys.stderr,
                )
                queries = load_gsm8k(
                    split="test",
                    data_file=_BUNDLED_GSM8K,
                    tail_max_samples=min(subset_size, 20),
                )
            else:
                _print_blocker(
                    "gsm8k_hard",
                    "openai/gsm8k via HuggingFace",
                    str(exc),
                    "no internet / HuggingFace access problem",
                    "Enable network or place local GSM8K JSON/JSONL and pass --gsm8k-data-file.",
                )
                sys.exit(1)
        slug = "hard_gsm8k_large" if subset_size >= 100 else "gsm8k_hard"
        return queries, slug, "numeric"

    if dataset == "math500":
        if math500_data_file:
            path = Path(math500_data_file)
            if not path.exists():
                _print_blocker(
                    "math500",
                    str(path),
                    f"File not found: {path}",
                    "dataset not found",
                    "Provide valid --math500-data-file.",
                )
                sys.exit(1)
            queries = load_math500(split="test", max_samples=subset_size, data_file=path)
        else:
            try:
                queries = load_math500(split="test", max_samples=subset_size)
            except Exception as exc:
                _print_blocker(
                    "math500",
                    "HuggingFaceH4/MATH-500",
                    str(exc),
                    "no internet / HuggingFace access problem",
                    "Ensure HF access or pass --math500-data-file with local JSONL.",
                )
                sys.exit(1)
        slug = "math500_large" if subset_size >= 100 else "math500"
        return queries, slug, "numeric"

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
                    _print_blocker(
                        "gsm8k100",
                        "openai/gsm8k",
                        str(exc),
                        "no internet / HuggingFace access problem",
                        "Enable network or use --gsm8k-data-file.",
                    )
                    sys.exit(1)
        return queries, "gsm8k100", "numeric"

    raise ValueError(f"unknown numeric dataset {dataset}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build multi-action oracle dataset.")
    parser.add_argument(
        "--dataset",
        choices=(
            "gsm8k_hard",
            "math500",
            "gsm8k100",
            "aime2024",
            "gpqa",
        ),
        default="gsm8k_hard",
    )
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--gsm8k-data-file", default=None)
    parser.add_argument("--math500-data-file", default=None)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Override CSV path",
    )
    parser.add_argument("--output-summary", default=None)
    parser.add_argument(
        "--output-disagreement",
        default=None,
        help="Override disagreement JSON path",
    )
    parser.add_argument(
        "--write-normalized-jsonl",
        default=None,
        help="aime2024/gpqa: path for normalized jsonl (default under data/)",
    )
    args = parser.parse_args(argv)

    _check_api_key()

    queries: list = []
    slug: str = args.dataset
    answer_mode: str = "numeric"
    gold_normalizer = None

    if args.dataset in ("gsm8k_hard", "math500", "gsm8k100"):
        queries, slug, answer_mode = _load_queries_numeric(
            args.dataset,
            args.subset_size,
            args.gsm8k_data_file,
            args.math500_data_file,
        )
        if args.output_csv:
            # User may force filename e.g. hard_gsm8k_large
            pass

    elif args.dataset == "aime2024":
        queries, src, errs = try_load_aime2024_hf(max_samples=args.subset_size)
        if not queries:
            for e in errs:
                _print_blocker(
                    "AIME 2024",
                    e["source"],
                    e["error"],
                    "dataset not found / schema mismatch / network",
                    "Check HuggingFace connectivity; try Maxwell-Jia/AIME_2024 or mirror JSONL.",
                )
            sys.exit(1)
        slug = "aime2024"
        answer_mode = "numeric"

        def gold_norm_aime(ans: str) -> str:
            return normalize_math_answer(ans)

        gold_normalizer = gold_norm_aime
        norm_path = Path(
            args.write_normalized_jsonl or _REPO_ROOT / "data" / "aime_2024_normalized.jsonl"
        )
        write_aime2024_jsonl(queries, norm_path)
        print(f"[INFO] Wrote normalized AIME JSONL: {norm_path} (source={src})")

    elif args.dataset == "gpqa":
        queries, src, errs = try_load_gpqa_diamond_hf(max_samples=args.subset_size)
        if not queries:
            for e in errs:
                _print_blocker(
                    "GPQA Diamond",
                    e["source"],
                    e["error"],
                    "dataset not found / gated / network",
                    "Use HF_TOKEN for gated sets, or rely on aradhye/gpqa_diamond with network.",
                )
            sys.exit(1)
        slug = "gpqa"
        answer_mode = "multiple_choice"
        norm_path = Path(
            args.write_normalized_jsonl
            or _REPO_ROOT / "data" / "gpqa_diamond_normalized.jsonl"
        )
        write_gpqa_normalized_jsonl(queries, norm_path)
        print(f"[INFO] Wrote normalized GPQA JSONL: {norm_path} (source={src})")

    if not queries:
        print("[BLOCKED] No queries loaded.", file=sys.stderr)
        sys.exit(1)

    strategies = list(MULTI_ACTION_ORACLE_STRATEGIES)
    for name in strategies:
        if name not in MULTI_ACTION_ORDER:
            print(f"[BLOCKED] Strategy {name} not in MULTI_ACTION_ORDER.", file=sys.stderr)
            sys.exit(1)

    print(
        f"[INFO] Model={args.model} dataset_slug={slug} n={len(queries)} "
        f"answer_mode={answer_mode} strategies={strategies}"
    )

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
            answer_mode=answer_mode,
            gold_normalizer=gold_normalizer,
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

    out_dir = _REPO_ROOT / "outputs" / "multi_action_oracle"
    csv_default = _REPO_ROOT / "data" / f"multi_action_routing_{slug}.csv"
    if args.output_csv:
        csv_path = Path(args.output_csv)
    elif args.dataset == "gsm8k_hard" and args.subset_size >= 100:
        csv_path = _REPO_ROOT / "data" / "multi_action_routing_hard_gsm8k_large.csv"
    elif args.dataset == "math500" and args.subset_size >= 100:
        csv_path = _REPO_ROOT / "data" / "multi_action_routing_math500_large.csv"
    else:
        csv_path = csv_default

    write_multi_action_csv(rows, csv_path)

    summary = compute_oracle_summary_json(
        rows,
        strategies,
        model=args.model,
        dataset_name=slug,
    )
    summary["created_at_utc"] = _now_utc()
    summary["subset_size_requested"] = args.subset_size
    summary["answer_mode"] = answer_mode

    summ_path = Path(
        args.output_summary or out_dir / f"{slug}_oracle_summary.json"
    )
    write_oracle_summary(summ_path, summary)

    disag_path = Path(
        args.output_disagreement or out_dir / f"{slug}_disagreement_analysis.json"
    )
    write_disagreement_analysis(
        disag_path,
        rows,
        strategies,
        extra_meta={
            "dataset_slug": slug,
            "model": args.model,
            "answer_mode": answer_mode,
            "num_queries": len(rows),
        },
    )

    print(json.dumps(summary, indent=2))
    print(f"\n[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {summ_path}")
    print(f"[INFO] Wrote {disag_path}")


if __name__ == "__main__":
    main()
