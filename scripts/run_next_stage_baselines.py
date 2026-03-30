#!/usr/bin/env python3
"""Run self-consistency, direct+revise, reasoning_then_revise on a query subset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.aime2024 import load_aime2024_hf  # noqa: E402, I001
from src.datasets.gsm8k import load_gsm8k  # noqa: E402
from src.datasets.math500 import load_math500  # noqa: E402
from src.evaluation.next_stage_experiments import match_gold  # noqa: E402
from src.evaluation.strategy_expansion_eval import (  # noqa: E402
    _normalize as _norm_numeric,
    run_direct_plus_revise,
    run_reasoning_then_revise,
    run_self_consistency_reasoning_n_math,
    run_self_consistency_reasoning_n_numeric,
)
from src.models.openai_llm import OpenAILLMModel  # noqa: E402
from src.utils.answer_extraction import (  # noqa: E402
    extract_math_answer,
    extract_numeric_answer,
    normalize_math_answer,
)


def load_queries_from_hard_csv(path: Path, limit: int) -> list[tuple[str, str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = []
    for r in rows[:limit]:
        out.append((r["question_id"], r["question"], r["gold_answer"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="gsm8k|hard_gsm8k|math500|aime2024")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument(
        "--hard-selection-csv",
        default="outputs/hard_regime_selection/hard_gsm8k_selection.csv",
    )
    p.add_argument("--model-name", default="gpt-4o-mini")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--output-dir", default="outputs/baselines")
    args = p.parse_args()

    mode = "math" if args.dataset in ("math500", "aime2024") else "numeric"
    reasoning_prefix = (
        "Solve this step by step. Put your final answer in \\boxed{...} "
        "or end with 'Final answer: ...'.\n\n"
        if mode == "math"
        else "Solve this step by step and end with 'Final answer: <number>'.\n\n"
    )
    direct_prefix = (
        "Answer the following math question. Give only the final answer; "
        "use \\boxed{...} when appropriate.\n\n"
        if mode == "math"
        else "Answer the following math question. Give only the final numeric answer."
    )
    rtr_rev_prefix = (
        "You verify step-by-step math reasoning. "
        "Check the reasoning and final answer carefully. If incorrect, fix it. "
        "If correct, return the same answer."
    )

    queries: list[tuple[str, str, str]] = []
    if args.dataset == "gsm8k":
        lim = args.limit
        qs = load_gsm8k(split="test", max_samples=lim, cache_dir="data")
        queries = [(q.id, q.question, q.answer) for q in qs]
    elif args.dataset == "hard_gsm8k":
        sel = Path(args.hard_selection_csv)
        if not sel.exists():
            print(f"BLOCKED: {sel} missing", file=sys.stderr)
            return 2
        queries = load_queries_from_hard_csv(sel, args.limit)
    elif args.dataset == "math500":
        qs = load_math500(max_samples=args.limit, cache_dir="data")
        queries = [(q.id, q.question, q.answer) for q in qs]
    elif args.dataset == "aime2024":
        qs = load_aime2024_hf(max_samples=None, cache_dir="data")
        queries = [(q.id, q.question, q.answer) for q in qs[: args.limit]]
    else:
        print("BLOCKED: unknown dataset", file=sys.stderr)
        return 2

    if not queries:
        print("BLOCKED: no queries", file=sys.stderr)
        return 2

    model_r1 = OpenAILLMModel(
        model_name=args.model_name,
        greedy_temperature=0.0,
        sample_temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        prompt_prefix=reasoning_prefix,
    )
    model_r = OpenAILLMModel(
        model_name=args.model_name,
        greedy_temperature=0.0,
        sample_temperature=0.7,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        prompt_prefix=reasoning_prefix,
    )
    model_d = OpenAILLMModel(
        model_name=args.model_name,
        greedy_temperature=0.0,
        sample_temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        prompt_prefix=direct_prefix,
    )
    model_rtr_rev = OpenAILLMModel(
        model_name=args.model_name,
        greedy_temperature=0.0,
        sample_temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        prompt_prefix=rtr_rev_prefix,
    )

    n = len(queries)
    strategies = [
        "reasoning_greedy",
        "self_consistency_3",
        "self_consistency_5",
        "direct_plus_revise",
        "reasoning_then_revise",
    ]
    agg: dict[str, dict[str, float]] = {}

    def run_all() -> None:
        for name in strategies:
            agg[name] = {"correct": 0.0, "samples": 0.0}
        # ladder includes reasoning_greedy as first rung

        for qid, question, gold in queries:
            raw1 = model_r1.generate(question)
            if mode == "math":
                p1 = extract_math_answer(raw1).strip()
                pred1 = normalize_math_answer(p1) if p1 else ""
            else:
                pred1 = _norm_numeric(extract_numeric_answer(raw1))
            if match_gold(gold, pred1, mode):
                agg["reasoning_greedy"]["correct"] += 1
            agg["reasoning_greedy"]["samples"] += 1.0

            for sn in (3, 5):
                if mode == "math":
                    res = run_self_consistency_reasoning_n_math(model_r, question, sn)
                else:
                    res = run_self_consistency_reasoning_n_numeric(model_r, question, sn)
                pred = str(res.get("predicted_answer", ""))
                if match_gold(gold, pred, mode):
                    agg[f"self_consistency_{sn}"]["correct"] += 1
                agg[f"self_consistency_{sn}"]["samples"] += float(res.get("samples_used", sn))

            dpr = run_direct_plus_revise(model_d, question)
            pred_d = str(dpr.get("predicted_answer", ""))
            if mode == "math":
                pred_d = normalize_math_answer(pred_d)
            if match_gold(gold, pred_d, mode):
                agg["direct_plus_revise"]["correct"] += 1
            agg["direct_plus_revise"]["samples"] += float(dpr.get("samples_used", 2))

            rtr = run_reasoning_then_revise(model_r, question, revise_model=model_rtr_rev)
            pred_r = str(rtr.get("predicted_answer", ""))
            if mode == "math":
                pred_r = normalize_math_answer(pred_r)
            if match_gold(gold, pred_r, mode):
                agg["reasoning_then_revise"]["correct"] += 1
            agg["reasoning_then_revise"]["samples"] += float(rtr.get("samples_used", 2))

    try:
        run_all()
    except Exception as exc:
        err = {
            "dataset": args.dataset,
            "status": "FAILED",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        outd = Path(args.output_dir)
        outd.mkdir(parents=True, exist_ok=True)
        outp = outd / f"{args.dataset}_baseline_summary.json"
        outp.write_text(json.dumps(err, indent=2), encoding="utf-8")
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 1

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "mode": mode,
        "n_queries": n,
        "model_name": args.model_name,
        "strategies": {},
    }
    for name in strategies:
        c = agg[name]["correct"]
        s = agg[name]["samples"]
        summary["strategies"][name] = {
            "accuracy": c / n,
            "avg_cost_proxy": s / n,
        }

    outd = Path(args.output_dir)
    outd.mkdir(parents=True, exist_ok=True)
    outp = outd / f"{args.dataset}_baseline_summary.json"
    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
