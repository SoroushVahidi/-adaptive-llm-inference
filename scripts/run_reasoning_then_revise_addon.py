#!/usr/bin/env python3
"""Append reasoning_then_revise (review-only) to an existing per-query CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.strategy_expansion_eval import (  # noqa: E402
    run_reasoning_then_revise_review_only,
)
from src.models.openai_llm import OpenAILLMModel  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--mode", choices=("numeric", "math"), required=True)
    p.add_argument("--model-name", default="gpt-4o-mini")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    inp = Path(args.input_csv)
    if not inp.exists():
        print(f"BLOCKED: missing input {inp}", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(inp.open(encoding="utf-8")))
    if not rows:
        print("BLOCKED: empty csv", file=sys.stderr)
        return 2

    prefix = (
        "You verify step-by-step math reasoning. "
        "Check the reasoning and final answer carefully. If incorrect, fix it. "
        "If correct, return the same answer."
    )
    model = OpenAILLMModel(
        model_name=args.model_name,
        greedy_temperature=0.0,
        sample_temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        prompt_prefix=prefix,
    )

    def match_gold(gold: str, pred: str) -> bool:
        if not pred.strip():
            return False
        if args.mode == "math":
            from src.utils.answer_extraction import normalize_math_answer

            g = normalize_math_answer(gold)
            p = normalize_math_answer(pred)
            return bool(g) and g == p
        from decimal import Decimal, InvalidOperation

        def norm(v: str) -> str:
            c = v.strip().replace(",", "").replace("$", "").rstrip(".")
            try:
                n = Decimal(c)
                t = format(n.normalize(), "f")
                if "." in t:
                    t = t.rstrip("0").rstrip(".")
                return t or "0"
            except InvalidOperation:
                return c.casefold()

        a, b = norm(gold), norm(pred)
        try:
            Decimal(a)
            Decimal(b)
            return a == b
        except InvalidOperation:
            return a == b

    out_rows: list[dict[str, str]] = []
    for r in rows:
        rr = r.get("reasoning_raw", "")
        if not rr.strip():
            r = dict(r)
            r["reasoning_then_revise_answer"] = ""
            r["reasoning_then_revise_correct"] = "0"
            r["reasoning_then_revise_helpful"] = "0"
            out_rows.append(r)
            continue
        res = run_reasoning_then_revise_review_only(
            model,
            r.get("question", ""),
            rr,
            mode=args.mode,
        )
        pred = str(res.get("predicted_answer", ""))
        gold = r.get("gold_answer", "")
        rc = int(r.get("reasoning_correct", 0))
        ok = int(match_gold(gold, pred))
        r = dict(r)
        r["reasoning_then_revise_answer"] = pred
        r["reasoning_then_revise_correct"] = str(ok)
        r["reasoning_then_revise_helpful"] = str(int((not rc) and ok))
        out_rows.append(r)

    outp = Path(args.output_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with outp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    acc = sum(int(r["reasoning_then_revise_correct"]) for r in out_rows) / n
    helpful = sum(int(r["reasoning_then_revise_helpful"]) for r in out_rows) / n
    summ = {
        "n": n,
        "reasoning_then_revise_accuracy": acc,
        "reasoning_then_revise_helpful_rate": helpful,
        "avg_cost_proxy": 2.0,
        "output_csv": str(outp),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
