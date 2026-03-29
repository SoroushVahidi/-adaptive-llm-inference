#!/usr/bin/env python3
"""Real OpenAI + V6 scoring on a tiny GSM8K-style set (bundled sample).

One chat completion per example. Writes results under results/real_v6_false_negative_probe/.
Requires OPENAI_API_KEY in the environment. Never prints the key.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

# Repo root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.gsm8k import Query, load_gsm8k  # noqa: E402
from src.evaluation.oracle_subset_eval import _normalize, run_reasoning_greedy  # noqa: E402
from src.models.openai_llm import OpenAILLMModel  # noqa: E402
from src.policies.adaptive_policy_v6 import AdaptivePolicyV6Config, compute_v6_scores  # noqa: E402

BUNDLED = _REPO_ROOT / "src" / "datasets" / "bundled" / "gsm8k_test_sample.json"

# Four doc cases + three cheap extras from the same bundled file.
DEFAULT_QUESTION_IDS = (
    "gsm8k_test_0",
    "gsm8k_test_1",
    "gsm8k_test_18",
    "gsm8k_test_13",
    "gsm8k_test_7",
    "gsm8k_test_11",
    "gsm8k_test_12",
)

REASONING_PREFIX = (
    "Solve this step by step and end with 'Final answer: <number>'.\n\n"
)


def _redact(s: str, max_len: int = 200) -> str:
    s = s[:max_len] + ("…" if len(s) > max_len else "")
    return re.sub(r"sk-[a-zA-Z0-9_-]{10,}", "[REDACTED_TOKEN]", s)


def _answers_match(gold: str, parsed: str) -> bool:
    if not parsed or not str(parsed).strip():
        return False
    g, p = gold.strip(), str(parsed).strip()
    g_norm, p_norm = _normalize(g), _normalize(p)
    try:
        Decimal(g_norm)
        Decimal(p_norm)
        return g_norm == p_norm
    except InvalidOperation:
        return g.casefold() == p.casefold()


def _load_targets(ids: tuple[str, ...]) -> list[Query]:
    all_q = load_gsm8k(
        split="test",
        max_samples=100,
        data_file=str(BUNDLED),
    )
    by_id = {q.id: q for q in all_q}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"Missing question ids in bundled sample: {missing}")
    return [by_id[i] for i in ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="Real API + V6 probe on bundled GSM8K subset")
    parser.add_argument(
        "--model",
        default=os.getenv("REAL_V6_PROBE_MODEL", "gpt-4o-mini"),
        help="OpenAI model name (default: gpt-4o-mini or REAL_V6_PROBE_MODEL)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max completion tokens per call",
    )
    parser.add_argument(
        "--output-dir",
        default="results/real_v6_false_negative_probe",
        help="Output directory (under repo root by default)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("OPENAI_API_KEY not found in environment. Set it and retry.")
        return 1

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v6_cfg = AdaptivePolicyV6Config()
    try:
        model = OpenAILLMModel(
            model_name=str(args.model),
            greedy_temperature=0.0,
            sample_temperature=0.0,
            max_tokens=int(args.max_tokens),
            timeout_seconds=90.0,
            prompt_prefix=REASONING_PREFIX,
        )
    except ValueError as exc:
        print(f"Model setup failed: {exc}")
        return 1

    targets = _load_targets(DEFAULT_QUESTION_IDS)
    raw_path = out_dir / "raw_responses.jsonl"
    rows: list[dict[str, object]] = []
    api_calls = 0

    with raw_path.open("w", encoding="utf-8") as raw_fh:
        for query in targets:
            try:
                result = run_reasoning_greedy(model, query.question)
                api_calls += 1
            except RuntimeError as exc:
                print("API call failed (redacted diagnostic):")
                print(_redact(str(exc)))
                return 1

            raw_text = str(result["raw_outputs"][0])
            oracle_pred = str(result.get("predicted_answer", ""))
            v6 = compute_v6_scores(query.question, raw_text, v6_cfg)
            parsed_v6 = str(v6.get("parsed_answer", "") or "")

            correct = _answers_match(query.answer, parsed_v6)
            wrong_miss = (not correct) and (not bool(v6["revise_recommended"]))

            record = {
                "question_id": query.id,
                "question": query.question,
                "gold_answer": query.answer,
                "raw_model_output": raw_text,
                "oracle_extracted_answer": oracle_pred,
                "v6_parsed_answer": parsed_v6,
                "correct": correct,
                "explanation_warning_score": v6["explanation_warning_score"],
                "answer_error_score": v6["answer_error_score"],
                "final_answer_confident": v6["final_answer_confident"],
                "revise_recommended": v6["revise_recommended"],
                "revise_reason": v6["revise_reason"],
                "wrong_and_v6_no_revise": wrong_miss,
                "v6_contributing_explanation": v6["contributing_explanation_signals"],
                "v6_contributing_answer_error": v6["contributing_answer_error_signals"],
            }
            rows.append(record)
            raw_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    scored_csv = out_dir / "scored_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with scored_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                flat = {
                    **r,
                    "v6_contributing_explanation": ",".join(
                        str(x) for x in r["v6_contributing_explanation"]
                    ),
                    "v6_contributing_answer_error": ",".join(
                        str(x) for x in r["v6_contributing_answer_error"]
                    ),
                }
                w.writerow(flat)

    n = len(rows)
    n_wrong = sum(1 for r in rows if not r["correct"])
    n_wrong_miss = sum(1 for r in rows if r["wrong_and_v6_no_revise"])
    n_wrong_revise = sum(
        1 for r in rows if (not r["correct"]) and r["revise_recommended"]
    )

    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "api_calls": api_calls,
        "num_examples": n,
        "num_wrong": n_wrong,
        "num_wrong_v6_missed_no_revise": n_wrong_miss,
        "num_wrong_v6_revise": n_wrong_revise,
        "output_dir": str(out_dir),
        "bundled_data_file": str(BUNDLED),
        "question_ids": list(DEFAULT_QUESTION_IDS),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("--- real_v6_false_negative_probe ---")
    print(f"model: {args.model}")
    print(f"examples: {n}  api_calls: {api_calls}")
    print(f"wrong: {n_wrong}  wrong_and_v6_no_revise: {n_wrong_miss}")
    print(f"wrong_and_v6_revise: {n_wrong_revise}")
    print(f"wrote: {raw_path}")
    print(f"wrote: {scored_csv}")
    print(f"wrote: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
