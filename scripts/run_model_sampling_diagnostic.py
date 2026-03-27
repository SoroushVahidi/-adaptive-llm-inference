#!/usr/bin/env python3
"""Diagnostic experiment: compare model strength and sampling strategies on GSM8K.

Goal: Understand whether
  1) the current model is too weak for reasoning, and
  2) structured sampling beats naive best-of-3.

Models
------
  - gpt-4o-mini  (current model used throughout the repo)
  - gpt-4o       (stronger model; skipped gracefully if not accessible)

Strategies (per model)
----------------------
  A) direct_greedy          – direct prompt, 1 sample
  B) reasoning_greedy       – reasoning prompt, 1 sample
  C) reasoning_best_of_3    – reasoning prompt, 3 samples, majority vote
  D) structured_sampling_3  – 3 different prompts, 1 sample each, majority vote

Output: outputs/model_sampling_diagnostic/
  - summary.json
  - summary.csv
  - per_query_results.csv

Usage
-----
  OPENAI_API_KEY=sk-... python3 scripts/run_model_sampling_diagnostic.py [--max-queries N]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work when the script
# is invoked directly (e.g. `python3 scripts/run_model_sampling_diagnostic.py`).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.gsm8k import load_gsm8k  # noqa: E402
from src.models.openai_llm import OpenAILLMModel  # noqa: E402
from src.utils.answer_extraction import extract_numeric_answer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUERIES_DEFAULT = 30

CURRENT_MODEL = "gpt-4o-mini"
STRONGER_MODEL = "gpt-4o"

# Prompt prefixes used per strategy
PROMPT_DIRECT = "Answer the following question. Give only the final numeric answer."
PROMPT_REASONING = (
    "Answer the following question step by step. "
    "At the end, state your final numeric answer clearly."
)
PROMPT_CONCISE = "Answer the following question with a direct, concise numeric answer."
PROMPT_STEP_BY_STEP = (
    "Solve the following question step by step, showing all your work. "
    "Conclude with: 'The answer is <number>.'"
)
PROMPT_DOUBLE_CHECK = (
    "Solve the following question, then double-check your answer. "
    "State the verified final numeric answer at the end."
)

OUTPUT_DIR = Path("outputs/model_sampling_diagnostic")


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


def _majority_vote(answers: list[str]) -> str:
    """Return the most common non-empty answer; fall back to first answer."""
    non_empty = [a for a in answers if a]
    if not non_empty:
        return answers[0] if answers else ""
    return Counter(non_empty).most_common(1)[0][0]


def run_direct_greedy(model: OpenAILLMModel, question: str) -> tuple[str, int]:
    """Single sample with direct prompt."""
    m = model.with_prompt_prefix(PROMPT_DIRECT)
    raw = m.generate(question)
    return extract_numeric_answer(raw), 1


def run_reasoning_greedy(model: OpenAILLMModel, question: str) -> tuple[str, int]:
    """Single sample with reasoning prompt."""
    m = model.with_prompt_prefix(PROMPT_REASONING)
    raw = m.generate(question)
    return extract_numeric_answer(raw), 1


def run_reasoning_best_of_3(model: OpenAILLMModel, question: str) -> tuple[str, int]:
    """Three samples with reasoning prompt; majority vote."""
    m = model.with_prompt_prefix(PROMPT_REASONING)
    raws = m.generate_n(question, 3)
    extracted = [extract_numeric_answer(r) for r in raws]
    return _majority_vote(extracted), 3


def run_structured_sampling_3(model: OpenAILLMModel, question: str) -> tuple[str, int]:
    """Three samples with distinct prompts; majority vote."""
    answers: list[str] = []
    for prefix in (PROMPT_CONCISE, PROMPT_STEP_BY_STEP, PROMPT_DOUBLE_CHECK):
        m = model.with_prompt_prefix(prefix)
        raw = m.generate(question)
        answers.append(extract_numeric_answer(raw))
    return _majority_vote(answers), 3


STRATEGIES: dict[str, Callable[[OpenAILLMModel, str], tuple[str, int]]] = {
    "direct_greedy": run_direct_greedy,
    "reasoning_greedy": run_reasoning_greedy,
    "reasoning_best_of_3": run_reasoning_best_of_3,
    "structured_sampling_3": run_structured_sampling_3,
}


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    model_name: str,
    queries: list,
    max_tokens: int = 512,
) -> list[dict]:
    """Run all four strategies for *model_name* and return per-query rows."""
    try:
        model = OpenAILLMModel(
            model_name=model_name,
            max_tokens=max_tokens,
            sample_temperature=0.7,
            greedy_temperature=0.0,
        )
    except ValueError as exc:
        print(f"ERROR: Cannot initialise model '{model_name}': {exc}")
        raise

    rows: list[dict] = []
    n_queries = len(queries)

    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"  strategy={strategy_name} ...", flush=True)
        for idx, query in enumerate(queries, 1):
            try:
                predicted, samples_used = strategy_fn(model, query.question)
            except RuntimeError as exc:
                print(
                    f"    ERROR on query {query.id} strategy={strategy_name}: {exc}",
                    file=sys.stderr,
                )
                raise

            correct = predicted == query.answer
            rows.append(
                {
                    "question_id": query.id,
                    "model": model_name,
                    "strategy": strategy_name,
                    "predicted_answer": predicted,
                    "ground_truth": query.answer,
                    "correct": correct,
                    "samples_used": samples_used,
                }
            )
            if idx % 10 == 0 or idx == n_queries:
                print(f"    {idx}/{n_queries} queries done", flush=True)

    return rows


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def compute_metrics(rows: list[dict]) -> list[dict]:
    """Aggregate per-query rows into per-(model, strategy) metric dicts."""
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["strategy"])].append(row)

    summaries: list[dict] = []
    for (model_name, strategy), group in sorted(groups.items()):
        total = len(group)
        correct = sum(1 for r in group if r["correct"])
        total_samples = sum(r["samples_used"] for r in group)
        summaries.append(
            {
                "model": model_name,
                "strategy": strategy,
                "accuracy": round(correct / total, 4) if total > 0 else 0.0,
                "correct": correct,
                "total_queries": total,
                "total_samples": total_samples,
                "avg_samples_per_query": round(total_samples / total, 2) if total > 0 else 0.0,
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_outputs(rows: list[dict], summaries: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))

    # summary.csv
    if summaries:
        summary_fields = list(summaries[0].keys())
        with (output_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summaries)

    # per_query_results.csv
    if rows:
        row_fields = list(rows[0].keys())
        with (output_dir / "per_query_results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nOutputs saved to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_summary(summaries: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Build lookup: (model, strategy) -> metrics
    lookup: dict[tuple[str, str], dict] = {
        (s["model"], s["strategy"]): s for s in summaries
    }

    models = sorted({s["model"] for s in summaries})
    strategies = [
        "direct_greedy",
        "reasoning_greedy",
        "reasoning_best_of_3",
        "structured_sampling_3",
    ]

    # Per-model, per-strategy table
    header = (
        f"{'Model':<20} {'Strategy':<26} {'Accuracy':>9}"
        f" {'Queries':>8} {'Samples':>8} {'Avg/Q':>6}"
    )
    print(header)
    print("-" * 70)
    for model_name in models:
        for strat in strategies:
            m = lookup.get((model_name, strat))
            if m is None:
                continue
            print(
                f"{model_name:<20} {strat:<26} {m['accuracy']:>9.1%} "
                f"{m['total_queries']:>8} {m['total_samples']:>8}"
                f" {m['avg_samples_per_query']:>6.1f}"
            )
        print()

    # Analysis: structured_sampling_3 vs reasoning_best_of_3
    print("-" * 70)
    print("Analysis: structured_sampling_3 vs reasoning_best_of_3")
    for model_name in models:
        bon3 = lookup.get((model_name, "reasoning_best_of_3"))
        ss3 = lookup.get((model_name, "structured_sampling_3"))
        if bon3 and ss3:
            diff = ss3["accuracy"] - bon3["accuracy"]
            verdict = (
                "structured_sampling WINS" if diff > 0
                else "best_of_3 WINS" if diff < 0
                else "TIE"
            )
            print(
                f"  {model_name}: structured={ss3['accuracy']:.1%}  "
                f"best_of_3={bon3['accuracy']:.1%}  diff={diff:+.1%}  → {verdict}"
            )

    # Analysis: model strength
    if len(models) == 2:
        print("-" * 70)
        print("Analysis: stronger model vs current model")
        m0, m1 = models[0], models[1]
        for strat in strategies:
            s0 = lookup.get((m0, strat))
            s1 = lookup.get((m1, strat))
            if s0 and s1:
                diff = s1["accuracy"] - s0["accuracy"]
                print(
                    f"  strategy={strat}: {m0}={s0['accuracy']:.1%}  "
                    f"{m1}={s1['accuracy']:.1%}  diff={diff:+.1%}"
                )

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic: model strength vs sampling strategies on GSM8K"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=MAX_QUERIES_DEFAULT,
        help=(
            f"Max number of GSM8K queries to use (default: {MAX_QUERIES_DEFAULT}; "
            f"capped at {MAX_QUERIES_DEFAULT} per cost constraints)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory for output files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    max_queries: int = min(args.max_queries, MAX_QUERIES_DEFAULT)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"Loading GSM8K (max {max_queries} queries) ...")
    queries = load_gsm8k(split="test", max_samples=max_queries)
    print(f"  Loaded {len(queries)} queries.\n")

    # ------------------------------------------------------------------
    # 2. Run experiments: current model first, then stronger model
    # ------------------------------------------------------------------
    all_rows: list[dict] = []

    models_to_run = [CURRENT_MODEL, STRONGER_MODEL]
    for model_name in models_to_run:
        print(f"--- Model: {model_name} ---")
        try:
            rows = run_experiment(model_name, queries)
            all_rows.extend(rows)
            print(f"  Done. {len(rows)} result rows collected.\n")
        except ValueError:
            # Missing API key — already printed, stop entirely
            sys.exit(1)
        except RuntimeError as exc:
            # API / model-access error for *this* model
            print(f"ERROR running model '{model_name}': {exc}", file=sys.stderr)
            if model_name == CURRENT_MODEL:
                print(
                    "  Current model failed. Cannot continue without baseline results.",
                    file=sys.stderr,
                )
                sys.exit(1)
            else:
                print(
                    f"  Skipping stronger model '{model_name}' due to error above.\n",
                    file=sys.stderr,
                )

    if not all_rows:
        print("No results collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Compute metrics and save outputs
    # ------------------------------------------------------------------
    summaries = compute_metrics(all_rows)
    save_outputs(all_rows, summaries, output_dir)

    # ------------------------------------------------------------------
    # 4. Print summary
    # ------------------------------------------------------------------
    print_summary(summaries)


if __name__ == "__main__":
    main()
