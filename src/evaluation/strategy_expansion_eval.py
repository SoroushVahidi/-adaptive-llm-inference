"""Strategy expansion evaluation: five inference strategies compared on GSM8K.

Strategies implemented:
  1. direct_greedy         - one direct prompt, greedy decode
  2. reasoning_best_of_3   - one reasoning-style prompt, 3 samples, majority vote
  3. structured_sampling_3 - 3 different prompts, majority vote
  4. direct_plus_verify     - direct answer + verifier; corrects if verifier rejects
  5. direct_plus_revise     - direct answer + revision prompt; extracts final revised answer
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from src.utils.answer_extraction import extract_mc_answer, extract_numeric_answer

# ---------------------------------------------------------------------------
# Typing shim – we only need generate / generate_n from any model object
# ---------------------------------------------------------------------------

class _ModelProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...
    def generate_n(self, prompt: str, n: int) -> list[str]: ...


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM = "Answer the following math question. Give only the final numeric answer."
_REASONING_SYSTEM = (
    "Answer the following math question step-by-step. "
    "Show your work and end with 'Final answer: <number>'."
)

_PROMPTS_STRUCTURED = [
    # concise direct
    "Answer this question briefly. Give only the final numeric answer.\n\n{question}",
    # step-by-step reasoning
    "Solve this step by step and end with 'Final answer: <number>'.\n\n{question}",
    # solve then double-check
    (
        "Solve this question, then double-check your work. "
        "End with 'Final answer: <number>'.\n\n{question}"
    ),
]

_VERIFY_SYSTEM = (
    "You are a math checker. "
    "Given a question and a proposed answer, decide if the answer is correct. "
    "If it is wrong, reply: 'WRONG. Correct answer: <number>'. "
    "If it is correct, reply: 'CORRECT'."
)

_REVISE_SYSTEM = (
    "You are reviewing your own answer to a math question. "
    "If you see an error, provide the corrected numeric answer on the last line as "
    "'Final answer: <number>'. If the answer is fine, repeat it as 'Final answer: <number>'."
)


# ---------------------------------------------------------------------------
# Numeric normalization (same logic as self_consistency baseline)
# ---------------------------------------------------------------------------

def _normalize(value: str) -> str:
    candidate = value.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        number = Decimal(candidate)
    except InvalidOperation:
        return candidate
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _parse_model_answer(raw: str, answer_mode: str) -> str:
    """Normalize prediction for grading: numeric (default) or multiple-choice letter."""
    if answer_mode == "multiple_choice":
        letter = extract_mc_answer(raw)
        return letter.upper() if letter else ""
    return _normalize(extract_numeric_answer(raw))


def _majority_vote(answers: list[str]) -> str:
    if not answers:
        return ""
    normalized = [_normalize(a) for a in answers]
    counter: Counter[str] = Counter(normalized)
    return counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Individual strategy runners
# ---------------------------------------------------------------------------

def run_direct_greedy(
    model: _ModelProtocol,
    question: str,
) -> dict[str, Any]:
    """Strategy A: single direct answer."""
    raw = model.generate(question)
    answer = _normalize(extract_numeric_answer(raw))
    return {
        "raw_outputs": [raw],
        "predicted_answer": answer,
        "samples_used": 1,
    }


def run_reasoning_best_of_3(
    model: _ModelProtocol,
    question: str,
) -> dict[str, Any]:
    """Strategy B: reasoning-style prompt, 3 samples, majority vote."""
    prompt = f"Solve this step by step and end with 'Final answer: <number>'.\n\n{question}"
    raws = model.generate_n(prompt, 3)
    answers = [_normalize(extract_numeric_answer(r)) for r in raws]
    final = _majority_vote(answers)
    return {
        "raw_outputs": raws,
        "predicted_answer": final,
        "samples_used": 3,
    }


def run_structured_sampling_3(
    model: _ModelProtocol,
    question: str,
) -> dict[str, Any]:
    """Strategy C: 3 distinct prompts, majority vote."""
    raws: list[str] = []
    for template in _PROMPTS_STRUCTURED:
        prompt = template.format(question=question)
        raws.append(model.generate(prompt))
    answers = [_normalize(extract_numeric_answer(r)) for r in raws]
    final = _majority_vote(answers)
    return {
        "raw_outputs": raws,
        "predicted_answer": final,
        "samples_used": 3,
    }


def run_direct_plus_verify(
    model: _ModelProtocol,
    question: str,
) -> dict[str, Any]:
    """Strategy D: direct answer + verifier; accept correction if verifier rejects.

    The verifier is rule-based: we look for 'WRONG' in its response and, if
    found, extract any numeric answer it proposes as the corrected value.
    """
    # Step 1 – direct answer
    first_raw = model.generate(question)
    first_answer = _normalize(extract_numeric_answer(first_raw))

    # Step 2 – verification prompt
    verify_prompt = (
        f"Question: {question}\n"
        f"Proposed answer: {first_answer}\n\n"
        "Is this answer correct? If wrong, say 'WRONG. Correct answer: <number>'. "
        "If correct, say 'CORRECT'."
    )
    verify_raw = model.generate(verify_prompt)

    # Rule-based decision: only override if we see WRONG + a corrected number
    final_answer = first_answer
    verify_upper = verify_raw.upper()
    if "WRONG" in verify_upper:
        corrected = _normalize(extract_numeric_answer(verify_raw))
        if corrected:
            final_answer = corrected

    return {
        "raw_outputs": [first_raw, verify_raw],
        "predicted_answer": final_answer,
        "samples_used": 2,
        "first_answer": first_answer,
        "revised_answer": final_answer,
    }


# Shared reasoning-style prompt (matches ``run_reasoning_greedy`` in oracle_subset_eval).
_REASONING_CHAIN_PROMPT = (
    "Solve this step by step and end with 'Final answer: <number>'.\n\n{question}"
)
_REASONING_CHAIN_PROMPT_MC = (
    "Solve this step by step. The question lists choices (A) through (D). "
    "End with 'Final answer: (X)' where X is exactly one letter A, B, C, or D.\n\n"
    "{question}"
)


def run_reasoning_then_revise(
    model: _ModelProtocol,
    question: str,
    answer_mode: str = "numeric",
) -> dict[str, Any]:
    """Run chain-of-thought once, then a revise pass on question + reasoning + answer."""
    if answer_mode == "multiple_choice":
        tmpl = _REASONING_CHAIN_PROMPT_MC
    else:
        tmpl = _REASONING_CHAIN_PROMPT
    stage1_raw = model.generate(tmpl.format(question=question))
    first_answer = _parse_model_answer(stage1_raw, answer_mode)
    reasoning_block = stage1_raw.strip()
    max_ctx = 6000
    if len(reasoning_block) > max_ctx:
        reasoning_block = reasoning_block[:max_ctx] + "\n[...truncated...]"
    if answer_mode == "multiple_choice":
        final_fmt = (
            "End your response with 'Final answer: (X)' where X is A, B, C, or D. "
            "If the answer is already correct, repeat it."
        )
    else:
        final_fmt = (
            "End your response with 'Final answer: <number>'. "
            "If the answer is already correct, repeat it as 'Final answer: <number>'."
        )
    revise_prompt = (
        f"Question: {question}\n\n"
        f"Your first reasoning and answer were:\n{reasoning_block}\n\n"
        f"Your stated final answer was: {first_answer or '(none)'}\n\n"
        "Review the reasoning carefully. If you find an error, correct it. "
        f"{final_fmt}"
    )
    revise_raw = model.generate(revise_prompt)
    revised_answer = _parse_model_answer(revise_raw, answer_mode)
    if not revised_answer:
        revised_answer = first_answer

    return {
        "raw_outputs": [stage1_raw, revise_raw],
        "predicted_answer": revised_answer,
        "samples_used": 2,
        "first_answer": first_answer,
        "revised_answer": revised_answer,
    }


def run_self_consistency_3(
    model: _ModelProtocol,
    question: str,
    answer_mode: str = "numeric",
) -> dict[str, Any]:
    """Three independent reasoning samples; majority vote over normalized numeric answers."""
    if answer_mode == "multiple_choice":
        tmpl = _REASONING_CHAIN_PROMPT_MC
    else:
        tmpl = _REASONING_CHAIN_PROMPT
    prompt = tmpl.format(question=question)
    raws = model.generate_n(prompt, 3)
    answers = [_parse_model_answer(r, answer_mode) for r in raws]
    nonempty = [a for a in answers if a]
    if not nonempty:
        return {
            "raw_outputs": raws,
            "predicted_answer": "",
            "samples_used": 3,
            "self_consistency_ambiguous": True,
            "self_consistency_tied_values": "",
        }

    counter: Counter[str] = Counter(nonempty)
    top_freq = counter.most_common(1)[0][1]
    tied = sorted([a for a, c in counter.items() if c == top_freq])
    ambiguous = len(tied) > 1
    final = tied[0]

    return {
        "raw_outputs": raws,
        "predicted_answer": final,
        "samples_used": 3,
        "self_consistency_ambiguous": ambiguous,
        "self_consistency_tied_values": "|".join(tied) if ambiguous else "",
    }


def run_direct_plus_revise(
    model: _ModelProtocol,
    question: str,
    answer_mode: str = "numeric",
) -> dict[str, Any]:
    """Strategy E: direct answer + self-revision prompt.

    The revision prompt asks the model to review its own answer and provide a
    'Final answer: <number>' on the last line.
    """
    # Step 1 – direct answer
    first_raw = model.generate(question)
    first_answer = _parse_model_answer(first_raw, answer_mode)

    # Step 2 – revision prompt
    if answer_mode == "multiple_choice":
        tail = (
            "Please review your work. If you spot an error, correct it. "
            "End your response with 'Final answer: (X)' where X is A, B, C, or D."
        )
    else:
        tail = (
            "Please review your work. If you spot an error, correct it. "
            "End your response with 'Final answer: <number>'."
        )
    revise_prompt = (
        f"Question: {question}\n\n"
        f"Your previous answer was: {first_answer}\n\n"
        f"{tail}"
    )
    revised_raw = model.generate(revise_prompt)
    revised_answer = _parse_model_answer(revised_raw, answer_mode)
    if not revised_answer:
        revised_answer = first_answer  # fall back to first answer if revision fails

    return {
        "raw_outputs": [first_raw, revised_raw],
        "predicted_answer": revised_answer,
        "samples_used": 2,
        "first_answer": first_answer,
        "revised_answer": revised_answer,
    }


# ---------------------------------------------------------------------------
# Per-query runner dispatcher
# ---------------------------------------------------------------------------

_STRATEGY_RUNNERS = {
    "direct_greedy": run_direct_greedy,
    "reasoning_best_of_3": run_reasoning_best_of_3,
    "structured_sampling_3": run_structured_sampling_3,
    "direct_plus_verify": run_direct_plus_verify,
    "direct_plus_revise": run_direct_plus_revise,
    "reasoning_then_revise": run_reasoning_then_revise,
    "self_consistency_3": run_self_consistency_3,
}

ALL_STRATEGIES = list(_STRATEGY_RUNNERS.keys())


def run_strategy(
    strategy: str,
    model: _ModelProtocol,
    question: str,
) -> dict[str, Any]:
    """Dispatch to the correct strategy runner."""
    if strategy not in _STRATEGY_RUNNERS:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {ALL_STRATEGIES}")
    return _STRATEGY_RUNNERS[strategy](model, question)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_strategy_expansion_eval(
    model: _ModelProtocol,
    queries: list[Any],  # list of Query(id, question, answer)
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Run all (or a subset of) strategies over the provided queries.

    Args:
        model: Any object with ``generate(prompt)`` and ``generate_n(prompt, n)``.
        queries: List of objects with ``.id``, ``.question``, ``.answer`` fields.
        strategies: Which strategies to run. Defaults to ALL_STRATEGIES.

    Returns:
        Dict with per-query rows and per-strategy summaries.
    """
    if strategies is None:
        strategies = ALL_STRATEGIES

    per_query_rows: list[dict[str, Any]] = []

    for query in queries:
        for strategy in strategies:
            result = run_strategy(strategy, model, query.question)
            predicted = result["predicted_answer"]
            gold = _normalize(query.answer)
            correct = predicted == gold
            row: dict[str, Any] = {
                "question_id": query.id,
                "strategy": strategy,
                "predicted_answer": predicted,
                "gold_answer": gold,
                "correct": correct,
                "samples_used": result["samples_used"],
            }
            if "first_answer" in result:
                row["first_answer"] = result["first_answer"]
                row["revised_answer"] = result["revised_answer"]
            per_query_rows.append(row)

    # Compute per-strategy summaries
    strategy_summaries: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        rows = [r for r in per_query_rows if r["strategy"] == strategy]
        n = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        total_samples = sum(int(r["samples_used"]) for r in rows)
        strategy_summaries[strategy] = {
            "strategy": strategy,
            "accuracy": correct / n if n > 0 else 0.0,
            "correct": correct,
            "total_queries": n,
            "total_samples_used": total_samples,
            "avg_samples_per_query": total_samples / n if n > 0 else 0.0,
        }

    # Pairwise comparisons
    def _pairwise(base_strategy: str, target_strategy: str) -> dict[str, Any]:
        base_map = {
            r["question_id"]: r
            for r in per_query_rows
            if r["strategy"] == base_strategy
        }
        target_map = {
            r["question_id"]: r
            for r in per_query_rows
            if r["strategy"] == target_strategy
        }
        improved = worsened = 0
        for qid, trow in target_map.items():
            brow = base_map.get(qid)
            if brow is None:
                continue
            if trow["correct"] and not brow["correct"]:
                improved += 1
            elif not trow["correct"] and brow["correct"]:
                worsened += 1
        return {
            "baseline": base_strategy,
            "target": target_strategy,
            "queries_improved": improved,
            "queries_worsened": worsened,
            "net_gain": improved - worsened,
        }

    pairwise: list[dict[str, Any]] = []
    comparison_baselines = ["direct_greedy", "structured_sampling_3"]
    for base in comparison_baselines:
        if base not in strategies:
            continue
        for target in strategies:
            if target == base:
                continue
            pairwise.append(_pairwise(base, target))

    return {
        "per_query_rows": per_query_rows,
        "strategy_summaries": strategy_summaries,
        "pairwise_comparisons": pairwise,
        "total_queries": len(queries),
        "strategies_run": strategies,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_strategy_expansion_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Save summary.json, summary.csv and per_query_results.csv."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    # summary.json – full result object (minus per_query_rows to keep it concise)
    summary_payload = {
        "total_queries": result["total_queries"],
        "strategies_run": result["strategies_run"],
        "strategy_summaries": result["strategy_summaries"],
        "pairwise_comparisons": result["pairwise_comparisons"],
    }
    summary_json = base / "summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2))

    # summary.csv – one row per strategy
    summaries = list(result["strategy_summaries"].values())
    summary_csv = base / "summary.csv"
    if summaries:
        fieldnames = list(summaries[0].keys())
        with summary_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    # per_query_results.csv
    per_query_rows = result["per_query_rows"]
    per_query_csv = base / "per_query_results.csv"
    if per_query_rows:
        # Ensure all optional columns are present (fill missing with "")
        all_fields: list[str] = []
        for row in per_query_rows:
            for key in row:
                if key not in all_fields:
                    all_fields.append(key)
        with per_query_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            for row in per_query_rows:
                writer.writerow({k: row.get(k, "") for k in all_fields})

    return {
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "per_query_csv": str(per_query_csv),
    }


def format_strategy_expansion_summary(
    result: dict[str, Any],
    paths: dict[str, str],
) -> str:
    """Format a human-readable summary of the strategy comparison."""
    lines: list[str] = [
        "--- Strategy Expansion Evaluation ---",
        f"total queries: {result['total_queries']}",
        "",
        f"{'Strategy':<30} {'Accuracy':>8} {'Correct':>8} {'Samples':>8} {'Avg S/Q':>8}",
        "-" * 66,
    ]
    for summary in result["strategy_summaries"].values():
        lines.append(
            f"{summary['strategy']:<30} "
            f"{summary['accuracy']:>8.4f} "
            f"{summary['correct']:>8} "
            f"{summary['total_samples_used']:>8} "
            f"{summary['avg_samples_per_query']:>8.2f}"
        )

    lines += ["", "Pairwise improvements:"]
    for pw in result["pairwise_comparisons"]:
        lines.append(
            f"  {pw['target']:<30} vs {pw['baseline']:<30} "
            f"improved={pw['queries_improved']} worsened={pw['queries_worsened']} "
            f"net={pw['net_gain']:+d}"
        )

    lines += [
        "",
        f"summary_json:  {paths['summary_json']}",
        f"summary_csv:   {paths['summary_csv']}",
        f"per_query_csv: {paths['per_query_csv']}",
    ]
    return "\n".join(lines)
