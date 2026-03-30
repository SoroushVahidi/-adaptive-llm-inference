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
from typing import Any, Literal, Protocol

from src.utils.answer_extraction import (
    extract_math_answer,
    extract_mc_answer,
    extract_numeric_answer,
    normalize_math_answer,
)

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

_REASONING_THEN_REVISE_CHECK = (
    "Check the reasoning and final answer carefully. If incorrect, fix it. "
    "If correct, return the same answer."
)

# Aligns with ``oracle_subset_eval.REASONING_GREEDY_PROMPT`` for numeric GSM8K-style runs.
_REASONING_THEN_REVISE_STEP1_USER = (
    "Solve this step by step and end with 'Final answer: <number>'.\n\n{question}"
)

_REASONING_THEN_REVISE_STEP1_MC = (
    "Solve this step by step. The question lists choices (A) through (D). "
    "End with 'Final answer: (X)' where X is exactly one letter A, B, C, or D.\n\n"
    "{question}"
)

_REASONING_CHAIN_PROMPT = _REASONING_THEN_REVISE_STEP1_USER
_REASONING_CHAIN_PROMPT_MC = _REASONING_THEN_REVISE_STEP1_MC


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


def run_self_consistency_reasoning_n_numeric(
    model: _ModelProtocol,
    question: str,
    n: int,
) -> dict[str, Any]:
    """N reasoning samples (numeric final), majority vote."""
    prompt = f"Solve this step by step and end with 'Final answer: <number>'.\n\n{question}"
    raws = model.generate_n(prompt, n)
    answers = [_normalize(extract_numeric_answer(r)) for r in raws]
    final = _majority_vote(answers)
    return {
        "raw_outputs": raws,
        "predicted_answer": final,
        "samples_used": n,
    }


def run_self_consistency_reasoning_n_math(
    model: _ModelProtocol,
    question: str,
    n: int,
) -> dict[str, Any]:
    """N reasoning samples with MATH-style finals, majority vote on normalized math."""
    prompt = (
        "Solve this step by step. Put your final answer in \\boxed{...} "
        "or end with 'Final answer: ...'.\n\n"
        f"{question}"
    )
    raws = model.generate_n(prompt, n)
    answers: list[str] = []
    for r in raws:
        parsed = extract_math_answer(r).strip()
        answers.append(normalize_math_answer(parsed) if parsed else "")
    nonempty = [a for a in answers if a]
    final = _majority_vote(nonempty) if nonempty else ""
    return {
        "raw_outputs": raws,
        "predicted_answer": final,
        "samples_used": n,
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


def run_reasoning_greedy(
    model: _ModelProtocol,
    question: str,
) -> dict[str, Any]:
    """Single reasoning-style pass (chain-of-thought prompt), one sample."""
    prompt = f"Solve this step by step and end with 'Final answer: <number>'.\n\n{question}"
    raw = model.generate(prompt)
    answer = _normalize(extract_numeric_answer(raw))
    return {
        "raw_outputs": [raw],
        "predicted_answer": answer,
        "samples_used": 1,
    }


def _final_from_model_output(text: str) -> str:
    """Prefer MATH-style extraction, fall back to numeric."""
    parsed = extract_math_answer(text).strip()
    if parsed:
        return normalize_math_answer(parsed)
    return _normalize(extract_numeric_answer(text))


def _parse_model_answer(raw: str, answer_mode: str) -> str:
    if answer_mode == "multiple_choice":
        letter = extract_mc_answer(raw)
        return letter.upper() if letter else ""
    return _final_from_model_output(raw)


def run_direct_plus_revise(
    model: _ModelProtocol,
    question: str,
    answer_mode: str = "numeric",
) -> dict[str, Any]:
    """Strategy E: direct answer + self-revision prompt."""
    first_raw = model.generate(question)
    if answer_mode == "multiple_choice":
        first_answer = _parse_model_answer(first_raw, answer_mode)
    else:
        first_answer = _normalize(extract_numeric_answer(first_raw))
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
    if answer_mode == "multiple_choice":
        revised_answer = _parse_model_answer(revised_raw, answer_mode)
    else:
        revised_answer = _normalize(extract_numeric_answer(revised_raw))
    if not revised_answer:
        revised_answer = first_answer
    return {
        "raw_outputs": [first_raw, revised_raw],
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
    """Three reasoning samples; majority vote (numeric/math or MC letter)."""
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


def run_reasoning_then_revise(
    model: _ModelProtocol,
    question: str,
    *,
    revise_model: _ModelProtocol | None = None,
    answer_mode: str = "numeric",
) -> dict[str, Any]:
    """Reasoning pass then a second pass that sees question + full reasoning.

    Step 1: one chain-of-thought style generation (uses *model*'s system prefix).
    Step 2: verifier prompt with original question and the full reasoning output;
    uses *revise_model* when provided, else ``model.with_prompt_prefix`` when
    available, else the same *model* (weaker: shared system prefix).
    """
    step1_tmpl = (
        _REASONING_THEN_REVISE_STEP1_MC
        if answer_mode == "multiple_choice"
        else _REASONING_THEN_REVISE_STEP1_USER
    )
    reasoning_raw = model.generate(step1_tmpl.format(question=question))
    reasoning_answer = _parse_model_answer(reasoning_raw, answer_mode)

    if answer_mode == "multiple_choice":
        end_instr = (
            "End with 'Final answer: (X)' where X is exactly one letter A, B, C, or D."
        )
    else:
        end_instr = (
            "End with a clear final answer using 'Final answer: ...' or \\boxed{{...}}."
        )

    revise_user = (
        f"Question:\n{question}\n\n"
        f"Reasoning and draft answer:\n{reasoning_raw}\n\n"
        f"{_REASONING_THEN_REVISE_CHECK}\n"
        f"{end_instr}"
    )

    mrev: _ModelProtocol
    if revise_model is not None:
        mrev = revise_model
    else:
        wm = getattr(model, "with_prompt_prefix", None)
        if callable(wm):
            mrev = wm(
                "You verify step-by-step math reasoning. "
                "Check the reasoning and final answer carefully. If incorrect, fix it. "
                "If correct, return the same answer."
            )
        else:
            mrev = model

    revised_raw = mrev.generate(revise_user)
    revised_answer = _parse_model_answer(revised_raw, answer_mode)
    if not revised_answer:
        revised_answer = reasoning_answer

    return {
        "raw_outputs": [reasoning_raw, revised_raw],
        "predicted_answer": revised_answer,
        "samples_used": 2,
        "first_answer": reasoning_answer,
        "revised_answer": revised_answer,
        "reasoning_raw": reasoning_raw,
    }


def run_reasoning_then_revise_review_only(
    revise_model: _ModelProtocol,
    question: str,
    prior_reasoning_raw: str,
    *,
    mode: Literal["numeric", "math"] = "numeric",
) -> dict[str, Any]:
    """Second stage only: reuse a frozen reasoning trace from an earlier run."""
    revise_user = (
        f"Question:\n{question}\n\n"
        f"Reasoning and draft answer:\n{prior_reasoning_raw}\n\n"
        f"{_REASONING_THEN_REVISE_CHECK}\n"
        "End with a clear final answer using 'Final answer: ...' or \\boxed{{...}}."
    )
    revised_raw = revise_model.generate(revise_user)
    if mode == "math":
        parsed = extract_math_answer(revised_raw).strip()
        revised_answer = normalize_math_answer(parsed) if parsed else ""
    else:
        revised_answer = _normalize(extract_numeric_answer(revised_raw))
    prior_ans = (
        normalize_math_answer(extract_math_answer(prior_reasoning_raw).strip() or "")
        if mode == "math"
        else _normalize(extract_numeric_answer(prior_reasoning_raw))
    )
    if not revised_answer:
        revised_answer = prior_ans

    return {
        "raw_outputs": [revised_raw],
        "predicted_answer": revised_answer,
        "samples_used": 1,
        "first_answer": prior_ans,
        "revised_answer": revised_answer,
        "reasoning_raw": prior_reasoning_raw,
    }


# ---------------------------------------------------------------------------
# Per-query runner dispatcher
# ---------------------------------------------------------------------------

_STRATEGY_RUNNERS = {
    "direct_greedy": run_direct_greedy,
    "reasoning_greedy": run_reasoning_greedy,
    "reasoning_best_of_3": run_reasoning_best_of_3,
    "reasoning_then_revise": run_reasoning_then_revise,
    "structured_sampling_3": run_structured_sampling_3,
    "direct_plus_verify": run_direct_plus_verify,
    "direct_plus_revise": run_direct_plus_revise,
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
