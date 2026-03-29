"""Recent baselines evaluation: static ladder, oracle bounds, lightweight routing.

Used by ``scripts/run_recent_baselines_experiment.py`` to produce
``outputs/recent_baselines/*.json`` and cross-dataset CSV rollups.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol

from src.baselines.base import BaselineResult
from src.evaluation.strategy_expansion_eval import run_direct_plus_revise
from src.features.precompute_features import extract_first_pass_features, extract_query_features
from src.utils.answer_extraction import (
    extract_math_answer,
    extract_numeric_answer,
    normalize_math_answer,
)

AnswerMode = Literal["numeric", "math"]

REASONING_PROMPT_NUMERIC = (
    "Solve this step by step and end with 'Final answer: <number>'.\n\n{question}"
)
REASONING_PROMPT_MATH = (
    "Solve this step by step. Give the final answer in \\boxed{{...}} using valid LaTeX "
    "(e.g. \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}).\n\n{question}"
)


def _reasoning_user_prompt(question: str, mode: AnswerMode) -> str:
    if mode == "math":
        return REASONING_PROMPT_MATH.format(question=question)
    return REASONING_PROMPT_NUMERIC.format(question=question)


REVISE_AFTER_REASONING_PROMPT = (
    "You previously solved the following question with step-by-step reasoning.\n\n"
    "Question:\n{question}\n\n"
    "Your full previous response (reasoning and answer):\n{first_raw}\n\n"
    "Carefully re-check every step. If you find an error, correct it. "
    "If the solution is already correct, restate the final answer unchanged.\n"
    "End with a single line: Final answer: <answer> "
    "(use the same final format as before, e.g. boxed fraction or integer).\n"
)


class _Model(Protocol):
    def generate(self, prompt: str) -> str: ...

    def generate_n(self, prompt: str, n: int) -> list[str]: ...


def _norm_numeric(value: str) -> str:
    candidate = value.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        number = Decimal(candidate)
    except InvalidOperation:
        return candidate
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _normalize_pred(mode: AnswerMode, text: str) -> str:
    if mode == "numeric":
        return _norm_numeric(extract_numeric_answer(text))
    return normalize_math_answer(extract_math_answer(text))


def _normalize_gold(mode: AnswerMode, gold: str) -> str:
    g = gold.strip()
    if mode == "numeric":
        return _norm_numeric(g)
    return normalize_math_answer(g)


STATIC_LADDER_BASELINES: list[str] = [
    "reasoning_greedy",
    "self_consistency_3",
    "self_consistency_5",
    "direct_plus_revise",
    "reasoning_then_revise",
    "always_most_expensive",
]

COST_PROXY: dict[str, float] = {
    "reasoning_greedy": 1.0,
    "direct_plus_revise": 2.0,
    "reasoning_then_revise": 2.0,
    "self_consistency_3": 3.0,
    "self_consistency_5": 5.0,
    "always_most_expensive": 5.0,
}

ORACLE_ACTIONS: list[str] = [
    "reasoning_greedy",
    "direct_plus_revise",
    "reasoning_then_revise",
    "self_consistency_3",
    "self_consistency_5",
]


def run_reasoning_greedy_mode(model: _Model, question: str, mode: AnswerMode) -> dict[str, Any]:
    raw = model.generate(_reasoning_user_prompt(question, mode))
    pred = _normalize_pred(mode, raw)
    return {
        "raw_outputs": [raw],
        "predicted_answer": pred,
        "samples_used": 1,
    }


def run_direct_plus_revise_mode(model: _Model, question: str, mode: AnswerMode) -> dict[str, Any]:
    first_raw = model.generate(question)
    first_pred = _normalize_pred(mode, first_raw)
    revise_prompt = (
        f"Question: {question}\n\n"
        f"Your previous answer was: {first_pred}\n\n"
        "Please review your work. If you spot an error, correct it. "
        "End with your final answer in \\boxed{...} (math mode) or "
        "'Final answer: <number>' (numeric word problem).\n"
    )
    revised_raw = model.generate(revise_prompt)
    revised_pred = _normalize_pred(mode, revised_raw)
    if not revised_pred:
        revised_pred = first_pred
    return {
        "raw_outputs": [first_raw, revised_raw],
        "predicted_answer": revised_pred,
        "samples_used": 2,
        "first_answer": first_pred,
        "revised_answer": revised_pred,
    }


def run_reasoning_then_revise(model: _Model, question: str, mode: AnswerMode) -> dict[str, Any]:
    first_raw = model.generate(_reasoning_user_prompt(question, mode))
    first_pred = _normalize_pred(mode, first_raw)
    revise_prompt = REVISE_AFTER_REASONING_PROMPT.format(
        question=question,
        first_raw=first_raw[:12000],
    )
    revised_raw = model.generate(revise_prompt)
    revised_pred = _normalize_pred(mode, revised_raw)
    if not revised_pred:
        revised_pred = first_pred
    revised_helped = first_pred != revised_pred
    return {
        "raw_outputs": [first_raw, revised_raw],
        "predicted_answer": revised_pred,
        "first_answer": first_pred,
        "revised_answer": revised_pred,
        "samples_used": 2,
        "revise_changed_output": revised_helped,
    }


def run_self_consistency_n(
    model: _Model,
    question: str,
    n: int,
    mode: AnswerMode,
) -> dict[str, Any]:
    prompt = _reasoning_user_prompt(question, mode)
    raws = model.generate_n(prompt, n)
    extracted = [_normalize_pred(mode, r) for r in raws]
    counter = Counter(extracted)
    most_common = counter.most_common()
    top_count = most_common[0][1] if most_common else 0
    top_val = most_common[0][0] if most_common else ""
    tied = [v for v, c in most_common if c == top_count and top_count > 0]
    ambiguous = len(tied) > 1
    chosen = sorted(tied)[0] if tied else top_val
    return {
        "raw_outputs": raws,
        "predicted_answer": chosen,
        "samples_used": n,
        "vote_counts": dict(counter),
        "self_consistency_ambiguous": ambiguous,
        "self_consistency_tied_answers": tied if ambiguous else [],
    }


def run_static_baseline(
    name: str,
    model: _Model,
    question: str,
    mode: AnswerMode,
) -> dict[str, Any]:
    if name == "reasoning_greedy":
        return run_reasoning_greedy_mode(model, question, mode)
    if name == "direct_plus_revise":
        if mode == "math":
            return run_direct_plus_revise_mode(model, question, mode)
        out = run_direct_plus_revise(model, question)
        pred = _normalize_pred(mode, out["predicted_answer"])
        return {**out, "predicted_answer": pred}
    if name == "reasoning_then_revise":
        return run_reasoning_then_revise(model, question, mode)
    if name == "self_consistency_3":
        return run_self_consistency_n(model, question, 3, mode)
    if name == "self_consistency_5":
        return run_self_consistency_n(model, question, 5, mode)
    if name == "always_most_expensive":
        return run_self_consistency_n(model, question, 5, mode)
    raise ValueError(f"Unknown static baseline: {name}")


def baseline_result_from_run(
    query: Any,
    baseline_name: str,
    run: dict[str, Any],
    mode: AnswerMode,
) -> BaselineResult:
    gold = _normalize_gold(mode, query.answer)
    pred = run["predicted_answer"]
    if mode == "numeric":
        correct = _norm_numeric(pred) == gold if pred else False
    else:
        correct = normalize_math_answer(pred) == gold if pred else False

    meta: dict[str, Any] = {
        "baseline_name": baseline_name,
        "cost_proxy": COST_PROXY.get(baseline_name, float(run.get("samples_used", 1))),
    }
    if "first_answer" in run:
        meta["first_answer"] = run["first_answer"]
        meta["revised_answer"] = run["revised_answer"]
        fa, ra = run.get("first_answer", ""), run.get("revised_answer", "")
        meta["revise_changed_prediction"] = fa != ra
        if baseline_name == "direct_plus_revise":
            meta["revise_helpful"] = (not _match_gold(mode, fa, query.answer)) and _match_gold(
                mode, ra, query.answer
            )
        if baseline_name == "reasoning_then_revise":
            meta["revise_helpful"] = (not _match_gold(mode, fa, query.answer)) and _match_gold(
                mode, ra, query.answer
            )
    if run.get("self_consistency_ambiguous"):
        meta["self_consistency_ambiguous"] = True
        meta["self_consistency_tied_answers"] = run.get("self_consistency_tied_answers", [])
    if "vote_counts" in run:
        meta["vote_counts"] = run["vote_counts"]

    return BaselineResult(
        query_id=query.id,
        question=query.question,
        candidates=run.get("raw_outputs", []),
        final_answer=pred,
        ground_truth=gold,
        correct=correct,
        samples_used=int(run.get("samples_used", 1)),
        metadata=meta,
    )


def _row_cost_proxy(row: dict[str, Any]) -> float:
    meta = row.get("metadata") or {}
    return float(meta.get("cost_proxy", row.get("samples_used", 1)))


def _match_gold(mode: AnswerMode, pred: str, gold_raw: str) -> bool:
    g = _normalize_gold(mode, gold_raw)
    if mode == "numeric":
        return _norm_numeric(pred) == g if pred else False
    return normalize_math_answer(pred) == g if pred else False


def summarize_static_results(
    baseline_name: str,
    results: list[BaselineResult],
) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"baseline_name": baseline_name, "accuracy": 0.0, "avg_cost_proxy": 0.0}

    correct = sum(1 for r in results if r.correct)
    cost = sum(r.metadata.get("cost_proxy", r.samples_used) for r in results)
    if baseline_name == "reasoning_then_revise":
        revise_rate = sum(1 for r in results if r.metadata.get("revise_changed_prediction")) / n
    elif baseline_name == "direct_plus_revise":
        revise_rate = sum(1 for r in results if len(r.candidates) >= 2) / n
    else:
        revise_rate = 0.0

    rh = [r for r in results if r.metadata.get("revise_helpful") is True]
    _rev = ("direct_plus_revise", "reasoning_then_revise")
    revise_helpful_rate = len(rh) / n if baseline_name in _rev else None

    amb = sum(1 for r in results if r.metadata.get("self_consistency_ambiguous"))
    amb_rate = amb / n if baseline_name.startswith("self_consistency") else None

    return {
        "baseline_name": baseline_name,
        "accuracy": correct / n,
        "avg_cost_proxy": cost / n,
        "total_queries": n,
        "extra_compute_or_revise_rate": revise_rate,
        "revise_helpful_rate": revise_helpful_rate,
        "self_consistency_ambiguity_rate": amb_rate,
        "ambiguous_query_count": amb if amb_rate is not None else 0,
    }


def evaluate_static_ladder(
    model: _Model,
    queries: list[Any],
    mode: AnswerMode,
    baselines: list[str] | None = None,
) -> dict[str, Any]:
    baselines = baselines or STATIC_LADDER_BASELINES
    per_baseline: dict[str, list[BaselineResult]] = {b: [] for b in baselines}

    for q in queries:
        for b in baselines:
            run = run_static_baseline(b, model, q.question, mode)
            per_baseline[b].append(baseline_result_from_run(q, b, run, mode))

    summaries = {b: summarize_static_results(b, per_baseline[b]) for b in baselines}
    return {
        "per_baseline_results": {k: [asdict(r) for r in v] for k, v in per_baseline.items()},
        "summaries": summaries,
    }


def _find_row(
    per_baseline_results: dict[str, list[dict[str, Any]]],
    action: str,
    qid: str,
) -> dict[str, Any] | None:
    for r in per_baseline_results.get(action, []):
        if r["query_id"] == qid:
            return r
    return None


def _pick_oracle_action(
    per_baseline_results: dict[str, list[dict[str, Any]]],
    qid: str,
    actions: tuple[str, ...],
    prefer_cheapest_on_tie: bool,
) -> tuple[str, dict[str, Any] | None]:
    """Pick best action by accuracy; tie-break by cost (cheapest if *prefer_cheapest_on_tie*)."""
    candidates: list[tuple[str, bool, float]] = []
    for action in actions:
        row = _find_row(per_baseline_results, action, qid)
        if row is None:
            continue
        ok = bool(row["correct"])
        cost = float(row.get("metadata", {}).get("cost_proxy", row.get("samples_used", 1)))
        candidates.append((action, ok, cost))
    if not candidates:
        return "", None
    any_ok = any(c[1] for c in candidates)
    pool = [c for c in candidates if c[1]] if any_ok else candidates
    best_ok = max(c[1] for c in pool)
    pool = [c for c in pool if c[1] == best_ok]
    if prefer_cheapest_on_tie:
        pool.sort(key=lambda t: (t[2], t[0]))
    else:
        pool.sort(key=lambda t: (-t[2], t[0]))
    chosen = pool[0][0]
    return chosen, _find_row(per_baseline_results, chosen, qid)


def compute_oracle_summaries(
    per_baseline_results: dict[str, list[dict[str, Any]]],
    lambdas: tuple[float, ...] = (0.0, 0.1, 0.25),
) -> dict[str, Any]:
    """Oracle picks per query from precomputed static baseline outcomes."""
    ids: set[str] = set()
    for rows in per_baseline_results.values():
        for r in rows:
            ids.add(r["query_id"])
    sorted_ids = sorted(ids)

    binary_actions = ("reasoning_greedy", "direct_plus_revise")
    action_freq_binary: Counter[str] = Counter()
    correct_binary = 0
    cost_binary = 0.0

    multi_freq: Counter[str] = Counter()
    correct_multi = 0
    cost_multi = 0.0

    utility_rows: dict[str, dict[str, Any]] = {}

    for qid in sorted_ids:
        act_b, row_b = _pick_oracle_action(
            per_baseline_results, qid, binary_actions, prefer_cheapest_on_tie=True
        )
        if row_b:
            action_freq_binary[act_b] += 1
            correct_binary += 1 if row_b["correct"] else 0
            cost_binary += _row_cost_proxy(row_b)

        act_m, row_m = _pick_oracle_action(
            per_baseline_results, qid, tuple(ORACLE_ACTIONS), prefer_cheapest_on_tie=True
        )
        if row_m:
            multi_freq[act_m] += 1
            correct_multi += 1 if row_m["correct"] else 0
            cost_multi += _row_cost_proxy(row_m)

    n = len(sorted_ids)
    binary_summary = {
        "oracle_accuracy": correct_binary / n if n else 0.0,
        "oracle_avg_cost_proxy": cost_binary / n if n else 0.0,
        "action_frequencies": dict(action_freq_binary),
    }
    multi_summary = {
        "oracle_accuracy": correct_multi / n if n else 0.0,
        "oracle_avg_cost_proxy": cost_multi / n if n else 0.0,
        "action_frequencies": dict(multi_freq),
    }

    for lam in lambdas:
        utility_freq: Counter[str] = Counter()
        util_correct = 0
        util_cost = 0.0
        for qid in sorted_ids:
            best_action = ""
            best_score = -1e9
            best_cost = 0.0
            for action in ORACLE_ACTIONS:
                row = _find_row(per_baseline_results, action, qid)
                if row is None:
                    continue
                c = 1.0 if row["correct"] else 0.0
                cost = _row_cost_proxy(row)
                score = c - lam * cost
                if score > best_score or (
                    math.isclose(score, best_score) and (best_action == "" or cost < best_cost)
                ):
                    best_score = score
                    best_action = action
                    best_cost = cost
            if not best_action:
                continue
            utility_freq[best_action] += 1
            row = _find_row(per_baseline_results, best_action, qid)
            if row:
                util_correct += 1 if row["correct"] else 0
                util_cost += _row_cost_proxy(row)
        utility_rows[str(lam)] = {
            "lambda": lam,
            "oracle_accuracy": util_correct / n if n else 0.0,
            "oracle_avg_cost_proxy": util_cost / n if n else 0.0,
            "utility_correct_sum": util_correct,
            "action_frequencies": dict(utility_freq),
        }

    return {
        "binary_oracle": binary_summary,
        "multi_action_oracle": multi_summary,
        "cost_aware_utility_oracles": utility_rows,
        "total_queries": n,
    }


def _difficulty_proxy(question: str) -> float:
    f = extract_query_features(question)
    return (
        0.01 * float(f["question_length_chars"])
        + 2.0 * float(f["num_numeric_mentions"])
        + 5.0 * float(f["has_multi_step_cue"])
    )


def _first_pass_signals(
    question: str,
    first_raw: str,
    first_pred: str,
    mode: AnswerMode,
) -> dict[str, Any]:
    fp = extract_first_pass_features(question, first_raw, parsed_answer=first_pred or None)
    nums_out = fp["first_pass_num_numeric_mentions"]
    tail = first_raw[-400:] if first_raw else ""
    tail_pred = _normalize_pred(mode, tail)
    instability = 0
    if first_pred and tail_pred and first_pred != tail_pred:
        instability = 1
    mismatch = 0
    qnums = extract_query_features(question)["num_numeric_mentions"]
    if qnums >= 3 and nums_out <= 1 and fp["first_pass_output_length"] < 80:
        mismatch = 1
    conf_proxy = 1.0 / (1.0 + float(bool(fp["first_pass_has_uncertainty_phrase"])))
    return {
        **fp,
        "extraction_tail_mismatch": instability,
        "reasoning_length_short_vs_question": mismatch,
        "confidence_proxy": conf_proxy,
        "difficulty_proxy": _difficulty_proxy(question),
    }


def evaluate_routing_baselines(
    model: _Model,
    queries: list[Any],
    mode: AnswerMode,
    per_query_reasoning: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Threshold routers using first reasoning pass (no extra cost before route)."""
    curves_a: list[dict[str, Any]] = []
    curves_b: list[dict[str, Any]] = []
    curves_c: list[dict[str, Any]] = []

    for thresh in [0.0, 0.5, 1.0, 1.5, 2.5]:
        correct = 0
        cost = 0.0
        escalated = 0
        for q in queries:
            info = per_query_reasoning[q.id]
            raw = str(info.get("first_raw", ""))
            first_pred = str(info.get("first_pred", ""))
            sig = _first_pass_signals(q.question, raw, first_pred, mode)
            cheap_ok = _match_gold(mode, first_pred, q.answer)
            route = sig["difficulty_proxy"] >= thresh or not sig["first_pass_parse_success"]
            if route:
                escalated += 1
                if mode == "math":
                    run = run_direct_plus_revise_mode(model, q.question, mode)
                else:
                    run = run_direct_plus_revise(model, q.question)
                    pred_norm = _normalize_pred(mode, run["predicted_answer"])
                    run = {**run, "predicted_answer": pred_norm}
                pred = run["predicted_answer"]
                cost += 2.0
                correct += 1 if _match_gold(mode, pred, q.answer) else 0
            else:
                cost += 1.0
                correct += 1 if cheap_ok else 0
        n = len(queries)
        curves_a.append(
            {
                "threshold": thresh,
                "accuracy": correct / n if n else 0.0,
                "avg_cost_proxy": cost / n if n else 0.0,
                "escalation_rate": escalated / n if n else 0.0,
            }
        )

    for thresh in [0.25, 0.5, 0.75]:
        correct = 0
        cost = 0.0
        escalated = 0
        for q in queries:
            info = per_query_reasoning[q.id]
            raw = str(info.get("first_raw", ""))
            first_pred = str(info.get("first_pred", ""))
            sig = _first_pass_signals(q.question, raw, first_pred, mode)
            cheap_ok = _match_gold(mode, first_pred, q.answer)
            score = (
                (1.0 - sig["confidence_proxy"])
                + 0.5 * float(sig["extraction_tail_mismatch"])
                + 0.5 * float(sig["reasoning_length_short_vs_question"])
            )
            route = score >= thresh
            if route:
                escalated += 1
                run = run_self_consistency_n(model, q.question, 3, mode)
                pred = run["predicted_answer"]
                cost += 3.0
                correct += 1 if _match_gold(mode, pred, q.answer) else 0
            else:
                cost += 1.0
                correct += 1 if cheap_ok else 0
        n = len(queries)
        curves_b.append(
            {
                "threshold": thresh,
                "accuracy": correct / n if n else 0.0,
                "avg_cost_proxy": cost / n if n else 0.0,
                "escalation_rate": escalated / n if n else 0.0,
            }
        )

    for thresh in [0.5, 1.5, 2.5]:
        correct = 0
        cost = 0.0
        for q in queries:
            info = per_query_reasoning[q.id]
            raw = str(info.get("first_raw", ""))
            first_pred = str(info.get("first_pred", ""))
            sig = _first_pass_signals(q.question, raw, first_pred, mode)
            d = float(sig["difficulty_proxy"])
            u = float(sig["confidence_proxy"])
            if d >= thresh and u < 0.75:
                run = run_self_consistency_n(model, q.question, 3, mode)
                pred = run["predicted_answer"]
                cost += 3.0
                correct += 1 if _match_gold(mode, pred, q.answer) else 0
            elif d >= thresh:
                run = run_reasoning_then_revise(model, q.question, mode)
                pred = run["predicted_answer"]
                cost += 2.0
                correct += 1 if _match_gold(mode, pred, q.answer) else 0
            else:
                cost += 1.0
                correct += 1 if _match_gold(mode, first_pred, q.answer) else 0
        n = len(queries)
        curves_c.append(
            {
                "difficulty_threshold": thresh,
                "accuracy": correct / n if n else 0.0,
                "avg_cost_proxy": cost / n if n else 0.0,
            }
        )

    def _best(curves: list[dict[str, Any]], key: str = "accuracy") -> dict[str, Any] | None:
        if not curves:
            return None
        return max(curves, key=lambda r: (r[key], -r.get("avg_cost_proxy", 0)))

    return {
        "baseline_A_confidence_difficulty_threshold": {
            "description": "Route to direct_plus_revise when difficulty_proxy>=T or parse fail",
            "cost_accuracy_curve": curves_a,
            "best_accuracy_point": _best(curves_a),
        },
        "baseline_B_output_aware": {
            "description": "Score from (1-confidence)+mismatch flags; route to self_consistency_3",
            "cost_accuracy_curve": curves_b,
            "best_accuracy_point": _best(curves_b),
        },
        "baseline_C_best_route_inspired_ladder": {
            "description": (
                "Ladder: reasoning_greedy, then reasoning_then_revise, "
                "then self_consistency_3 (difficulty + confidence)."
            ),
            "cost_accuracy_curve": curves_c,
            "best_accuracy_point": _best(curves_c),
        },
        "note": "BEST-Route paper not reproduced; lightweight threshold ladder only.",
    }


def hard_gsm8k_validation_summary(
    easy_queries: list[Any],
    hard_queries: list[Any],
    easy_one_shot_acc: float,
    hard_one_shot_acc: float,
) -> dict[str, Any]:
    def _avg_feats(qs: list[Any]) -> dict[str, float]:
        if not qs:
            return {}
        acc: dict[str, list[float]] = {}
        for q in qs:
            f = extract_query_features(q.question)
            for k, v in f.items():
                acc.setdefault(k, []).append(float(v))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    return {
        "easy_slice_size": len(easy_queries),
        "hard_slice_size": len(hard_queries),
        "one_shot_reasoning_greedy_accuracy_easy": easy_one_shot_acc,
        "one_shot_reasoning_greedy_accuracy_hard": hard_one_shot_acc,
        "hard_is_lower_accuracy": hard_one_shot_acc < easy_one_shot_acc,
        "avg_features_easy": _avg_feats(easy_queries),
        "avg_features_hard": _avg_feats(hard_queries),
    }
