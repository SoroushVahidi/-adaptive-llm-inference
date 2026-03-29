"""Next-stage EAAI experiments: oracle routing, budget curves, cascade, baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from src.evaluation.strategy_expansion_eval import (
    run_direct_plus_revise,
    run_self_consistency_reasoning_n_math,
    run_self_consistency_reasoning_n_numeric,
)
from src.models.openai_llm import OpenAILLMModel
from src.utils.answer_extraction import normalize_math_answer


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x: Any) -> float:
    try:
        return float(str(x).strip() or 0)
    except ValueError:
        return 0.0


def _i(x: Any) -> int:
    try:
        return int(float(str(x).strip()))
    except ValueError:
        return 0


def match_gold(gold: str, pred: str, mode: Literal["numeric", "math"]) -> bool:
    if not pred or not str(pred).strip():
        return False
    if mode == "math":
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


def oracle_revise_helpful_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Oracle: revise iff revise_helpful==1; else keep reasoning outcome."""
    n = len(rows)
    if n == 0:
        return {"accuracy": 0.0, "avg_cost": 0.0, "revise_rate": 0.0, "n": 0}
    correct = 0
    cost = 0.0
    rev = 0
    for r in rows:
        h = _i(r.get("revise_helpful", 0))
        if h:
            rev += 1
            cost += 2.0
            correct += _i(r.get("revise_correct", 0))
        else:
            cost += 1.0
            correct += _i(r.get("reasoning_correct", 0))
    return {
        "accuracy": correct / n,
        "avg_cost": cost / n,
        "revise_rate": rev / n,
        "n": n,
    }


def budget_curve_marginal_gain(
    rows: list[dict[str, Any]],
    cost_targets: list[float],
) -> list[dict[str, Any]]:
    """Greedy: revise queries with highest (revise_correct - reasoning_correct) first."""
    n = len(rows)
    if n == 0:
        return []

    gains: list[tuple[int, int, int]] = []
    for i, r in enumerate(rows):
        rc = _i(r.get("reasoning_correct", 0))
        vc = _i(r.get("revise_correct", 0))
        gains.append((vc - rc, i, rc, vc))
    gains.sort(key=lambda x: -x[0])

    out: list[dict[str, Any]] = []
    for target in cost_targets:
        if target <= 1.0 + 1e-9:
            k = 0
        elif target >= 2.0 - 1e-9:
            k = n
        else:
            k = min(n, max(0, round(n * (target - 1.0))))
        revised = {g[1] for g in gains[:k]}
        correct = 0
        for i, r in enumerate(rows):
            if i in revised:
                correct += _i(r.get("revise_correct", 0))
            else:
                correct += _i(r.get("reasoning_correct", 0))
        out.append(
            {
                "target_avg_cost": target,
                "achieved_avg_cost": 1.0 + k / n,
                "revise_rate": k / n,
                "accuracy": correct / n,
            }
        )
    return out


def cascade_curve(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    *,
    higher_confidence_skip_revise: bool = True,
) -> list[dict[str, Any]]:
    """Revise when unified_confidence_score < threshold (lower confidence → more revise)."""
    n = len(rows)
    out: list[dict[str, Any]] = []
    for t in thresholds:
        correct = 0
        cost = 0.0
        rev = 0
        for r in rows:
            conf = _f(r.get("unified_confidence_score", 0.0))
            do_rev = conf < t if higher_confidence_skip_revise else conf >= t
            if do_rev:
                rev += 1
                cost += 2.0
                correct += _i(r.get("revise_correct", 0))
            else:
                cost += 1.0
                correct += _i(r.get("reasoning_correct", 0))
        out.append(
            {
                "threshold": t,
                "accuracy": correct / n if n else 0.0,
                "avg_cost": cost / n if n else 0.0,
                "revise_rate": rev / n if n else 0.0,
            }
        )
    return out


def best_policy_v6_v7_from_eval_summary(policy_summary_path: Path) -> tuple[str, float, float]:
    data = json.loads(policy_summary_path.read_text(encoding="utf-8"))
    comp = data.get("comparison", [])
    best = ""
    best_acc = -1.0
    best_cost = 0.0
    for row in comp:
        name = str(row.get("route", ""))
        if name not in ("adaptive_policy_v6", "adaptive_policy_v7"):
            continue
        acc = float(row.get("accuracy", 0))
        cost = float(row.get("avg_cost", 0))
        if acc > best_acc or (acc == best_acc and cost < best_cost):
            best_acc = acc
            best_cost = cost
            best = name
    return best, best_acc, best_cost


def run_bon_and_direct_strategies_on_queries(
    queries: list[tuple[str, str, str]],
    *,
    mode: Literal["numeric", "math"],
    model_name: str,
    max_tokens: int,
    timeout: float,
    reasoning_prefix: str,
    direct_prefix: str = "Answer the following math question. Give only the final numeric answer.",
    ns: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    """Run self-consistency and direct_plus_revise on a list of (id, question, gold)."""
    model_r = OpenAILLMModel(
        model_name=model_name,
        greedy_temperature=0.0,
        sample_temperature=0.7,
        max_tokens=max_tokens,
        timeout_seconds=timeout,
        prompt_prefix=reasoning_prefix,
    )
    model_d = OpenAILLMModel(
        model_name=model_name,
        greedy_temperature=0.0,
        sample_temperature=0.0,
        max_tokens=max_tokens,
        timeout_seconds=timeout,
        prompt_prefix=direct_prefix,
    )

    results: dict[str, list[dict[str, Any]]] = {f"self_consistency_{n}": [] for n in ns}
    results["direct_plus_revise"] = []

    for qid, question, gold in queries:
        for n in ns:
            if mode == "math":
                res = run_self_consistency_reasoning_n_math(model_r, question, n)
            else:
                res = run_self_consistency_reasoning_n_numeric(model_r, question, n)
            pred = str(res.get("predicted_answer", ""))
            results[f"self_consistency_{n}"].append(
                {
                    "question_id": qid,
                    "correct": int(match_gold(gold, pred, mode)),
                    "samples_used": res.get("samples_used", n),
                }
            )
        dpr = run_direct_plus_revise(model_d, question)
        pred_d = str(dpr.get("predicted_answer", ""))
        if mode == "math":
            pred_d = normalize_math_answer(pred_d)
        results["direct_plus_revise"].append(
            {
                "question_id": qid,
                "correct": int(match_gold(gold, pred_d, mode)),
                "samples_used": dpr.get("samples_used", 2),
            }
        )

    summary: dict[str, Any] = {}
    m = len(queries)
    for key, lst in results.items():
        acc = sum(r["correct"] for r in lst) / m if m else 0.0
        samples = sum(r["samples_used"] for r in lst)
        summary[key] = {
            "accuracy": acc,
            "avg_cost_proxy": samples / m if m else 0.0,
            "n_queries": m,
        }
    return {"per_strategy_rows": results, "summary": summary}
