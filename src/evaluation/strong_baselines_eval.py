"""Strong baseline suite: compute ladder, threshold routers, and BEST-Route-style routing.

Designed for paper-grade comparisons on GSM8K, Hard GSM8K, and MATH500.
Cost proxy is **model forward calls** (samples_used), consistent with existing
strategy expansion code.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Protocol

from src.baselines.self_consistency import self_consistency_result_for_samples
from src.evaluation.strategy_expansion_eval import (
    run_direct_plus_revise,
    run_reasoning_greedy,
    run_reasoning_then_revise,
)
from src.features.constraint_violation_features import extract_constraint_violation_features
from src.features.precompute_features import extract_first_pass_features, extract_query_features
from src.utils.answer_extraction import extract_math_answer, extract_numeric_answer

# Cost proxy: one generate() = 1 unit; generate_n(k) = k units.
LADDER_METHOD_COST: dict[str, float] = {
    "reasoning_greedy": 1.0,
    "self_consistency_3": 3.0,
    "self_consistency_5": 5.0,
    "direct_plus_revise": 2.0,
    "reasoning_then_revise": 2.0,
}

COMPUTE_LADDER_METHODS = list(LADDER_METHOD_COST.keys())


class _ModelProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...
    def generate_n(self, prompt: str, n: int) -> list[str]: ...


def _normalize_gsm8k_gold(answer: str) -> str:
    from src.evaluation.strategy_expansion_eval import _normalize

    return _normalize(answer)


def _extract_pred(use_math: bool, text: str) -> str:
    if use_math:
        return extract_math_answer(text)
    return _normalize_gsm8k_gold(extract_numeric_answer(text))


def run_static_method(
    model: _ModelProtocol,
    method: str,
    question: str,
    *,
    use_math_extraction: bool,
) -> dict[str, Any]:
    """Run a single static baseline; return predicted, samples_used, extras."""
    if method == "reasoning_greedy":
        r = run_reasoning_greedy(model, question)
        raw = r["raw_outputs"][0]
        pred = extract_math_answer(raw) if use_math_extraction else r["predicted_answer"]
        return {
            "predicted_answer": pred,
            "samples_used": r["samples_used"],
            "self_consistency_ambiguous": False,
            "self_consistency_tie": False,
        }
    if method == "direct_plus_revise":
        r = run_direct_plus_revise(model, question)
        pred = r["predicted_answer"]
        if use_math_extraction:
            pred = extract_math_answer(r["raw_outputs"][-1])
        return {
            "predicted_answer": pred,
            "samples_used": r["samples_used"],
            "self_consistency_ambiguous": False,
            "self_consistency_tie": False,
        }
    if method == "reasoning_then_revise":
        r = run_reasoning_then_revise(model, question)
        pred = r["predicted_answer"]
        if use_math_extraction:
            pred = extract_math_answer(r["raw_outputs"][-1])
        return {
            "predicted_answer": pred,
            "samples_used": r["samples_used"],
            "self_consistency_ambiguous": False,
            "self_consistency_tie": False,
        }
    if method == "self_consistency_3":
        res = self_consistency_result_for_samples(
            model, "", question, "",
            3,
            use_math_extraction=use_math_extraction,
        )
        return {
            "predicted_answer": res.final_answer,
            "samples_used": res.samples_used,
            "self_consistency_ambiguous": res.self_consistency_ambiguous,
            "self_consistency_tie": res.self_consistency_tie,
        }
    if method == "self_consistency_5":
        res = self_consistency_result_for_samples(
            model, "", question, "",
            5,
            use_math_extraction=use_math_extraction,
        )
        return {
            "predicted_answer": res.final_answer,
            "samples_used": res.samples_used,
            "self_consistency_ambiguous": res.self_consistency_ambiguous,
            "self_consistency_tie": res.self_consistency_tie,
        }
    raise ValueError(f"Unknown static method: {method}")


def _confidence_from_first_reasoning(
    question: str, raw_output: str, parsed: str
) -> float:
    fp = extract_first_pass_features(question, raw_output, parsed_answer=parsed)
    cv = extract_constraint_violation_features(question, raw_output, predicted_answer=parsed)
    n_signals = int(cv.get("constraint_signal_count", 0))
    conf = 1.0 if fp["first_pass_parse_success"] else 0.25
    if fp["first_pass_has_uncertainty_phrase"]:
        conf *= 0.55
    conf *= max(0.15, 1.0 - 0.12 * n_signals)
    if fp["first_pass_empty_or_malformed_flag"]:
        conf *= 0.4
    return max(0.0, min(1.0, conf))


def _difficulty_score(question: str) -> float:
    z = extract_query_features(question)
    # Normalized heuristic in ~[0,1]
    char_n = min(1.0, z["question_length_chars"] / 1200.0)
    num_n = min(1.0, z["num_numeric_mentions"] / 12.0)
    sent_n = min(1.0, z["num_sentences_approx"] / 8.0)
    return max(0.0, min(1.0, 0.35 * char_n + 0.35 * num_n + 0.3 * sent_n))


_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _tail_numeric_disagreement(text: str, window: int = 400) -> bool:
    tail = text.strip()[-window:]
    nums = [_normalize_gsm8k_gold(m) for m in _NUMBER_RE.findall(tail)]
    nums = [n for n in nums if n]
    return len(set(nums)) >= 2


def _reasoning_final_mismatch(full_text: str, *, use_math: bool) -> bool:
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    if not lines:
        return False
    last_line = lines[-1]
    a_full = _extract_pred(use_math, full_text)
    a_tail = _extract_pred(use_math, last_line)
    if not a_full or not a_tail:
        return False
    return a_full != a_tail


def _output_router_escalate(
    question: str,
    reasoning_raw: str,
    parsed: str,
    *,
    use_math: bool,
) -> bool:
    fp = extract_first_pass_features(question, reasoning_raw, parsed_answer=parsed)
    cv = extract_constraint_violation_features(
        question, reasoning_raw, predicted_answer=parsed
    )
    low_conf = _confidence_from_first_reasoning(question, reasoning_raw, parsed) < 0.45
    mismatch = _reasoning_final_mismatch(reasoning_raw, use_math=use_math)
    violations = int(cv.get("constraint_signal_count", 0)) > 0
    tail_dis = _tail_numeric_disagreement(reasoning_raw)
    uncertain = bool(fp["first_pass_has_uncertainty_phrase"])
    bad_parse = not bool(fp["first_pass_parse_success"])
    return bool(
        low_conf or mismatch or violations or tail_dis or uncertain or bad_parse
    )


def evaluate_compute_ladder(
    model: _ModelProtocol,
    queries: list[Any],
    *,
    dataset_key: str,
    use_math_extraction: bool,
) -> dict[str, Any]:
    per_method: dict[str, dict[str, Any]] = {}
    for method in COMPUTE_LADDER_METHODS:
        correct = 0
        total_samples = 0
        ambiguous = 0
        ties = 0
        for q in queries:
            out = run_static_method(
                model, method, q.question, use_math_extraction=use_math_extraction
            )
            pred = out["predicted_answer"]
            if use_math_extraction:
                from src.utils.answer_extraction import normalize_math_answer

                pred_n = normalize_math_answer(pred)
                gold_n = q.answer
            else:
                pred_n = _normalize_gsm8k_gold(pred)
                gold_n = _normalize_gsm8k_gold(q.answer)
            ok = pred_n == gold_n
            correct += int(ok)
            su = int(out["samples_used"])
            total_samples += su
            ambiguous += int(out.get("self_consistency_ambiguous", False))
            ties += int(out.get("self_consistency_tie", False))
        n = len(queries)
        avg_cost = total_samples / n if n else 0.0
        base_ref = LADDER_METHOD_COST["reasoning_greedy"]
        per_method[method] = {
            "method": method,
            "accuracy": correct / n if n else 0.0,
            "avg_cost_proxy": avg_cost,
            "extra_compute_rate": max(0.0, avg_cost - base_ref),
            "self_consistency_ambiguous_count": ambiguous,
            "self_consistency_tie_count": ties,
        }
    return {
        "dataset": dataset_key,
        "use_math_extraction": use_math_extraction,
        "n_queries": len(queries),
        "methods": per_method,
    }


def evaluate_confidence_router_curve(
    model: _ModelProtocol,
    queries: list[Any],
    *,
    dataset_key: str,
    use_math_extraction: bool,
    strong_action: str,
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    if strong_action not in ("direct_plus_revise", "self_consistency_3"):
        raise ValueError("strong_action must be direct_plus_revise or self_consistency_3")
    if thresholds is None:
        thresholds = [round(x, 4) for x in _linspace(0.05, 0.95, 19)]
    rows: list[dict[str, Any]] = []
    for tau in thresholds:
        correct = 0
        total_cost = 0.0
        routed = 0
        for q in queries:
            r0 = run_reasoning_greedy(model, q.question)
            raw = r0["raw_outputs"][0]
            parsed = _extract_pred(use_math_extraction, raw)
            conf = _confidence_from_first_reasoning(q.question, raw, parsed)
            if conf < tau:
                routed += 1
                out = run_static_method(
                    model, strong_action, q.question, use_math_extraction=use_math_extraction
                )
                pred = out["predicted_answer"]
                cost = float(out["samples_used"])
            else:
                pred = (
                    extract_math_answer(raw)
                    if use_math_extraction
                    else r0["predicted_answer"]
                )
                cost = 1.0
            if use_math_extraction:
                from src.utils.answer_extraction import normalize_math_answer

                ok = normalize_math_answer(pred) == q.answer
            else:
                ok = _normalize_gsm8k_gold(pred) == _normalize_gsm8k_gold(q.answer)
            correct += int(ok)
            total_cost += cost
        n = len(queries)
        rows.append(
            {
                "dataset": dataset_key,
                "router": "confidence_threshold",
                "strong_action": strong_action,
                "threshold": tau,
                "accuracy": correct / n if n else 0.0,
                "avg_cost_proxy": total_cost / n if n else 0.0,
                "routing_rate": routed / n if n else 0.0,
            }
        )
    return rows


def evaluate_output_router(
    model: _ModelProtocol,
    queries: list[Any],
    *,
    dataset_key: str,
    use_math_extraction: bool,
    escalate_action: str,
) -> dict[str, Any]:
    if escalate_action not in ("reasoning_then_revise", "self_consistency_3"):
        raise ValueError("escalate_action must be reasoning_then_revise or self_consistency_3")
    correct = 0
    total_cost = 0.0
    routed = 0
    for q in queries:
        r0 = run_reasoning_greedy(model, q.question)
        raw = r0["raw_outputs"][0]
        parsed = _extract_pred(use_math_extraction, raw)
        escalate = _output_router_escalate(
            q.question, raw, parsed, use_math=use_math_extraction
        )
        if escalate:
            routed += 1
            out = run_static_method(
                model, escalate_action, q.question, use_math_extraction=use_math_extraction
            )
            pred = out["predicted_answer"]
            cost = float(out["samples_used"])
        else:
            pred = (
                extract_math_answer(raw)
                if use_math_extraction
                else r0["predicted_answer"]
            )
            cost = 1.0
        if use_math_extraction:
            from src.utils.answer_extraction import normalize_math_answer

            ok = normalize_math_answer(pred) == q.answer
        else:
            ok = _normalize_gsm8k_gold(pred) == _normalize_gsm8k_gold(q.answer)
        correct += int(ok)
        total_cost += cost
    n = len(queries)
    return {
        "dataset": dataset_key,
        "router": "output_aware",
        "escalate_action": escalate_action,
        "accuracy": correct / n if n else 0.0,
        "avg_cost_proxy": total_cost / n if n else 0.0,
        "routing_rate": routed / n if n else 0.0,
        "n_queries": n,
    }


def evaluate_best_route_style(
    model: _ModelProtocol,
    queries: list[Any],
    *,
    dataset_key: str,
    use_math_extraction: bool,
    difficulty_hi: float = 0.55,
    difficulty_lo: float = 0.35,
    conf_hi: float = 0.55,
    conf_lo: float = 0.35,
) -> dict[str, Any]:
    """Simplified BEST-Route-style router (not official code; see report)."""
    counts: dict[str, int] = {
        "reasoning_greedy": 0,
        "self_consistency_3": 0,
        "reasoning_then_revise": 0,
    }
    correct = 0
    total_cost = 0.0
    per_query: list[dict[str, Any]] = []
    for q in queries:
        r0 = run_reasoning_greedy(model, q.question)
        raw = r0["raw_outputs"][0]
        parsed = _extract_pred(use_math_extraction, raw)
        conf = _confidence_from_first_reasoning(q.question, raw, parsed)
        diff = _difficulty_score(q.question)
        score = diff + (1.0 - conf)
        # Higher score → harder / less confident → stronger action
        if score >= difficulty_hi + (1.0 - conf_lo):
            action = "reasoning_then_revise"
        elif score >= difficulty_lo + (1.0 - conf_hi):
            action = "self_consistency_3"
        else:
            action = "reasoning_greedy"
        counts[action] += 1
        out = run_static_method(
            model, action, q.question, use_math_extraction=use_math_extraction
        )
        pred = out["predicted_answer"]
        cost = float(out["samples_used"])
        if use_math_extraction:
            from src.utils.answer_extraction import normalize_math_answer

            ok = normalize_math_answer(pred) == q.answer
        else:
            ok = _normalize_gsm8k_gold(pred) == _normalize_gsm8k_gold(q.answer)
        correct += int(ok)
        total_cost += cost
        per_query.append(
            {
                "question_id": q.id,
                "action": action,
                "difficulty_proxy": diff,
                "confidence_proxy": conf,
                "score": score,
                "correct": ok,
                "samples_used": cost,
            }
        )
    n = len(queries)
    return {
        "dataset": dataset_key,
        "router": "best_route_style_simplified",
        "note": (
            "Official BEST-Route code is not integrated (see external/best_route). "
            "This is a threshold router on difficulty + confidence proxies."
        ),
        "params": {
            "difficulty_hi": difficulty_hi,
            "difficulty_lo": difficulty_lo,
            "conf_hi": conf_hi,
            "conf_lo": conf_lo,
        },
        "action_counts": counts,
        "accuracy": correct / n if n else 0.0,
        "avg_cost_proxy": total_cost / n if n else 0.0,
        "n_queries": n,
        "per_query": per_query,
    }


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n < 2:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def write_compute_ladder_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def append_confidence_router_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = [
        "dataset",
        "router",
        "strong_action",
        "threshold",
        "accuracy",
        "avg_cost_proxy",
        "routing_rate",
    ]
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_output_router_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_best_route_style_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = {k: v for k, v in payload.items() if k != "per_query"}
    path.write_text(json.dumps(to_write, indent=2))


def _r4(x: float) -> float:
    return round(float(x), 4)


def build_summary_rows(
    dataset_key: str,
    ladder: dict[str, Any],
    output_routers: list[dict[str, Any]],
    best_route: dict[str, Any],
    confidence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten results for final_baseline_summary.csv."""
    rows: list[dict[str, Any]] = []
    base_cost = LADDER_METHOD_COST["reasoning_greedy"]
    for method, stats in ladder.get("methods", {}).items():
        extra = float(stats.get("avg_cost_proxy", 0)) - base_cost
        note = ""
        if method.startswith("self_consistency"):
            note = (
                f"ambiguous={stats.get('self_consistency_ambiguous_count', 0)} "
                f"ties={stats.get('self_consistency_tie_count', 0)}"
            )
        rows.append(
            {
                "dataset": dataset_key,
                "method": method,
                "accuracy": _r4(stats.get("accuracy", 0.0)),
                "avg_cost": _r4(stats.get("avg_cost_proxy", 0.0)),
                "extra_compute_rate": _r4(max(0.0, extra)),
                "notes": note.strip(),
            }
        )
    for orow in output_routers:
        m = f"output_router_{orow.get('escalate_action', '')}"
        rows.append(
            {
                "dataset": dataset_key,
                "method": m,
                "accuracy": _r4(orow.get("accuracy", 0.0)),
                "avg_cost": _r4(orow.get("avg_cost_proxy", 0.0)),
                "extra_compute_rate": _r4(
                    max(0.0, float(orow.get("avg_cost_proxy", 0)) - base_cost)
                ),
                "notes": f"routing_rate={orow.get('routing_rate', 0):.4f}",
            }
        )
    rows.append(
        {
            "dataset": dataset_key,
            "method": "best_route_style",
            "accuracy": _r4(best_route.get("accuracy", 0.0)),
            "avg_cost": _r4(best_route.get("avg_cost_proxy", 0.0)),
            "extra_compute_rate": _r4(
                max(0.0, float(best_route.get("avg_cost_proxy", 0)) - base_cost)
            ),
            "notes": str(best_route.get("note", ""))[:200],
        }
    )
    for crow in confidence_rows:
        sa = crow.get("strong_action", "")
        th = crow.get("threshold", 0)
        rows.append(
            {
                "dataset": dataset_key,
                "method": f"conf_router_{sa}_t{th}",
                "accuracy": _r4(crow.get("accuracy", 0.0)),
                "avg_cost": _r4(crow.get("avg_cost_proxy", 0.0)),
                "extra_compute_rate": _r4(
                    max(0.0, float(crow.get("avg_cost_proxy", 0)) - base_cost)
                ),
                "notes": f"rout_rate={crow.get('routing_rate', 0):.4f}",
            }
        )
    return rows


def write_final_summary_csv(all_rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "method", "accuracy", "avg_cost", "extra_compute_rate", "notes"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_dataset_rollup_csv(
    all_rows: list[dict[str, Any]], path: str | Path
) -> None:
    """One row per dataset: best static ladder method by accuracy, then adaptive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_ds.setdefault(str(row["dataset"]), []).append(row)
    static_names = set(COMPUTE_LADDER_METHODS)
    out_rows: list[dict[str, Any]] = []
    for ds, rows in sorted(by_ds.items()):
        static_rows = [r for r in rows if r["method"] in static_names]
        best_static = max(static_rows, key=lambda r: float(r["accuracy"])) if static_rows else {}
        adaptive = [
            r
            for r in rows
            if str(r["method"]).startswith(
                ("output_router", "best_route_style", "conf_router")
            )
        ]
        best_adapt = max(adaptive, key=lambda r: float(r["accuracy"])) if adaptive else {}
        out_rows.append(
            {
                "dataset": ds,
                "best_static_method": best_static.get("method", ""),
                "best_static_accuracy": best_static.get("accuracy", ""),
                "best_static_avg_cost": best_static.get("avg_cost", ""),
                "best_adaptive_method": best_adapt.get("method", ""),
                "best_adaptive_accuracy": best_adapt.get("accuracy", ""),
                "best_adaptive_avg_cost": best_adapt.get("avg_cost", ""),
            }
        )
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "dataset",
        "best_static_method",
        "best_static_accuracy",
        "best_static_avg_cost",
        "best_adaptive_method",
        "best_adaptive_accuracy",
        "best_adaptive_avg_cost",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
