"""Multi-action oracle dataset: labels, utilities, and CSV assembly."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.oracle_subset_eval import STRATEGY_COST_PROXY
from src.features.precompute_features import extract_first_pass_features, extract_query_features

MULTI_ACTION_ORDER: tuple[str, ...] = (
    "reasoning_greedy",
    "direct_plus_revise",
    "reasoning_then_revise",
    "self_consistency_3",
)

LAMBDA_VALUES: tuple[float, ...] = (0.0, 0.10, 0.25)


def _action_sort_key(action: str) -> tuple[int, int]:
    try:
        return (MULTI_ACTION_ORDER.index(action), 0)
    except ValueError:
        return (999, hash(action) % 10_000)


def best_accuracy_action(
    correctness: dict[str, int],
    costs: dict[str, int],
) -> str:
    """Maximize correctness; tie-break lower cost, then MULTI_ACTION_ORDER."""
    actions = list(correctness.keys())
    best_corr = max(correctness[a] for a in actions)
    candidates = [a for a in actions if correctness[a] == best_corr]
    return min(candidates, key=lambda a: (costs.get(a, 99), _action_sort_key(a)))


def best_utility_action(
    correctness: dict[str, int],
    costs: dict[str, int],
    lam: float,
) -> str:
    """Maximize correctness - lambda * cost; same tie-breaks as accuracy."""
    actions = list(correctness.keys())

    def utility(a: str) -> float:
        return float(correctness[a]) - lam * float(costs.get(a, 0))

    best_u = max(utility(a) for a in actions)
    candidates = [a for a in actions if abs(utility(a) - best_u) < 1e-12]
    return min(candidates, key=lambda a: (costs.get(a, 99), _action_sort_key(a)))


def build_multi_action_rows(
    queries: list[Any],
    per_query_rows: list[dict[str, Any]],
    strategies: list[str],
    dataset_name: str,
) -> list[dict[str, Any]]:
    """One flat row per query with features, per-action outcomes, and oracle labels."""
    rows_by_q: dict[str, dict[str, dict[str, Any]]] = {}
    for row in per_query_rows:
        qid = row["question_id"]
        strat = row["strategy"]
        if qid not in rows_by_q:
            rows_by_q[qid] = {}
        rows_by_q[qid][strat] = row

    query_by_id: dict[str, Any] = {}
    for q in queries:
        qid = getattr(q, "id", None) or q.get("id") or q.get("question_id")
        query_by_id[str(qid)] = q

    out: list[dict[str, Any]] = []
    for qid in sorted(rows_by_q.keys(), key=lambda x: x):
        strat_rows = rows_by_q[qid]
        qobj = query_by_id.get(qid)
        if qobj is None:
            continue
        qtext = getattr(qobj, "question", None) or qobj.get("question", "")
        gold = getattr(qobj, "answer", None) or qobj.get("answer", "")

        rg = strat_rows.get("reasoning_greedy", {})
        rg_raw = ""
        if rg.get("raw_outputs"):
            rg_raw = str(rg["raw_outputs"][0])
        fp_feats = extract_first_pass_features(
            qtext,
            rg_raw,
            parsed_answer=str(rg.get("predicted_answer", "") or ""),
        )
        q_feats = extract_query_features(qtext)

        row: dict[str, Any] = {
            "query_id": qid,
            "dataset": dataset_name,
            "gold_answer": gold,
            **{f"qf__{k}": v for k, v in q_feats.items()},
            **{f"fp__{k}": v for k, v in fp_feats.items()},
            "reasoning_greedy_raw_output": rg_raw[:8000] if rg_raw else "",
        }

        correctness: dict[str, int] = {}
        costs: dict[str, int] = {}
        for s in strategies:
            sr = strat_rows.get(s, {})
            correctness[s] = int(sr.get("correct", 0))
            c = int(sr.get("cost_proxy", sr.get("samples_used", STRATEGY_COST_PROXY.get(s, 0))))
            costs[s] = c
            row[f"{s}__correct"] = correctness[s]
            row[f"{s}__cost"] = c
            row[f"{s}__predicted_answer"] = sr.get("predicted_answer", "")
            if s == "self_consistency_3":
                row[f"{s}__ambiguous"] = int(bool(sr.get("self_consistency_ambiguous")))
                row[f"{s}__tied_values"] = sr.get("self_consistency_tied_values", "")

        row["best_accuracy_action"] = best_accuracy_action(correctness, costs)
        for lam in LAMBDA_VALUES:
            suffix = f"{lam:.2f}".replace(".", "_")
            row[f"best_utility_action_lambda_{suffix}"] = best_utility_action(
                correctness, costs, lam
            )

        sc3 = strat_rows.get("self_consistency_3", {})
        row["self_consistency_ambiguous_query_count"] = int(
            bool(sc3.get("self_consistency_ambiguous"))
        )
        out.append(row)

    return out


def write_multi_action_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def compute_oracle_summary_json(
    rows: list[dict[str, Any]],
    strategies: list[str],
    model: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Summary for ``outputs/multi_action_oracle/<dataset>_oracle_summary.json``."""
    n = len(rows)
    win_acc = Counter(r["best_accuracy_action"] for r in rows)
    util_wins: dict[str, Counter[str]] = {}
    for lam in LAMBDA_VALUES:
        suffix = f"{lam:.2f}".replace(".", "_")
        util_wins[suffix] = Counter(
            r[f"best_utility_action_lambda_{suffix}"] for r in rows
        )

    tie_acc = sum(1 for r in rows if _accuracy_label_tie(r, strategies))
    tie_util: dict[str, int] = {}
    for lam in LAMBDA_VALUES:
        suffix = f"{lam:.2f}".replace(".", "_")
        tie_util[suffix] = sum(1 for r in rows if _utility_label_tie(r, strategies, lam))

    action_accuracy: dict[str, float] = {}
    action_avg_cost: dict[str, float] = {}
    for s in strategies:
        if n:
            action_accuracy[s] = sum(int(r[f"{s}__correct"]) for r in rows) / n
            action_avg_cost[s] = sum(int(r[f"{s}__cost"]) for r in rows) / n
        else:
            action_accuracy[s] = 0.0
            action_avg_cost[s] = 0.0

    oracle_utility_mean: dict[str, float] = {}
    for lam in LAMBDA_VALUES:
        suffix = f"{lam:.2f}".replace(".", "_")
        if n:
            vals = []
            for r in rows:
                chosen = r[f"best_utility_action_lambda_{suffix}"]
                c = int(r[f"{chosen}__correct"])
                cost = int(r[f"{chosen}__cost"])
                vals.append(float(c) - lam * float(cost))
            oracle_utility_mean[suffix] = sum(vals) / n
        else:
            oracle_utility_mean[suffix] = 0.0

    amb_queries = sum(int(r.get("self_consistency_ambiguous_query_count", 0)) for r in rows)

    return {
        "dataset": dataset_name,
        "model": model,
        "num_queries": n,
        "strategies": strategies,
        "action_win_counts_best_accuracy": dict(win_acc),
        "action_accuracy": action_accuracy,
        "average_cost_per_action": action_avg_cost,
        "oracle_utility_mean_by_lambda": oracle_utility_mean,
        "tie_counts_best_accuracy": tie_acc,
        "tie_counts_utility_by_lambda": tie_util,
        "self_consistency_ambiguous_queries_total": amb_queries,
    }


def _accuracy_label_tie(row: dict[str, Any], strategies: list[str]) -> bool:
    """True when multiple actions share the maximum correctness score."""
    corr = {s: int(row[f"{s}__correct"]) for s in strategies}
    m = max(corr.values())
    return sum(1 for s in strategies if corr[s] == m) > 1


def _utility_label_tie(row: dict[str, Any], strategies: list[str], lam: float) -> bool:
    """True when multiple actions share the maximum utility (pre tie-break)."""
    costs = {s: int(row[f"{s}__cost"]) for s in strategies}
    corr = {s: int(row[f"{s}__correct"]) for s in strategies}

    def u(s: str) -> float:
        return float(corr[s]) - lam * float(costs[s])

    best_u = max(u(s) for s in strategies)
    return sum(1 for s in strategies if abs(u(s) - best_u) < 1e-12) > 1


def write_oracle_summary(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
