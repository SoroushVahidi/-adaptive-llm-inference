#!/usr/bin/env python3
"""Run offline signal-ablation routing experiment from enriched datasets.

No new model calls. Uses existing per-query outcomes + precomputed signal columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "signal_ablation"
PAPER_TABLES = ROOT / "outputs" / "paper_tables"
PAPER_FIGS = ROOT / "outputs" / "paper_figures"

REGIME_FILES = {
    "gsm8k_random_100": DATA / "real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": DATA / "real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": DATA / "real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": DATA / "real_math500_routing_dataset_enriched.csv",
}

# Grounded from v6 defaults in src/policies/adaptive_policy_v6.py
THRESH = {
    "answer_error": 2,
    "explanation_warning": 3,
    "combined_equal": 3,
    "answer_error_dominant": 4,
    "explanation_dominant": 4,
}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _variant_flags(df: pd.DataFrame) -> dict[str, pd.Series]:
    a = _to_num(df["v6_answer_error_score"])
    e = _to_num(df["v6_explanation_warning_score"])

    return {
        "explanation_only_router": e >= THRESH["explanation_warning"],
        "answer_error_only_router": a >= THRESH["answer_error"],
        "combined_equal_router": (a + e) >= THRESH["combined_equal"],
        "answer_error_dominant_router": (2 * a + e) >= THRESH["answer_error_dominant"],
        "explanation_dominant_router": (a + 2 * e) >= THRESH["explanation_dominant"],
    }


def _eval_policy(df: pd.DataFrame, revise_flag: pd.Series) -> dict[str, float]:
    rg = _to_num(df["reasoning_correct"]).astype(int)
    dpr = _to_num(df["revise_correct"]).astype(int)
    helpful = _to_num(df["revise_helpful"]).astype(int)

    chosen_correct = np.where(revise_flag, dpr, rg)
    revise_rate = float(np.mean(revise_flag))
    accuracy = float(np.mean(chosen_correct))
    avg_cost = 1.0 + revise_rate

    # FP: revised when RG already correct and revise not needed/no improvement
    false_positive = revise_flag & (rg == 1) & (helpful == 0)
    # FN: did not revise on revise-helpful query
    false_negative = (~revise_flag) & (helpful == 1)

    return {
        "accuracy": accuracy,
        "average_cost": avg_cost,
        "revise_rate": revise_rate,
        "false_positive_count": int(false_positive.sum()),
        "false_negative_count": int(false_negative.sum()),
    }


def _bootstrap_ci(correct: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(correct)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = correct[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    if np.allclose(means, 0.0):
        p_two = 1.0
    else:
        p_one = float(np.mean(means <= 0))
        p_two = float(min(1.0, 2 * min(p_one, 1 - p_one)))
    return float(lo), float(hi), p_two


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)

    policy_rows: list[dict] = []
    per_query_rows: list[dict] = []
    regime_rows: list[dict] = []
    boot_rows: list[dict] = []
    paired_rows: list[dict] = []

    for regime, path in REGIME_FILES.items():
        df = pd.read_csv(path)
        n = len(df)

        required = [
            "reasoning_correct",
            "revise_correct",
            "revise_helpful",
            "v6_answer_error_score",
            "v6_explanation_warning_score",
        ]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"Missing columns in {path}: {miss}")

        rg = _to_num(df["reasoning_correct"]).astype(int)
        dpr = _to_num(df["revise_correct"]).astype(int)
        helpful = _to_num(df["revise_helpful"]).astype(int)
        both_wrong = ((rg == 0) & (dpr == 0)).astype(int)
        oracle = np.maximum(rg, dpr)

        baseline_stats = {
            "reasoning_greedy": {
                "accuracy": float(rg.mean()),
                "average_cost": 1.0,
                "revise_rate": 0.0,
                "false_positive_count": 0,
                "false_negative_count": int(helpful.sum()),
            },
            "direct_plus_revise": {
                "accuracy": float(dpr.mean()),
                "average_cost": 2.0,
                "revise_rate": 1.0,
                "false_positive_count": int(((rg == 1) & (helpful == 0)).sum()),
                "false_negative_count": 0,
            },
        }

        variant_flags = _variant_flags(df)
        ablation_correct_vectors: dict[str, np.ndarray] = {}

        for pol, stats in baseline_stats.items():
            policy_rows.append(
                {
                    "regime": regime,
                    "policy": pol,
                    **stats,
                    "gain_vs_reasoning_greedy": stats["accuracy"] - float(rg.mean()),
                    "oracle_gap": float(oracle.mean()) - stats["accuracy"],
                    "n": n,
                }
            )

        for pol, flag in variant_flags.items():
            stats = _eval_policy(df, flag)
            corr = np.where(flag, dpr, rg).astype(int)
            ablation_correct_vectors[pol] = corr

            policy_rows.append(
                {
                    "regime": regime,
                    "policy": pol,
                    **stats,
                    "gain_vs_reasoning_greedy": stats["accuracy"] - float(rg.mean()),
                    "oracle_gap": float(oracle.mean()) - stats["accuracy"],
                    "n": n,
                }
            )

            for i, (_, row) in enumerate(df.iterrows()):
                rv = bool(flag.iloc[i])
                per_query_rows.append(
                    {
                        "regime": regime,
                        "question_id": row.get("question_id", i),
                        "policy": pol,
                        "chosen_route": "direct_plus_revise" if rv else "reasoning_greedy",
                        "reasoning_correct": int(rg.iloc[i]),
                        "revise_correct": int(dpr.iloc[i]),
                        "revise_helpful": int(helpful.iloc[i]),
                        "correct": int(corr[i]),
                        "cost": 2.0 if rv else 1.0,
                        "false_positive": int(rv and (int(rg.iloc[i]) == 1) and (int(helpful.iloc[i]) == 0)),
                        "false_negative": int((not rv) and (int(helpful.iloc[i]) == 1)),
                        "v6_answer_error_score": float(df.loc[row.name, "v6_answer_error_score"]),
                        "v6_explanation_warning_score": float(df.loc[row.name, "v6_explanation_warning_score"]),
                    }
                )

            lo, hi = _bootstrap_ci(corr)
            boot_rows.append(
                {
                    "regime": regime,
                    "policy": pol,
                    "accuracy": float(corr.mean()),
                    "ci_lower_95": lo,
                    "ci_upper_95": hi,
                    "n": n,
                }
            )

        exp_focus = ["explanation_only_router", "explanation_dominant_router"]
        ans_focus = ["answer_error_only_router", "answer_error_dominant_router"]
        ablation_df = pd.DataFrame([r for r in policy_rows if r["regime"] == regime and r["policy"] in variant_flags])

        best_exp = ablation_df[ablation_df["policy"].isin(exp_focus)].sort_values(
            ["accuracy", "average_cost"], ascending=[False, True]
        ).iloc[0]
        best_ans = ablation_df[ablation_df["policy"].isin(ans_focus)].sort_values(
            ["accuracy", "average_cost"], ascending=[False, True]
        ).iloc[0]
        best_all = ablation_df.sort_values(["accuracy", "average_cost"], ascending=[False, True]).iloc[0]

        lo, hi, p = _paired_bootstrap_diff(
            ablation_correct_vectors[str(best_ans["policy"])],
            ablation_correct_vectors[str(best_exp["policy"])],
        )
        paired_rows.append(
            {
                "regime": regime,
                "comparison": f"{best_ans['policy']} minus {best_exp['policy']}",
                "mean_diff": float(best_ans["accuracy"] - best_exp["accuracy"]),
                "ci_lower_95": lo,
                "ci_upper_95": hi,
                "p_value_two_sided": p,
                "n": n,
                "method": "paired-bootstrap",
            }
        )

        regime_rows.append(
            {
                "regime": regime,
                "n": n,
                "revise_helpful_prevalence": float(helpful.mean()),
                "both_wrong_prevalence": float(both_wrong.mean()),
                "best_explanation_policy": str(best_exp["policy"]),
                "best_explanation_accuracy": float(best_exp["accuracy"]),
                "best_explanation_cost": float(best_exp["average_cost"]),
                "best_answer_error_policy": str(best_ans["policy"]),
                "best_answer_error_accuracy": float(best_ans["accuracy"]),
                "best_answer_error_cost": float(best_ans["average_cost"]),
                "best_overall_ablation_policy": str(best_all["policy"]),
                "best_overall_ablation_accuracy": float(best_all["accuracy"]),
                "best_overall_ablation_cost": float(best_all["average_cost"]),
                "answer_error_beats_explanation_accuracy": bool(best_ans["accuracy"] > best_exp["accuracy"]),
                "answer_error_ties_explanation_accuracy": bool(best_ans["accuracy"] == best_exp["accuracy"]),
                "answer_error_cost_advantage_when_tied": bool(
                    best_ans["accuracy"] == best_exp["accuracy"] and best_ans["average_cost"] < best_exp["average_cost"]
                ),
                "accuracy_diff_answer_minus_explanation": float(best_ans["accuracy"] - best_exp["accuracy"]),
            }
        )

    policy_df = pd.DataFrame(policy_rows).sort_values(["regime", "policy"])
    per_query_df = pd.DataFrame(per_query_rows).sort_values(["regime", "policy", "question_id"])
    regime_df = pd.DataFrame(regime_rows).sort_values("regime")
    boot_df = pd.DataFrame(boot_rows).sort_values(["regime", "policy"])
    paired_df = pd.DataFrame(paired_rows).sort_values("regime")

    policy_df.to_csv(OUT / "policy_comparison.csv", index=False)
    per_query_df.to_csv(OUT / "per_query_decisions.csv", index=False)
    regime_df.to_csv(OUT / "regime_summary.csv", index=False)
    boot_df.to_csv(OUT / "bootstrap_ci.csv", index=False)
    paired_df.to_csv(OUT / "paired_tests.csv", index=False)

    main_rows = []
    for regime in regime_df["regime"]:
        pr = policy_df[policy_df["regime"] == regime]
        rr = regime_df[regime_df["regime"] == regime].iloc[0]

        chosen = {
            "reasoning_greedy": pr[pr["policy"] == "reasoning_greedy"].iloc[0],
            "direct_plus_revise": pr[pr["policy"] == "direct_plus_revise"].iloc[0],
            "best_explanation": pr[pr["policy"] == rr["best_explanation_policy"]].iloc[0],
            "best_answer_error": pr[pr["policy"] == rr["best_answer_error_policy"]].iloc[0],
            "best_overall_ablation": pr[pr["policy"] == rr["best_overall_ablation_policy"]].iloc[0],
        }
        for label, row in chosen.items():
            main_rows.append(
                {
                    "regime": regime,
                    "entry": label,
                    "policy": row["policy"],
                    "accuracy": row["accuracy"],
                    "average_cost": row["average_cost"],
                    "revise_rate": row["revise_rate"],
                    "false_positive_count": int(row["false_positive_count"]),
                    "false_negative_count": int(row["false_negative_count"]),
                    "note": (
                        "answer-focused better than explanation-focused"
                        if label == "best_answer_error" and rr["accuracy_diff_answer_minus_explanation"] > 0
                        else "answer/explanation tie; compare cost"
                        if label == "best_answer_error" and rr["accuracy_diff_answer_minus_explanation"] == 0
                        else ""
                    ),
                }
            )

    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(PAPER_TABLES / "signal_ablation_main_table.csv", index=False)

    # Figure: best explanation-focused vs best answer-focused in acc-cost plane.
    figure_generated = False
    figure_path = PAPER_FIGS / "signal_ablation_summary.png"
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for _, r in regime_df.iterrows():
            ex = policy_df[(policy_df["regime"] == r["regime"]) & (policy_df["policy"] == r["best_explanation_policy"])].iloc[0]
            an = policy_df[(policy_df["regime"] == r["regime"]) & (policy_df["policy"] == r["best_answer_error_policy"])].iloc[0]
            ax.scatter(ex["average_cost"], ex["accuracy"], marker="o", s=70, label=f"{r['regime']} explanation")
            ax.scatter(an["average_cost"], an["accuracy"], marker="^", s=70, label=f"{r['regime']} answer")
            ax.plot([ex["average_cost"], an["average_cost"]], [ex["accuracy"], an["accuracy"]], linestyle="--", alpha=0.5)

        ax.set_xlabel("Average cost")
        ax.set_ylabel("Accuracy")
        ax.set_title("Signal ablation: best explanation-focused vs answer-error-focused")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)
        figure_generated = True
    hard = regime_df[regime_df["regime"].isin(["hard_gsm8k_100", "hard_gsm8k_b2"])]
    easy = regime_df[regime_df["regime"].isin(["gsm8k_random_100", "math500_100"])]

    summary = {
        "evidence_status": "measured_now",
        "thresholds": THRESH,
        "regimes": regime_df.to_dict(orient="records"),
        "overall": {
            "answer_error_better_count": int(regime_df["answer_error_beats_explanation_accuracy"].sum()),
            "answer_error_tie_count": int(regime_df["answer_error_ties_explanation_accuracy"].sum()),
            "mean_accuracy_diff_answer_minus_explanation": float(regime_df["accuracy_diff_answer_minus_explanation"].mean()),
            "hard_regimes_mean_diff": float(hard["accuracy_diff_answer_minus_explanation"].mean()),
            "easy_regimes_mean_diff": float(easy["accuracy_diff_answer_minus_explanation"].mean()),
        },
        "artifacts": {
            "policy_comparison": str(OUT / "policy_comparison.csv"),
            "per_query_decisions": str(OUT / "per_query_decisions.csv"),
            "regime_summary": str(OUT / "regime_summary.csv"),
            "bootstrap_ci": str(OUT / "bootstrap_ci.csv"),
            "paired_tests": str(OUT / "paired_tests.csv"),
            "paper_table": str(PAPER_TABLES / "signal_ablation_main_table.csv"),
            "figure": str(figure_path),
            "figure_generated": figure_generated,
        },
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote signal ablation artifacts to", OUT)


if __name__ == "__main__":
    main()
