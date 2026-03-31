# Manuscript Artifacts

> OUTDATED / SUPERSEDED (2026-03-31): Use `docs/CANONICAL_MANUSCRIPT_DECISIONS.md` and
> `outputs/paper_tables_final/` + `outputs/paper_figures_final/` as the authoritative manuscript package.
> This file is retained as historical inventory.

This document inventories all committed experiment outputs that support the
manuscript.  Files are marked as **Final** (suitable for citation) or
**Partial** (preliminary; should not be cited without additional verification).

---

## Main Tables

### Table 1 — Cross-Regime Routing Summary (Main Results Table)

| Status | Path |
|--------|------|
| **Final** | `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv` |
| Copy | `outputs/cross_regime_comparison/final_cross_regime_summary.csv` |

**Columns:** `dataset`, `reasoning_accuracy`, `revise_accuracy`,
`oracle_accuracy`, `best_policy_accuracy`, `best_policy_cost`,
`best_policy_name`, `revise_helpful_rate`, `reasoning_then_revise_accuracy`

**Key values (gpt-4o-mini):**

| Dataset | Reasoning | Best Policy | Oracle | Cost |
|---------|-----------|-------------|--------|------|
| GSM8K-100 | 90.0 % | 92.0 % (v6) | 92.0 % | 1.18× |
| Hard-GSM8K-100 | 79.0 % | 82.0 % (v7) | 91.0 % | 1.46× |
| MATH500-100 | 64.0 % | 65.0 % (v6) | 70.0 % | 1.03× |
| AIME-2024 | 13.3 % | n/a | 13.3 % | — |

> **Caveat:** Policy gains are modest on easy regimes (GSM8K, MATH500) and the
> gap to oracle remains large on harder regimes.  These numbers reflect
> `gpt-4o-mini` only and may not generalise.

---

### Table 2 — Baseline Strategy Comparison

| Status | Path |
|--------|------|
| **Final** | `outputs/paper_tables/baselines/baselines_gsm8k_strategies.csv` |
| **Final** | `outputs/paper_tables/baselines/baselines_hard_gsm8k_strategies.csv` |
| **Final** | `outputs/paper_tables/baselines/baselines_math500_strategies.csv` |

Strategies covered: `reasoning_greedy`, `self_consistency_3/5`,
`direct_plus_revise`, `reasoning_then_revise`.

---

### Table 3 — Oracle Routing Upper Bound

| Status | Path |
|--------|------|
| **Final** | `outputs/paper_tables/oracle_routing/oracle_routing_eval_summaries.csv` |
| Supporting | `outputs/oracle_routing_eval/` (per-dataset JSON summaries) |

---

### Table 4 — Next-Stage Budget Curves

| Status | Path |
|--------|------|
| **Final** | `outputs/paper_tables/next_stage/next_stage_budget_curves_all_datasets.csv` |
| Supporting | `outputs/next_stage_eval/` (per-dataset `budget_curve.csv`, `cascade_curve.csv`) |

---

### Table 5 — Policy Routing Comparison (Long Format)

| Status | Path |
|--------|------|
| **Final** | `outputs/paper_tables/real_routing/real_policy_eval_comparison_long.csv` |

---

## Supporting / Appendix Artifacts

### Budget Sweep Curves

| Status | Path |
|--------|------|
| **Final** | `outputs/budget_sweep/gsm8k_random100_budget_curve.csv` |
| **Final** | `outputs/budget_sweep/hard_gsm8k_100_budget_curve.csv` |
| **Final** | `outputs/budget_sweep/hard_gsm8k_b2_budget_curve.csv` |
| **Final** | `outputs/budget_sweep/math500_100_budget_curve.csv` |

### Adaptive Policy V7 Analysis

| Status | Path |
|--------|------|
| **Final** | `outputs/adaptive_policy_v7/summary.json` |
| **Final** | `outputs/adaptive_policy_v7/per_case_results.csv` |
| **Final** | `outputs/adaptive_policy_v7/signal_summary.csv` |
| Diagnostic | `outputs/adaptive_policy_v7/false_negative_probe.csv` |
| Diagnostic | `outputs/adaptive_policy_v7/false_positive_recheck.csv` |

### Routing Model (Learned)

| Status | Path |
|--------|------|
| **Final** | `outputs/real_math500_routing_model/model_metrics.csv` |
| **Final** | `outputs/real_math500_routing_model/feature_importance.csv` |
| **Final** | `outputs/real_math500_routing_model/summary.json` |

### Baseline Summaries (JSON)

| Status | Path |
|--------|------|
| **Final** | `outputs/baselines/gsm8k_baseline_summary.json` |
| **Final** | `outputs/baselines/hard_gsm8k_baseline_summary.json` |
| **Final** | `outputs/baselines/math500_baseline_summary.json` |

### Policy Evaluation Summaries

| Status | Path |
|--------|------|
| **Final** | `outputs/real_math500_policy_eval/summary.json` |
| **Final** | `outputs/real_hard_gsm8k_policy_eval/summary.json` |
| **Final** | `outputs/real_hard_gsm8k_b2_policy_eval/summary.json` |

### Hard-Regime Dataset Selection

| Status | Path |
|--------|------|
| **Final** | `outputs/hard_regime_selection/selection_summary.json` |
| **Final** | `outputs/hard_regime_selection_b2/selection_summary.json` |

---

## Raw API Response Files

The following files contain raw LLM outputs and are committed for
reproducibility traceability.  They are **not intended to be cited** directly.

| Path | Queries | Note |
|------|---------|------|
| `outputs/real_routing_dataset/raw_responses.jsonl` | 100 | GSM8K routing build |
| `outputs/real_aime2024_routing/raw_responses.jsonl` | 30 | AIME-2024 routing build |

---

## BLOCKED / Partial Artifacts

The following artifacts are referenced in the manuscript but cannot be
regenerated without an API key (marked **Needs API**) or require an additional
offline run (marked **Needs run**).

| Artifact | Status | Action required |
|----------|--------|-----------------|
| Simulated allocation sweep tables | **Needs run** | `python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml` |
| Oracle subset table | **Needs run** | `python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml` |
| Hard-GSM8K routing raw responses | **Needs API** | `python3 scripts/run_build_hard_gsm8k_routing_dataset.py` |

See `outputs/paper_tables/export_manifest.json` for the machine-readable list.

---

## Export Manifest

`outputs/paper_tables/export_manifest.json` — auto-generated by the paper
table export scripts; lists every written file and every blocker with its
regeneration command.
