# Ablation Blockers

This document records ablations that cannot be produced from current repository artifacts,
along with the minimal data or code needed to unblock them.

---

## Available Ablations (Completed)

The following ablations **were** successfully produced:

| Ablation | Output File |
|---|---|
| Confidence threshold sweep | `outputs/paper_tables/threshold_sweep_summary.csv` |
| Cost ratio sensitivity (1:1.5, 1:2, 1:3) | `outputs/paper_tables/cost_ratio_sensitivity.csv` |
| Policy ranking stability across cost ratios | `outputs/paper_tables/policy_ranking_stability.csv` |

---

## Blocked Ablations

### 1. Threshold Sweep for Adaptive Policy Signal Weights

**What it would show:** How policy v5/v6/v7 accuracy changes as internal signal thresholds are varied (e.g., `unified_error_revise_threshold` in v5).

**Why blocked:**
- The enriched dataset CSVs contain pre-computed scores (`unified_error_score`, `v6_answer_error_score`, etc.) but the full evaluation loop requires calling the policy code with the new threshold against every query.
- The policy code calls `src/policies/adaptive_policy_v5.py` which reads from enriched CSVs — this could in principle be rerun offline, but it requires the enriched datasets to be in their current location and the policy logic to be re-executed.
- This is a lightweight rerun (no LLM calls needed), but it is a code execution step, not just a table derivation from stored summaries.

**Minimal code/data needed:**
```python
# Pseudo-code: load enriched CSV, re-run adaptive_policy_v5 with different thresholds
import pandas as pd
from src.policies.adaptive_policy_v5 import AdaptivePolicyV5Config, run_policy

for threshold in [0.25, 0.30, 0.34, 0.40, 0.50]:
    cfg = AdaptivePolicyV5Config(unified_error_revise_threshold=threshold)
    results = run_policy(enriched_df, cfg)
    # record accuracy, avg_cost
```
**Missing:** A `run_policy` wrapper that accepts a config and an enriched DataFrame. Does not exist as a standalone function currently.

---

### 2. AIME 2024 Oracle Gap and Adaptive Policy Comparison

**What it would show:** Oracle accuracy, best adaptive accuracy, and oracle gap for the AIME 2024 regime.

**Why blocked:**
- `outputs/real_aime2024_routing/per_query_outputs.csv` exists with `reasoning_correct` and `revise_correct`.
- However, there is no `outputs/oracle_routing_eval/aime2024_oracle_summary.json`.
- More critically, there is no `outputs/real_aime2024_policy_eval/per_query_policy_decisions.csv` — the adaptive policies (v5/v6/v7) were not evaluated on AIME.

**Minimal code/data needed:**
- Run `scripts/run_adaptive_policy_v5_eval.py` (or v6/v7) with AIME dataset config.
- Requires: `data/real_aime2024_routing_dataset.csv` (exists) and an enriched version with pre-computed features.
- The enriched AIME dataset does not currently exist (`data/real_aime2024_routing_dataset_enriched.csv` is absent).

---

### 3. GPQA Diamond Policy Comparison

**What it would show:** Routing outcomes and adaptive policy performance on GPQA Diamond (multiple-choice science questions).

**Why blocked:**
- `outputs/real_gpqa_routing_dataset/gpqa_per_query_outputs.csv` exists with raw correctness data.
- No enriched GPQA dataset exists (`data/real_gpqa_diamond_routing_dataset_enriched.csv` is absent).
- No policy evaluation run has been completed for GPQA.

**Minimal code/data needed:**
- Run `scripts/run_build_real_routing_dataset.py` with GPQA config to produce enriched features.
- Then run adaptive policy eval.
- Requires no new LLM inference (uses stored `gpqa_per_query_outputs.csv`), but does require feature computation.

---

### 4. Per-Query CI for Confidence Threshold and Learned Router

**What it would show:** Bootstrap 95% CIs for the confidence-threshold and learned-router baselines at their operating points.

**Why blocked:**
- `outputs/baselines/confidence_threshold/confidence_threshold_sweep.csv` and
  `outputs/baselines/learned_router/learned_router_summary.csv` contain only aggregated metrics.
- No per-query decision vectors are stored for these baselines.

**Minimal code/data needed:**
- Re-run `scripts/run_confidence_threshold_baseline.py` with a flag to save per-query decisions.
- Re-run `scripts/run_learned_router_baseline.py` with a flag to save per-query predictions.
- Both use only existing enriched CSVs — no LLM inference required.

---

### 5. Budget Curve Uncertainty Bands

**What it would show:** 95% CI bands around the budget curves in `outputs/budget_sweep/`.

**Why blocked:**
- Budget curve CSVs contain aggregated (accuracy, avg_cost) at each budget fraction.
- No per-query routing decisions are stored for the budget sweep.

**Minimal code/data needed:**
- Re-run the budget sweep evaluation and save per-query traces alongside the sweep.
