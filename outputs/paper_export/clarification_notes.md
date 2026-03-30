# Clarification Notes: Policy Table vs Budget-Curve Analysis

## Purpose

These notes accompany `clarification_table.csv` and are intended to eliminate reviewer confusion
between two distinct but related result types that appear in the manuscript.

---

## Three Distinct Objects in the Results

### 1. Best Practical Adaptive Policy (Table 5-style)
**What it is:** A deployable routing policy that observes a single per-instance feature vector
(e.g., confidence score, token count, problem-type signals) and makes a binary route decision
— "revise now" or "do not revise" — for each query at inference time.

**How it is selected:** Among the trained policy variants (v5, v6, v7), we report the one with
the best accuracy on the 100-query evaluation slice, together with its average cost and revise
rate.

**Key property:** *Deployable in practice.* The routing decision uses only features observable
before the answer is generated; it incurs no oracle knowledge about correctness.

**Where the numbers come from:**
- `outputs/paper_tables_cleaned/policy_eval_comparison.csv`
- `outputs/paper_tables_cleaned/final_cross_regime_summary_fixed.csv`
- Upstream: `outputs/real_*_policy_eval/summary.json`

---

### 2. Oracle Routing Upper Bound
**What it is:** A theoretical ceiling computed by routing each query to "revise" if and only if
revision *actually improves* correctness (known only in hindsight from gold labels).

**How it is computed:** Post-hoc, using the recorded ground-truth outcomes stored in the
evaluation datasets. It answers: "What is the best possible accuracy achievable by perfect binary
routing at the minimum necessary cost?"

**Key property:** *Not deployable.* Oracle routing requires knowing whether revision helps before
revision is run. It serves as an upper bound for adaptive policy accuracy and as a measure of
how much headroom remains.

**Where the numbers come from:**
- `outputs/paper_tables/oracle_routing/oracle_routing_eval_summaries.csv`
- `outputs/paper_tables_cleaned/oracle_routing_eval.csv`
- Upstream: `outputs/oracle_routing_eval/*_oracle_summary.json`

---

### 3. Budget-Aware Frontier / Next-Stage Curve
**What it is:** A curve that maps a *target average cost budget* (e.g., 1.1, 1.2, 1.5) to the
best achievable accuracy if exactly that fraction of queries are revised. The fraction revised
equals `(target_cost − 1.0)`, so cost 1.1 means 10% of queries are revised, cost 1.2 means 20%,
etc.

**How it is computed:** Queries are ranked by a confidence proxy (e.g., routing model score or
a heuristic). The lowest-confidence fraction matching the budget is sent to revision. This is a
frontier-style analysis: it shows the *Pareto-optimal* accuracy for every possible cost level,
assuming the best possible ordering of queries by revision priority.

**Key property:** *Budget-aware and sweep-style.* Unlike a deployed policy (which uses a fixed
threshold learned on training data), the budget-curve picks the threshold retrospectively to
match each target cost. It answers: "If I could spend exactly X on average per query, what is the
best accuracy I could achieve?" The curve is closely related to the oracle analysis but is
parametrised by cost level rather than by routing correctness.

**Where the numbers come from:**
- `outputs/paper_tables/next_stage/next_stage_budget_curves_all_datasets.csv`
- `outputs/paper_tables_cleaned/budget_curves_all_datasets.csv`
- Upstream: `outputs/next_stage_eval/*/budget_curve.csv`

---

## How These Three Objects Relate

```
Oracle routing (upper bound, not deployable)
    ↑  shows how much headroom exists
    
Best practical adaptive policy  ←→  Budget-frontier curve
(single threshold, trained policy)    (sweep over all thresholds /
                                       cost budgets)
```

- The **best practical policy** is a single operating point on (or near) the budget-frontier
  curve, selected by the policy's learned threshold.
- The **budget frontier** shows the full trade-off curve; the practical policy is one dot on it.
- The **oracle upper bound** is the ceiling: the accuracy achievable only with perfect knowledge
  of per-instance revision benefit.

They are related but **not the same object**:
| Property | Practical policy | Oracle | Budget frontier |
|---|---|---|---|
| Deployable at inference | ✓ | ✗ | ✗ (sweep) |
| Uses gold labels | ✗ | ✓ | ✗ |
| Single operating point | ✓ | ✓ (per dataset) | ✗ (curve) |
| Shows full cost trade-off | ✗ | ✗ | ✓ |

---

## Values Used in `clarification_table.csv`

All values are drawn directly from existing artifacts — **no new experiments were run**.

| Column | Source file |
|---|---|
| `reasoning_accuracy` | `final_cross_regime_summary_fixed.csv` |
| `revise_accuracy` | `final_cross_regime_summary_fixed.csv` |
| `best_adaptive_policy_name` | `final_cross_regime_summary_fixed.csv` |
| `best_adaptive_accuracy` | `final_cross_regime_summary_fixed.csv` |
| `best_adaptive_avg_cost` | `final_cross_regime_summary_fixed.csv` |
| `oracle_accuracy` | `oracle_routing_eval.csv` / `oracle_routing_eval_summaries.csv` |
| `oracle_avg_cost` | `oracle_routing_eval.csv` / `oracle_routing_eval_summaries.csv` |
| `budget_frontier_acc_cost_1_1` | `budget_curves_all_datasets.csv` (target_avg_cost = 1.1) |
| `budget_frontier_acc_cost_1_2` | `budget_curves_all_datasets.csv` (target_avg_cost = 1.2) |

---

## Files Created

| File | Description |
|---|---|
| `outputs/paper_export/clarification_table.csv` | Main per-regime summary table (this task) |
| `outputs/paper_export/clarification_notes.md` | These notes |
| `outputs/paper_export/clarification_table.tex` | LaTeX-ready version of the summary table |
