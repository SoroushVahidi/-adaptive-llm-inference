# Statistical Methods – Lightweight

This document describes the statistical methods used in the lightweight paper-strengthening
pass.  All computations use only pre-existing stored outputs; no new LLM inference was run.

---

## Source Artifacts

All statistical tests are derived from the following per-query binary correctness vectors:

| Regime | File |
|---|---|
| `math500_100` | `outputs/real_math500_policy_eval/per_query_policy_decisions.csv` |
| `hard_gsm8k_100` | `outputs/real_hard_gsm8k_policy_eval/per_query_policy_decisions.csv` |
| `hard_gsm8k_b2` | `outputs/real_hard_gsm8k_b2_policy_eval/per_query_policy_decisions.csv` |
| `gsm8k_random_100` | `outputs/real_policy_eval/per_query_policy_decisions.csv` |

Each file contains one row per query (n=100) with binary columns:
- `reasoning_correct` – 1 if Always-RG is correct, 0 otherwise
- `revise_correct` – 1 if Always-DPR is correct, 0 otherwise
- `correct_if_v5`, `correct_if_v6`, `correct_if_v7` – 1 if the adaptive policy is correct

---

## Bootstrap Confidence Intervals

**File:** `outputs/paper_tables/bootstrap_accuracy_ci.csv`

**Method:**
- Non-parametric bootstrap over queries (sampling with replacement)
- Fixed random seed: 42
- Number of resamples: 10,000
- 95% CI: 2.5th and 97.5th percentiles of the bootstrap distribution of the mean

**Applied to:**
- Always-RG accuracy (binary: `reasoning_correct`)
- Always-DPR accuracy (binary: `revise_correct`)
- Adaptive v5, v6, v7 accuracy (binary: `correct_if_v{5,6,7}`)
- Oracle accuracy (binary: `max(reasoning_correct, revise_correct)`)

**Rationale:**  
Bootstrap is appropriate here because:
1. We have per-query binary outcomes (not a parametric distribution assumption).
2. Sample sizes are small (n=100), making asymptotic normality less reliable.
3. Bootstrap directly reflects variability in the test sample.

---

## Paired Difference Tests

**File:** `outputs/paper_tables/paired_difference_tests.csv`

**Method:**
- Paired bootstrap over query-level differences (a_i - b_i) for pairs (a, b)
- Same seed (42) and number of resamples (10,000) as above
- 95% CI: percentile bootstrap of the mean difference
- p-value: two-sided, computed as `2 * min(p_one_sided, 1 - p_one_sided)` where
  `p_one_sided` is the fraction of bootstrap samples with mean difference ≤ 0

**Comparisons computed:**
1. `best_adaptive minus reasoning_greedy` – tests whether the best adaptive policy significantly outperforms Always-RG
2. `direct_plus_revise minus best_adaptive` – tests whether DPR still significantly outperforms the best adaptive policy
3. `oracle minus best_adaptive` – tests whether there is a significant remaining gap to the oracle

**Why paired?**
All policies are evaluated on the same fixed set of queries, so pairing removes query-level
difficulty variance.  The test statistic is the mean of per-query differences, which is more
powerful than an unpaired comparison.

**Interpretation note:**
With n=100 and effects of 2–8 percentage points, the tests are underpowered for small effects
(p > 0.05 does not confirm the null).  CIs are more informative than p-values for effect sizes
of this magnitude.

---

## Oracle Correctness

The oracle binary vector is constructed as:
```python
oracle_correct_i = max(reasoning_correct_i, revise_correct_i)
```
This represents the best achievable outcome per query given the two available actions.
The oracle accuracy from `outputs/oracle_routing_eval/` matches this construction.

---

## Cost Model

All cost computations use the following deterministic model:
- cheap action (reasoning_greedy): cost = 1.0
- expensive action (direct_plus_revise): cost = 2.0
- adaptive policy cost: `(1 - revise_rate) * 1.0 + revise_rate * 2.0`

Alternative cost ratios (1:1.5, 1:2, 1:3) are explored in
`outputs/paper_tables/cost_ratio_sensitivity.csv` and
`outputs/paper_tables/policy_ranking_stability.csv`.

---

## Implementation

All computations are in `scripts/generate_lightweight_paper_artifacts.py`, functions
`_bootstrap_ci` and `_paired_diff_bootstrap`.  The script is deterministic given the
fixed seed and produces the same outputs on every run.
