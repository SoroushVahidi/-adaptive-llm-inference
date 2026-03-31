# Bootstrap Uncertainty Analysis — Notes
Generated: 2026-03-31 21:26 UTC
## 1. Canonical Regime & Policy Evidence
The four canonical main-paper regimes and the primary adaptive policy
(adaptive_policy_v5) were inferred exclusively from the following
committed repository files:
- `FINAL_MANUSCRIPT_QUICKSTART.md`
- `outputs/paper_tables_final/main_results_summary.csv`
- `outputs/paper_tables_final/cross_regime_summary.csv`
- `outputs/real_policy_eval/per_query_policy_decisions.csv`
- `outputs/real_hard_gsm8k_policy_eval/per_query_policy_decisions.csv`
- `outputs/real_hard_gsm8k_b2_policy_eval/per_query_policy_decisions.csv`
- `outputs/real_math500_policy_eval/per_query_policy_decisions.csv`

**Key evidence:**
- `FINAL_MANUSCRIPT_QUICKSTART.md` names the four regimes
  (gsm8k_random_100, hard_gsm8k_100, hard_gsm8k_b2, math500_100)
  and designates `adaptive_policy_v5` as the canonical primary policy.
- `outputs/paper_tables_final/main_results_summary.csv` lists
  `adaptive_policy_v5` as `adaptive_primary_policy` for all four regimes.

## 2. Per-Query Artifacts Used
- `outputs/real_policy_eval/per_query_policy_decisions.csv`
- `outputs/real_hard_gsm8k_policy_eval/per_query_policy_decisions.csv`
- `outputs/real_hard_gsm8k_b2_policy_eval/per_query_policy_decisions.csv`
- `outputs/real_math500_policy_eval/per_query_policy_decisions.csv`

Each file contains 100 rows (one per query) with columns:
`reasoning_correct`, `revise_correct`, `revise_helpful`,
`correct_if_v5`, `correct_if_v6`, `correct_if_v7`.

## 3. Oracle Construction
Oracle accuracy was reconstructed per query from committed data:

```
oracle_correct[i] = reasoning_correct[i]  if revise_helpful[i] == 0
                   = revise_correct[i]     if revise_helpful[i] == 1
```

This equals `max(reasoning_correct, revise_helpful)` because
`revise_helpful==1` implies `revise_correct==1` and `reasoning_correct==0`.
Resulting per-regime aggregate oracle accuracies match the committed
`outputs/oracle_routing_eval/*_oracle_summary.json` values exactly.

## 4. Bootstrap Method
- **Resamples:** 10,000
- **CI level:** 95%  (percentile method)
- **RNG seed:** 42
- **Paired by query:** Yes — each resample draws row indices with
  replacement; both policy outcomes at the same row index are kept
  together, so the within-query pairing is preserved.
- **Statistic:** mean accuracy difference (minuend minus subtrahend)

## 5. Results
| Regime | Comparison | Observed Δ | 95 % CI | CI excludes 0? |
|--------|------------|:----------:|:-------:|:--------------:|
| GSM8K Random-100 | adaptive_v5 minus always_cheap | +0.020 | [+0.000, +0.050] | no |
| GSM8K Random-100 | adaptive_v5 minus always_revise | +0.000 | [+0.000, +0.000] | no |
| GSM8K Random-100 | oracle minus adaptive_v5 | +0.000 | [+0.000, +0.000] | no |
| Hard GSM8K-100 | adaptive_v5 minus always_cheap | +0.070 | [+0.000, +0.150] | no |
| Hard GSM8K-100 | adaptive_v5 minus always_revise | +0.000 | [-0.030, +0.030] | no |
| Hard GSM8K-100 | oracle minus adaptive_v5 | +0.050 | [+0.010, +0.100] | **yes** |
| Hard GSM8K-B2 | adaptive_v5 minus always_cheap | +0.080 | [+0.030, +0.140] | **yes** |
| Hard GSM8K-B2 | adaptive_v5 minus always_revise | +0.000 | [-0.030, +0.030] | no |
| Hard GSM8K-B2 | oracle minus adaptive_v5 | +0.010 | [+0.000, +0.030] | no |
| MATH500-100 | adaptive_v5 minus always_cheap | +0.020 | [-0.040, +0.080] | no |
| MATH500-100 | adaptive_v5 minus always_revise | +0.020 | [-0.020, +0.060] | no |
| MATH500-100 | oracle minus adaptive_v5 | +0.040 | [+0.010, +0.080] | **yes** |

## 6. Interpretation
**Claim 1 — Adaptive v5 outperforms always-cheap (reasoning_greedy):**
Supported with 95% CI excluding zero in: Hard GSM8K-B2.
In the remaining regimes the confidence interval crosses zero,
indicating the difference is not individually significant at n=100,
but the point estimate is non-negative in all four regimes.

**Claim 2 — Adaptive v5 accuracy matches always-revise at lower cost:**
The v5−revise difference CIs are centered near zero for all regimes.
This supports the paper's claim that v5 achieves comparable accuracy
to always-revise while incurring significantly lower cost.
(Cost comparisons are taken directly from committed summary tables;
this bootstrap is restricted to accuracy differences.)

**Claim 3 — Residual oracle gap (remaining headroom):**
A statistically significant oracle gap exists in: Hard GSM8K-100, MATH500-100.
This confirms that the routing policy has not yet captured all
revise-helpful queries and that non-trivial headroom remains.

**Overall:** The bootstrap analysis is consistent with the paper's
main claims. The small per-regime sample size (n=100) limits statistical
power, particularly for regimes with very low revise-helpful rates (e.g.,
GSM8K Random-100, revise-helpful rate = 2%). The hard regimes
(Hard GSM8K-100 and Hard GSM8K-B2), which have higher revise-helpful
rates (12% and 9% respectively), show the strongest statistical support.

## 7. Files Generated
- `outputs/uncertainty_analysis/bootstrap_summary.csv` — full results table
- `outputs/uncertainty_analysis/bootstrap_summary.json` — machine-readable
- `outputs/uncertainty_analysis/bootstrap_notes.md` — this document
