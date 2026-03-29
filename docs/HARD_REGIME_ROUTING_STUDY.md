# Hard-Regime Routing Study (MATH500 + Hard GSM8K)

**Evidence:** **measured_now** for numbers that match committed artifacts in `outputs/` and `data/` from this branch. **exploratory_only** for generalization beyond these three 100-query slices.

---

## Motivation

The first random GSM8K-100 run showed **~90%** one-shot reasoning accuracy and **2%** `revise_helpful`, so learned routing collapsed. This follow-up asks whether **harder regimes** increase revise opportunity and make routing models less degenerate.

## Provider / model

- **OpenAI** `gpt-4o-mini`, temperature 0, same cost proxy as before (1 = reasoning only, 2 = +revise pipeline).

## Regimes

| Regime | Construction | N |
|--------|--------------|---|
| **gsm8k_random100** | First real study: HF GSM8K test, sequential cap | 100 |
| **math500_100** | `HuggingFaceH4/MATH-500`, first 100 rows | 100 |
| **hard_gsm8k_100** | Top 100 by question-side hardness over **1319** GSM8K test items (HF; full test has 1319) | 100 |

### Hard GSM8K selection

Script: `scripts/run_select_hard_gsm8k.py`  
Artifacts: `outputs/hard_regime_selection/hard_gsm8k_selection.csv`, `selection_summary.json`

**Score (higher = harder):**

`z_len + z_num + 0.5*multi_step + 0.5*equation + 0.3*percent + 0.3*fraction + 0.15*target_quantity_true_count`

where `z_len` / `z_num` are population z-scores over the pool using `extract_query_features`, and `target_quantity_true_count` counts True flags in `extract_target_quantity_features`.

**Bugfix during MATH500 run:** `constraint_violation_features._question_profile` skipped non-`Decimal` tokens (e.g. `\\frac` fragments) so MATH questions no longer crash feature extraction.

## revise_helpful prevalence

| Regime | Prevalence | Count (of 100) |
|--------|------------|----------------|
| gsm8k_random100 | **2%** | 2 |
| math500_100 | **6%** | 6 |
| hard_gsm8k_100 | **12%** | 12 |

**Conclusion (measured):** Prevalence **increases** in harder slices; hard GSM8K shows a **6×** lift vs random GSM8K on this draw.

## Routing value vs random GSM8K

- **hard_gsm8k:** `direct_plus_revise` **+7 pp** over reasoning (0.86 vs 0.79); `revise_helpful` 12%.
- **math500:** reasoning and full revise **tie** at **0.64** on this run — second stage rarely fixes symbolic/non-numeric errors under current extraction.
- **gsm8k_random100:** small +2 pp from always-revise; sparse positives.

## V6 vs V7 (heuristic only)

See `outputs/cross_regime_comparison/cross_regime_summary.json`.

| Regime | V6 acc / cost | V7 acc / cost | Winner (acc) |
|--------|---------------|---------------|----------------|
| gsm8k_random100 | 0.92 / 1.18 | 0.92 / 1.30 | Tie |
| math500_100 | 0.65 / 1.03 | 0.65 / 1.09 | Tie |
| hard_gsm8k_100 | 0.81 / 1.26 | 0.82 / 1.46 | V7 (+1 pp, +0.20 cost) |

**measured_now:** On hard GSM8K, V7 slightly improves accuracy at **higher** revise rate.

## Learned router (CV)

| Regime | Positives | Best F1 (approx) | Routing sim note |
|--------|-----------|------------------|------------------|
| gsm8k_random100 | 2 | ~0 | Collapse |
| math500_100 | 6 | ~0.4 (tree) | Still weak vs always-revise |
| hard_gsm8k_100 | 12 | ~0.69 (bagging) | **Bagging** sim: **0.88** acc, **1.14** cost vs always-revise 0.86 @ 2.0 |

**exploratory_only:** N=100, 5-fold CV with rare positives — treat F1 and simulation as **signals**, not production claims.

## Bottleneck (after hard study)

1. **measured_now:** Easy GSM8K had **too few** revise-helpful positives; harder slices **partially** fix label sparsity.
2. **measured_now:** **MATH500** breaks the “revise fixes reasoning” story when **gold is non-numeric** and **direct+revise** stays numeric-heavy — need aligned extraction or strategy if MATH is a target.
3. **exploratory_only:** Cost–accuracy Pareto needs **larger N** and possibly **stratified** hard pools.

## Commands

```bash
python3 scripts/run_select_hard_gsm8k.py --subset-size 100 --pool-size 2000
python3 scripts/run_build_math500_routing_dataset.py --subset-size 100
python3 scripts/run_build_hard_gsm8k_routing_dataset.py
python3 scripts/run_real_policy_eval.py --dataset-csv data/real_math500_routing_dataset.csv \
  --output-dir outputs/real_math500_policy_eval
python3 scripts/run_real_policy_eval.py --dataset-csv data/real_hard_gsm8k_routing_dataset.csv \
  --output-dir outputs/real_hard_gsm8k_policy_eval
python3 scripts/run_real_routing_model_eval.py --dataset-csv data/real_math500_routing_dataset.csv \
  --output-dir outputs/real_math500_routing_model
python3 scripts/run_real_routing_model_eval.py --dataset-csv data/real_hard_gsm8k_routing_dataset.csv \
  --output-dir outputs/real_hard_gsm8k_routing_model
python3 scripts/run_cross_regime_comparison.py
```
