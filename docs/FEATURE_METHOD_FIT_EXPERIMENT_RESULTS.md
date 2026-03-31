# Feature–Method-Fit Experiment Results

**Date:** auto-generated  
**Method:** Offline only — no API calls or new LLM inference.  
**Data:** Existing routing datasets + policy-eval outputs.  
**Total rows:** 400 (gsm8k_random_100: 100, hard_gsm8k_100: 100, hard_gsm8k_b2: 100, math500_100: 100)

---

## 0. Dataset and Outcome Summary

### Regime counts

| Regime | N |
|---|---|
| `gsm8k_random_100` | 100 |
| `hard_gsm8k_100` | 100 |
| `hard_gsm8k_b2` | 100 |
| `math500_100` | 100 |

### Outcome class sizes (across all regimes)

| Outcome label | Count | Rate |
|---|---|---|
| `revise_helpful` | 29 | 0.072 |
| `safe_cheap` | 316 | 0.790 |
| `both_wrong` | 55 | 0.138 |
| `unnecessary_revise_candidate` | 90 | 0.225 |

---

## 1. Which features are most associated with revise-helpful cases?

Features with the highest mean in the `revise_helpful` group (RG wrong, DPR correct):

| Feature | Side | Mean (revise_helpful) | Mean (safe_cheap) | Effect size proxy |
|---|---|---|---|---|
| `prompt_token_length` | question | 66.7586 | 52.8703 | 13.8884 |
| `prompt_number_count` | question | 7.4138 | 4.8070 | 2.7021 |
| `final_answer_parseable` | output | 1.0000 | 1.0000 | 0.0182 |
| `answer_error_signal` | output | 0.8276 | 0.2025 | 0.6251 |
| `special_structure_presence` | question | 0.7586 | 0.7057 | 0.3283 |
| `explicit_constraint_presence` | question | 0.5862 | 0.4146 | 0.2589 |
| `multi_stepness_proxy` | question | 0.3448 | 0.3165 | 0.2111 |
| `relational_wording_presence` | question | 0.3103 | 0.2532 | 0.2293 |

**Observation:** Features with high `mean_revise_helpful` and a positive difference versus `mean_safe_cheap` indicate that when a revise is genuinely helpful, the routing features are systematically different from when the cheap route already succeeds.

---

## 2. Which features are most associated with safe cheap routing?

Features with the highest mean in the `safe_cheap` group (RG correct):

| Feature | Side | Mean (safe_cheap) | Mean (revise_helpful) | Effect size proxy |
|---|---|---|---|---|
| `prompt_token_length` | question | 52.8703 | 66.7586 | 13.8884 |
| `prompt_number_count` | question | 4.8070 | 7.4138 | 2.7021 |
| `final_answer_parseable` | output | 1.0000 | 1.0000 | 0.0182 |
| `cheap_route_confidence` | output | 0.8354 | 0.2414 | 0.5941 |
| `special_structure_presence` | question | 0.7057 | 0.7586 | 0.3283 |
| `explicit_constraint_presence` | question | 0.4146 | 0.5862 | 0.2589 |
| `copied_question_number_as_final_answer` | output | 0.3924 | 0.2759 | 0.1969 |
| `multi_stepness_proxy` | question | 0.3165 | 0.3448 | 0.2111 |

---

## 3. Which features are most associated with both-wrong hard cases?

| Feature | Side | Mean (both_wrong) | Mean (safe_cheap) | Effect size proxy |
|---|---|---|---|---|
| `prompt_token_length` | question | 53.8727 | 52.8703 | 13.8884 |
| `prompt_number_count` | question | 7.5091 | 4.8070 | 2.7021 |
| `final_answer_parseable` | output | 0.9818 | 1.0000 | 0.0182 |
| `answer_error_signal` | output | 0.5636 | 0.2025 | 0.6251 |
| `special_structure_presence` | question | 0.5273 | 0.7057 | 0.3283 |
| `copied_question_number_as_final_answer` | output | 0.4727 | 0.3924 | 0.1969 |
| `cheap_route_confidence` | output | 0.4182 | 0.8354 | 0.5941 |
| `explicit_constraint_presence` | question | 0.3273 | 0.4146 | 0.2589 |

---

## 4. Which features appear to favor each routing method?

Feature means by best-method group:

| method_best_label | n | prompt_token_length | prompt_number_count | answer_error_signal | cheap_route_confidence | body_final_numeric_mismatch | special_structure_presence |
|---|---|---|---|---|---|---|---|
| `direct_plus_revise` | 29 | 66.7586 | 7.4138 | 0.8276 | 0.2414 | 0.1034 | 0.7586 |
| `reasoning_greedy` | 371 | 53.0189 | 5.2075 | 0.2561 | 0.7736 | 0.1429 | 0.6792 |

---

## 5. Do feature results support the answer-error > explanation-warning story?

Key comparisons:

| Signal | Side | Mean (revise_helpful) | Mean (safe_cheap) | Effect proxy |
|---|---|---|---|---|
| `answer_error_signal` | output | 0.8276 | 0.2025 | 0.6251 |
| `explanation_warning_signal` | output | 0.1034 | 0.0158 | 0.0876 |
| `cheap_route_confidence` | output | 0.2414 | 0.8354 | 0.5941 |

**Result:** `answer_error_signal` has a larger effect-size proxy (0.6251) than `explanation_warning_signal` (0.0876) for separating `revise_helpful` from `safe_cheap`. This supports the story that answer-error-focused signals are more discriminative than generic explanation irregularity.

Logistic regression coefficients (standardised):

| Feature | Coefficient |
|---|---|
| `cheap_route_confidence` | -0.9734 |
| `answer_error_signal` | 0.9066 |
| `special_structure_presence` | 0.5188 |
| `copied_question_number_as_final_answer` | -0.4926 |
| `constraint_violation_signal` | -0.4068 |
| `prompt_token_length` | 0.3770 |
| `prompt_number_count` | 0.3253 |
| `explicit_constraint_presence` | 0.2610 |

The logistic model also ranks `answer_error_signal` (coef=0.9066) above `explanation_warning_signal` (coef=0.1690).

---

## 6. Which 8–10 features should be kept in the manuscript?

### Recommended features (`yes`)

| Feature | Side | Effect proxy | Interpretability | Notes |
|---|---|---|---|---|
| `prompt_number_count` | question | 2.7021 | 5 | — |
| `answer_error_signal` | output | 0.6251 | 5 | — |
| `cheap_route_confidence` | output | 0.5941 | 5 | Summary signal; correlated with answer_error_signal |
| `special_structure_presence` | question | 0.3283 | 4 | — |
| `constraint_violation_signal` | output | 0.2861 | 4 | — |
| `relational_wording_presence` | question | 0.2293 | 4 | — |

### Conditional features (`maybe`)

| Feature | Side | Effect proxy | Interpretability | Notes |
|---|---|---|---|---|
| `prompt_token_length` | question | 13.8884 | 5 | Correlated with prompt_number_count |
| `explicit_constraint_presence` | question | 0.2589 | 4 | Subset of relational_wording_presence signals |
| `target_quantity_mismatch` | output | 0.2374 | 4 | Subset of constraint_violation_signal |
| `multi_stepness_proxy` | question | 0.2111 | 4 | Overlaps with explicit_constraint_presence |
| `explanation_warning_signal` | output | 0.0876 | 4 | Weak association expected from V6 architecture |
| `final_answer_parseable` | output | 0.0182 | 5 | — |
| `target_quantity_type` | question |  | 5 | — |

---

## 7. Caveats

1. **Sample size:** Each regime has 100 rows; combined dataset has 400 rows across 4 regimes. Effect sizes should be treated as exploratory and order-of-magnitude only.

2. **Class imbalance:** `revise_helpful` is rare (see outcome summary above). Logistic regression and decision tree results may be dominated by the majority class (`safe_cheap`). Cross-validated accuracy may be misleading on imbalanced data.

3. **V6/V7 features only for output-side:** Features F8–F15 are derived from V6/V7 feature extraction run offline on real first-pass outputs. Results are specific to the `gpt-4o-mini` model run captured in the routing datasets.

4. **`copied_question_number_as_final_answer` (F12)** is a question-side proxy (`tq_potential_answer_echo_risk`) rather than an output-side signal, as no direct output-side echo detection column exists in the routing CSVs.

5. **`body_final_numeric_mismatch` (F9)** relies on `v7_extra_answer_error`, which fires only for the two specific V7 patterns (weekday+numeric, need_more+list_price, tail_equals). Its base rate is very low in most regimes.

6. **Evidence labels:** All population-level claims carry `exploratory_only` status per the repo's existing evidence conventions. The fixture- and probe-level V5/V6/V7 analysis in `docs/V5_V6_V7_STORY_CHECK.md` remains the more controlled evidence for the routing-signal story.

---

## Appendix: Decision Tree Feature Importances

| Feature | Importance |
|---|---|
| `prompt_token_length` | 0.3091 |
| `answer_error_signal` | 0.2319 |
| `body_final_numeric_mismatch` | 0.2190 |
| `cheap_route_confidence` | 0.2088 |
| `prompt_number_count` | 0.0149 |
| `multi_stepness_proxy` | 0.0113 |
| `constraint_violation_signal` | 0.0032 |
| `copied_question_number_as_final_answer` | 0.0019 |
| `explicit_constraint_presence` | 0.0000 |
| `relational_wording_presence` | 0.0000 |

## Appendix: Logistic Regression CV Performance

3-fold cross-validated accuracy: **0.9250** (±0.0163).  
Note: this is class-imbalanced data; accuracy may be near the majority-class baseline.

---

*Report auto-generated by `scripts/run_feature_method_fit_analysis.py`.*