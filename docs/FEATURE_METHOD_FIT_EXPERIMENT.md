# Feature–Method-Fit Experiment: Feature Schema

## Overview

This document defines the 15-feature candidate set used in the offline
feature-method-fit analysis.  All features are derived from columns that
already exist in the per-regime routing datasets
(`data/real_*_routing_dataset.csv` / `..._enriched.csv`).  No new model
calls or API requests are required.

**Script:** `scripts/run_feature_method_fit_analysis.py`  
**Core module:** `src/analysis/feature_method_fit.py`  
**Output directory:** `outputs/feature_method_fit/`

---

## 15 Candidate Features

### Question-Side Features (1–7)

| # | Conceptual name | Source column(s) | Type | Notes |
|---|---|---|---|---|
| 1 | `prompt_number_count` | `q_num_numeric_mentions` | numeric | Direct mapping. Count of numeric tokens in the question. |
| 2 | `prompt_token_length` | `q_question_length_tokens_approx` | numeric | Direct mapping. Approximate word-token count of question. |
| 3 | `target_quantity_type` | `tq_asks_remaining_or_left`, `tq_asks_total`, `tq_asks_difference`, `tq_asks_rate_or_unit`, `tq_asks_money`, `tq_asks_time` | categorical str | Priority-ordered label: first matching `tq_*` flag wins; defaults to `"other"`. |
| 4 | `multi_stepness_proxy` | `tq_has_multi_operation_hint` | binary 0/1 | Direct mapping. Proxy for multi-step arithmetic requirement. |
| 5 | `explicit_constraint_presence` | `tq_has_subtraction_trap_verb` OR `tq_has_addition_trap_structure` | binary 0/1 | Derived: 1 if either column is non-zero. |
| 6 | `relational_wording_presence` | `tq_asks_difference` OR `tq_asks_remaining_or_left` | binary 0/1 | Derived: 1 if either column is non-zero. Covers "how much more / left" phrasing. |
| 7 | `special_structure_presence` | `q_has_percent_symbol` OR `q_has_fraction_pattern` OR `tq_asks_rate_or_unit` OR `tq_asks_time` | binary 0/1 | Derived: 1 if any column is non-zero. Covers unit, fraction, percent, time, rate structures. |

### Cheap-Output-Side Features (8–15)

| # | Conceptual name | Source column(s) | Type | Notes |
|---|---|---|---|---|
| 8 | `final_answer_parseable` | `fp_first_pass_parse_success` | binary 0/1 | Direct mapping. Whether the first-pass answer was successfully parsed. |
| 9 | `body_final_numeric_mismatch` | `v7_extra_answer_error > 0` | binary 0/1 | Proxy: V7 extra error > 0 indicates one of the V7-specific structural mismatch signals fired (e.g. weekday question with numeric final, "how much more" returns list price, tail-equals disagreement). |
| 10 | `target_quantity_mismatch` | `cons_target_quantity_mismatch_suspected` | binary 0/1 | Direct mapping from constraint-violation feature extraction. |
| 11 | `constraint_violation_signal` | `cons_answer_type_mismatch_suspected` OR `cons_unit_mismatch_suspected` OR `cons_impossible_sign_suspected` OR `cons_integer_expected_but_noninteger_suspected` OR `cons_constraint_word_conflict_suspected` OR `cons_bound_violation_suspected` | binary 0/1 | Derived: 1 if any strong constraint signal is non-zero. Excludes `cons_target_quantity_mismatch_suspected` (covered by feature 10). |
| 12 | `copied_question_number_as_final_answer` | `tq_potential_answer_echo_risk` | binary 0/1 | Closest available proxy. This is a question-side signal but identifies prompts where a question literal could be mistaken for the answer. |
| 13 | `cheap_route_confidence` | `v6_final_answer_confident` | binary 0/1 | Direct mapping. V6 trust gate: True when parse ok + finalization cue + answer_error == 0 or coherent coverage. |
| 14 | `explanation_warning_signal` | `v6_explanation_warning_score > 0` | binary 0/1 | Derived: 1 if V6 explanation warning score is positive (role echoes missing, short reasoning, intermediate stop). |
| 15 | `answer_error_signal` | `v6_answer_error_score > 0` | binary 0/1 | Derived: 1 if V6 answer error score is positive (constraint violations, parse failure, consistency checks). |

---

## Outcome Labels

| Label | Definition |
|---|---|
| `revise_helpful` | `reasoning_correct == 0` AND `revise_correct == 1` — RG is wrong, DPR is correct. |
| `safe_cheap` | `reasoning_correct == 1` — RG is correct; no escalation needed. |
| `both_wrong` | `reasoning_correct == 0` AND `revise_correct == 0` — neither RG nor DPR is correct. |
| `unnecessary_revise_candidate` | `reasoning_correct == 1` AND (`v6_revise_recommended == 1` OR `v7_revise_recommended == 1`) — RG is correct but a policy would trigger revise. |
| `method_best_label` | Among {RG=`reasoning_greedy`, DPR=`direct_plus_revise`, v5, v6, v7}: the method(s) with highest correctness (primary) and lowest cost (tie-break). String label, e.g. `"reasoning_greedy"` or `"direct_plus_revise"`. |

**Note on `method_best_label`:** Derived from `reasoning_correct`, `revise_correct`, `correct_if_v5`, `correct_if_v6`, `correct_if_v7`, `cost_v5`, `cost_v6`, `cost_v7` columns available in the policy-eval per-query CSVs.  When multiple methods tie on correctness, the lowest-cost one wins.

---

## Data Sources by Regime

| Regime label | Routing dataset CSV | Policy decisions CSV |
|---|---|---|
| `gsm8k_random_100` | `data/real_gsm8k_routing_dataset.csv` | `outputs/real_policy_eval/per_query_policy_decisions.csv` |
| `hard_gsm8k_100` | `data/real_hard_gsm8k_routing_dataset.csv` | `outputs/real_hard_gsm8k_policy_eval/per_query_policy_decisions.csv` |
| `hard_gsm8k_b2` | `data/real_hard_gsm8k_b2_routing_dataset.csv` | `outputs/real_hard_gsm8k_b2_policy_eval/per_query_policy_decisions.csv` |
| `math500_100` | `data/real_math500_routing_dataset.csv` | `outputs/real_math500_policy_eval/per_query_policy_decisions.csv` |

---

## Reproducibility

```bash
python3 scripts/run_feature_method_fit_analysis.py
```

All outputs are deterministic (no randomness beyond train/test splits in the
logistic regression, which use `random_state=42`).  Re-running the script
overwrites the output directory.

---

## Related Files

- `src/analysis/feature_method_fit.py` — core analysis module
- `scripts/run_feature_method_fit_analysis.py` — runner script
- `outputs/feature_method_fit/` — all generated CSVs and figure
- `docs/FEATURE_METHOD_FIT_EXPERIMENT_RESULTS.md` — grounded findings report
- `outputs/paper_tables/feature_method_fit_main_table.csv` — manuscript table
