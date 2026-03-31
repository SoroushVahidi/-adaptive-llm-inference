# Signal Ablation Definitions (Repository-Grounded)

## Scope

This file defines the signal groups used in the offline signal-ablation experiment. Definitions are grounded in:

- `src/policies/adaptive_policy_v6.py` (explicit separation logic),
- `src/policies/adaptive_policy_v7.py` (targeted answer-error additions),
- enriched dataset columns in `data/real_*_routing_dataset_enriched.csv`.

`docs/V5_V6_V7_STORY_CHECK.md` is not present in this repository at time of writing.

---

## Core grouping principle

- **answer_error_signals**: indicators intended to imply likely wrong final answer and therefore stronger revise-worthiness.
- **explanation_warning_signals**: indicators of incomplete/odd reasoning trace quality that do not necessarily imply wrong final answer.

This follows V6’s stated invariant: explanation incompleteness is not equivalent to answer wrongness.

---

## A) answer_error_signals

## Direct aggregate columns used in ablation

1. `v6_answer_error_score`
2. `v6_final_answer_confident` (used as context/ambiguity, not direct threshold in this ablation)
3. `v7_answer_error_score` and `v7_extra_answer_error` (audited for mapping, not the main ablation score)

## Underlying feature families mapped into answer_error_score (from code + columns)

### Constraint/consistency-like columns (`cons_*`)
- `cons_answer_type_mismatch_suspected`
- `cons_target_quantity_mismatch_suspected`
- `cons_unit_mismatch_suspected`
- `cons_impossible_sign_suspected`
- `cons_integer_expected_but_noninteger_suspected`
- `cons_percent_or_ratio_mismatch_suspected`
- `cons_answer_not_mentioned_in_final_statement_suspected`
- `cons_constraint_word_conflict_suspected`
- `cons_bound_violation_suspected`
- `cons_obvious_upper_bound_exceeded_suspected`
- `cons_obvious_lower_bound_violated_suspected`

### First-pass reliability adjuncts (indirect)
- `fp_first_pass_parse_success`
- `fp_first_pass_empty_or_malformed_flag`
- `fp_first_pass_has_final_answer_cue`

### V7 targeted answer-risk additions (code-level)
- weekday question + numeric final mismatch,
- “how much more” answer equals first dollar amount,
- tail equation value disagrees with final answer,
- low-confidence escalation in categorical/high-warning settings.

---

## B) explanation_warning_signals

## Direct aggregate column used in ablation

1. `v6_explanation_warning_score`

## Underlying explanation-warning family (code-level)

From V6 scoring logic, explanation warning includes:
- missing required number echoes,
- possible intermediate-stop suspicion,
- required-number missing by operation type,
- short reasoning traces.

In enriched CSVs, explanation-family raw components are not all exposed separately; the grounded proxy is the aggregate `v6_explanation_warning_score`.

---

## C) Ambiguous signals (documented as ambiguous)

These can correlate with both answer correctness and explanation quality:

1. `role_warning_score`  
   - Used in earlier policy generations; can encode both concise-correct behavior and genuine risk.
2. `unified_error_score` / `unified_confidence_score`  
   - Composite by design; not purely answer-error or explanation-warning.
3. `step_*` structural metrics  
   - May represent completeness/readability rather than final answer correctness.
4. length/prolixity proxies (`reasoning_raw_chars`, `fp_first_pass_output_length`)  
   - Risk of superficial correlation.

For this ablation, ambiguous signals are **not** used as primary independent groups to preserve interpretability.

---

## D) Exact mapping used in the implemented ablation variants

Ablation variants were defined on two primary aggregates available in all target regimes:

- `A = v6_answer_error_score`
- `E = v6_explanation_warning_score`

with grounded thresholds from V6 defaults:
- answer-error threshold: `A >= 2`
- explanation-warning threshold: `E >= 3`

Variants:

1. **`explanation_only_router`**: revise iff `E >= 3`
2. **`answer_error_only_router`**: revise iff `A >= 2`
3. **`combined_equal_router`**: revise iff `A + E >= 3`
4. **`answer_error_dominant_router`**: revise iff `2A + E >= 4`
5. **`explanation_dominant_router`**: revise iff `A + 2E >= 4`

These are offline proxies for the conceptual question “answer-error-focused vs explanation-warning-focused escalation,” using only committed artifact columns.

---

## E) Feasibility notes

- Exact re-execution of all internal V6/V7 sub-signal branches per query is not always feasible from CSV-only artifacts because some intermediate role/consistency internals are not fully materialized per row.
- Therefore, this experiment uses the most grounded, regime-consistent proxies available (`v6_answer_error_score`, `v6_explanation_warning_score`) and documents this approximation explicitly.
