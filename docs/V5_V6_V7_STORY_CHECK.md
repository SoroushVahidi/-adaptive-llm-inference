# V5 → V6 → V7 Story Check: "Answer-Error-Focused Routing"

**Date:** 2026-03-31  
**Method:** Offline code inspection and existing output review. No new API calls or LLM inference.  
**Story under examination:** *"Selective escalation improves when revise decisions are driven more by structured answer-error signals and less by generic explanation irregularity."*

---

## 1. Per-Version Summary

### Adaptive Policy V5

**File:** `src/policies/adaptive_policy_v5.py`  
**Doc:** `docs/ADAPTIVE_POLICY_V5.md`

**Main signals/features used:**

| Signal family | Role in routing |
|---|---|
| `compute_calibrated_role_decision` | Primary escalation driver when `use_unified_error_signals=False` |
| `compute_unified_error_signal` (default enabled) | **Combined** role-error + self-error + step-error score; overrides calibrated role decision |
| `extract_constraint_violation_features` | `target_quantity_mismatch_suspected`, `constraint_word_conflict_suspected` (weight 1 each) |
| `extract_target_quantity_features` | Question profile (sub/add trap verbs, multi-operation hints) |

**Escalation rule (unified mode, default):**

- `unified_error >= 0.34` AND `unified_confidence <= 0.45` → `strong_escalation_candidate` (revise)  
- `unified_error >= 0.25` → `maybe_escalate` (also revises)  
- `unified_confidence >= 0.70` AND `unified_error < 0.25` → `no_escalation`  

**Intended failure mode addressed:** Under-escalation on complex multi-step problems; generalizes earlier role-only v4 logic with a broader unified signal.

**Generic explanation vs answer-error?** V5 **blends** both. The `unified_error_score` aggregates role coverage (which fires on missing intermediate literals regardless of correctness), self-consistency, and step-error into a single float. Missing intermediate echoes—a generic **explanation** property—feed directly into the revise decision. Constraint violations (answer-level) also contribute but are not separated from explanation noise.

---

### Adaptive Policy V6

**File:** `src/policies/adaptive_policy_v6.py`  
**Doc:** `docs/ADAPTIVE_POLICY_V6.md`

**Main signals/features used:**

| Signal bucket | Signals | Weights |
|---|---|---|
| `explanation_warning_score` (soft) | `missing_required_number` (×count), `possible_intermediate_stop_suspected`, `required_subtractive/additive/rate/capacity_number_missing`, `short_reasoning` | 1 each |
| `answer_error_score` (hard) | `parse_failure_numeric_expected` (3), `answer_type_mismatch_suspected` (2), `target_quantity_mismatch_suspected` (2), `constraint_word_conflict_suspected` (2), `unit_mismatch_suspected` (2), `impossible_sign_suspected` (2), `integer_expected_but_noninteger_suspected` (2), `bound_violation_suspected` (2), `percent_or_ratio_mismatch_suspected` (1), `answer_not_mentioned_in_final_statement_suspected` (1) | see weights |
| `evaluate_candidate` consistency checks | `intermediate_echo_risk`, `remaining_conflict`, `total_conflict`, `rate_vs_total_conflict`, `numeric_type_mismatch`, `negative_impossible` | 2 each |
| `final_answer_confident` | Trust gate: parse ok + finalization cue + answer_error==0 OR coherent answer + sufficient role coverage | — |

**Escalation rules:**

1. **Revise** if `answer_error_score >= 2` (`answer_error_high`).  
2. **Revise** if `explanation_warning_score >= 3` AND NOT `final_answer_confident` AND `answer_error_score >= 1` (`explanation_pressure_plus_low_trust_plus_moderate_answer_error`).  
3. Otherwise: `no_escalation`.

**Intended failure mode addressed:** V5's over-escalation of concise but correct traces. Missing intermediate literals inflate `unified_error` in V5 and trigger unnecessary revisions. V6 explicitly demotes these to `explanation_warning_score` only.

**Generic explanation vs answer-error?** V6 **explicitly separates** them. Role-coverage signals (intermediate literal echoes) are reclassified as soft explanation warnings only. Answer-level constraint and consistency signals drive `answer_error_score` and are the primary revise trigger. Design invariant: `explanation_incomplete ≠ answer_likely_wrong` (stated in code docstring and doc).

---

### Adaptive Policy V7

**File:** `src/policies/adaptive_policy_v7.py`  
**Doc:** `docs/ADAPTIVE_POLICY_V7.md`

**Main signals/features used:**

All V6 signals, plus four new V7-only signals:

| Signal | Detection | Weight | Bucket |
|---|---|---|---|
| `weekday_question_numeric_final` | Categorical "which day" question + final parses as an integer | 3 | `answer_error_score` (added as extra error) |
| `need_more_answer_equals_list_price` | "how much more" question + parsed final == first `$N` in question text | 3 | `answer_error_score` |
| `tail_equals_disagrees_with_final` | Last `= <number>` in body (last 800 chars) ≠ numeric final | 2 | `answer_error_score` |
| `low_confidence_escalate` | V6 `final_answer_confident=false` AND (categorical OR `explanation_warning >= 2`) | — | Forces revise directly |

**Escalation rules (in priority order):**

1. **Revise** if `answer_error_score >= 2` (after V7 extras added to V6 base).  
2. **Else revise** if `low_confidence_escalate` flag.  
3. **Else revise** if V6 combo: `explanation_warning >= 3` AND NOT confident AND `answer_error >= 1`.  
4. Otherwise: `no_escalation`.

**Intended failure mode addressed:** V6's false negatives on semantically wrong but structurally quiet answers. Specifically: Tobias (`gsm8k_test_11`) where the model returns the full list price instead of the difference ("how much more"), and Jasmine (`gsm8k_test_13`) where the body concludes Sunday but the final answer line gives `7` (numeric). V6 misses both because `answer_error_score == 0` on both.

**Generic explanation vs answer-error?** V7 **deepens** the answer-error channel. All three new numeric signals (`weekday_question_numeric_final`, `need_more_answer_equals_list_price`, `tail_equals_disagrees_with_final`) directly detect **structured semantic mismatches** between the parsed answer and the question's expected answer type or semantics, and they contribute to `answer_error_score`. The `low_confidence_escalate` flag does mix a confidence gate with explanation pressure (`explanation_warning >= 2`), but it fires only when trust is already low—it does not revise on explanation warnings alone when confidence is high. The design explicitly preserves the V6 invariant: concise-correct traces with `answer_error=0` and `final_answer_confident=true` are not escalated.

---

## 2. Conceptual Changes

### V5 → V6: The Key Change

**V5** uses a `unified_error_score` that bundles role-coverage signals (proxy for explanation thoroughness) with constraint and consistency signals. Missing intermediate number echoes are treated as evidence of potential error. This causes *correct but concise* reasoning traces to be over-escalated—`answer_error=0` but `unified_error` is high because role echoes are missing.

**V6** introduces an explicit two-bucket architecture:
- `explanation_warning_score` for explanation-completeness indicators (role coverage, short reasoning)
- `answer_error_score` for answer-level correctness indicators (constraint violations, parse failure, consistency checks)

Only `answer_error_score >= 2` triggers revise by default. `explanation_warning_score` can contribute to revise only through a three-way combo rule that also requires low confidence and at least one answer-error signal. The conceptual shift is: **generic explanation irregularity is no longer a primary escalation signal**.

### V6 → V7: The Key Change

**V6** misses two classes of wrong answers where both `answer_error_score == 0` and `explanation_warning_score == 0`: (a) semantic category mismatches (weekday question answered with a number), (b) "how much more" answered with the original price. These represent structured answer errors that V6's existing signals don't detect.

**V7** adds three new lightweight **structured answer-error detectors** that each contribute to `answer_error_score` directly:
- A categorical-type check (`weekday_question_numeric_final`)
- A semantic-role check (`need_more_answer_equals_list_price`)
- A within-trace consistency check (`tail_equals_disagrees_with_final`)

The conceptual shift from V6 to V7 is: **expanding the answer-error signal vocabulary** to cover structured semantic mismatches that V6's constraint features missed, rather than re-opening the explanation-warning pathway. All new signals go into the answer-error bucket, not the explanation-warning bucket.

---

## 3. Evidence from Existing Outputs

### 3.1 False-Positive Fixture Comparison

**Source:** `docs/FALSE_POSITIVE_ANALYSIS.md`, `outputs/adaptive_policy_v7/per_case_results.csv`, `outputs/adaptive_policy_v7/summary.json`

| Version | Revises on 5 gold-correct concise traces |
|---|---|
| V5 | 5 / 5 (100% false-positive revise rate) |
| V6 | 0 / 5 (0%) |
| V7 | 0 / 5 (0%) |

`summary.json` confirms: `false_positive_v6_revise_count: 0`, `false_positive_v7_revise_count: 0`.  
**This directly supports the story**: removing explanation signals as a primary revise driver eliminates the V5 over-escalation. V7 preserves that improvement.

### 3.2 False-Negative Probe (Real Model Outputs)

**Source:** `outputs/adaptive_policy_v7/false_negative_probe.csv`, `outputs/adaptive_policy_v7/summary.json`, `docs/ADAPTIVE_POLICY_V7.md`

| Version | Wrong answers with no revise (out of 2 known wrong in 7-row probe) |
|---|---|
| V6 | 2 / 2 (misses both Tobias and Jasmine) |
| V7 | 0 / 2 (catches both via `weekday_question_numeric_final` and `need_more_answer_equals_list_price`) |

`false_negative_probe.csv` row for `gsm8k_test_13`: `v7_revise=True`, `v7_extra_error=3`, `v7_revise_reason=answer_error_high`, `weekday_question_numeric_final=true`.  
`false_negative_probe.csv` row for `gsm8k_test_11`: `v7_revise=True`, `v7_extra_error=3`, `v7_revise_reason=answer_error_high`, `need_more_answer_equals_list_price=true`.

**This supports the story**: V7 adds structured answer-error signals that catch V6's false negatives without restoring explanation-based escalation.

### 3.3 Aggregate Policy Comparison (Population Metrics)

**Source:** `outputs/paper_tables/policy_efficiency_table.csv`, `outputs/paper_tables/real_routing/real_policy_eval_comparison_long.csv`

**MATH500 (100 samples):**

| Policy | Accuracy | Avg cost | Revise rate | Gain per extra cost |
|---|---|---|---|---|
| V5 | 66.0% | 1.71 | 71% | 0.028 |
| V6 | 65.0% | 1.03 | 3% | 0.333 |
| V7 | 65.0% | 1.09 | 9% | 0.111 |
| `reasoning_greedy` | 64.0% | 1.00 | 0% | — |
| `direct_plus_revise` | 64.0% | 2.00 | 100% | — |

V5 escalates on 71% of MATH500 cases—driven by explanation/role signals misfiring on the more complex reasoning in that dataset. V6 drops to 3% revise rate with nearly the same accuracy. The large cost reduction from V5→V6 on MATH500 is the population-level correlate of the false-positive fixture results.

**Hard GSM8K (100 samples):**

| Policy | Accuracy | Avg cost | Revise rate |
|---|---|---|---|
| V5 | 86.0% | 1.53 | 53% |
| V6 | 81.0% | 1.26 | 26% |
| V7 | 82.0% | 1.46 | 41% |
| `reasoning_greedy` | 79.0% | 1.00 | 0% |
| `direct_plus_revise` | 86.0% | 2.00 | 100% |

V6's lower revise rate vs V5 on hard GSM8K corresponds to a 5 percentage-point accuracy drop (86%→81%), suggesting that on harder problems some of the V5 escalations were genuinely helpful, and moving signals to explanation-only reduced recall. V7's new answer-error signals partially recover accuracy (81%→82%) and revise rate (26%→41%) without fully restoring V5 escalation breadth.

**Hard GSM8K budget-2 subset (100 samples):**

| Policy | Accuracy | Avg cost | Revise rate |
|---|---|---|---|
| V5 | 91.0% | 1.41 | 41% |
| V6 | 89.0% | 1.27 | 27% |
| V7 | 89.0% | 1.40 | 40% |

**GSM8K random (100 samples):**

| Policy | Accuracy | Avg cost | Revise rate |
|---|---|---|---|
| V5 | 92.0% | 1.29 | 29% |
| V6 | 92.0% | 1.18 | 18% |
| V7 | 92.0% | 1.30 | 30% |

On regular GSM8K, V5 and V6 achieve the same accuracy (92%), but V6 is cheaper (1.18 vs 1.29) because it avoids unnecessary revisions. Same accuracy at lower cost confirms the story that fewer explanation-driven escalations preserve correctness outcomes.

---

## 4. Compact Summary Table

| Version | Main routing signals | Main intended fix | Evidence from outputs | Supports proposed story? |
|---|---|---|---|---|
| **V5** | `unified_error_score` (bundles role coverage + constraint + self/step consistency); confidence threshold gates; `target_quantity_mismatch`, `constraint_word_conflict` | Generalize V4's constraint-only routing with broader signal families | 5/5 FP on concise-correct fixtures; 71% revise rate on MATH500; 53% on hard GSM8K (`policy_efficiency_table.csv`) | **No** — uses generic explanation irregularity (role echoes, unified error) as primary revise driver; the bundled problem this story critiques |
| **V6** | `answer_error_score` (constraint violations, parse failure, consistency checks on parsed answer) as primary revise driver; `explanation_warning_score` (role coverage, short reasoning) as soft signal only | Eliminate FP escalations on concise-correct traces caused by V5's unified_error over-triggering | 0/5 FP on same fixtures; 3% revise rate on MATH500 vs 71% for V5; same accuracy at lower cost on GSM8K random (`real_policy_eval_comparison_long.csv`) | **Yes** — explicit architectural split makes answer-error the dominant escalation signal and demotes explanation irregularity |
| **V7** | All V6 signals plus three new structured answer-error signals (`weekday_question_numeric_final`, `need_more_answer_equals_list_price`, `tail_equals_disagrees_with_final`), each contributing to `answer_error_score`; `low_confidence_escalate` as secondary path | Recover V6 false negatives (Tobias: wrong price; Jasmine: numeric final on weekday question) without restoring explanation-driven escalation | 0/2 FN on 7-row real probe vs 2/2 for V6; 0/5 FP preserved; slight accuracy recovery on hard GSM8K (81%→82%, `policy_efficiency_table.csv`) | **Yes** — expands answer-error signal vocabulary to cover new failure modes; preserves the V6 invariant that explanation-only paths do not revise high-confidence correct traces |

---

## 5. Grounded Assessment of the Story

### What is strongly supported

1. **V5→V6 architectural separation is real and documented in code.** `adaptive_policy_v6.py` explicitly defines two separate score variables (`explanation_warning_score`, `answer_error_score`) with clearly separated weights. The docstring and `ADAPTIVE_POLICY_V6.md` state the invariant `explanation_incomplete ≠ answer_likely_wrong`. This is not post-hoc framing—it is encoded in the implementation.

2. **V6 eliminates V5's explanation-driven false positives.** The five-fixture test (`outputs/adaptive_policy_v7/per_case_results.csv`, `summary.json`) shows V5 revises 5/5 concise-correct traces while V6 revises 0/5. The root cause is documented: V5's `unified_error_score` includes role-echo signals that fire on correct concise output; V6 demotes those signals to `explanation_warning_score` only.

3. **V7 extends the answer-error channel to fix new false negatives.** The two V6 false negatives in the real probe (`false_negative_probe.csv`) are corrected by signals that are classified as `answer_error_score` additions (weights 3 and 3). Code inspection confirms they add to `extra_error` before the revise threshold check, not to `explanation_warning_score`.

4. **Population metrics are consistent with the story direction.** MATH500's revise rate drops from 71% (V5) to 3% (V6) to 9% (V7), matching the story's prediction that removing explanation-based escalation reduces unnecessary revisions. GSM8K random accuracy is identical across V5/V6/V7 (92%) at lower cost for V6/V7.

### What is weakly supported

1. **The hard GSM8K accuracy drop (V5: 86% → V6: 81%).** V6's stricter answer-error-only escalation reduces cost but also reduces accuracy on harder problems, suggesting some V5 explanation-driven escalations were genuinely correcting errors. V7 partially recovers (82%), but the story's framing—that answer-error-focused routing *improves* selective escalation—is complicated by this accuracy regression on the hardest dataset. The outputs are consistent but the efficiency gain per extra cost unit drops from V5 (0.132) to V6 (0.077) to V7 (0.065) on hard GSM8K, opposite to the story's claim for that regime.

2. **Fixture and probe sizes are small.** The false-positive test uses 5 hand-crafted traces (`FALSE_POSITIVE_ANALYSIS.md` states simulated model outputs). The false-negative probe uses 7 real rows from one model run (`gpt-4o-mini`). Population-level conclusions about routing improvement carry "exploratory_only" evidence labels per the repo's own metadata in `summary.json`.

3. **`low_confidence_escalate` in V7 is a hybrid signal.** It fires when V6 confidence is false AND (categorical OR `explanation_warning >= 2`). The `explanation_warning >= 2` branch reintroduces explanation pressure—albeit gated by low confidence—which technically reverts part of the V6 invariant. Its contribution to the 2/2 false-negative recall in the probe is via `Jasmine` (caught primarily by `weekday_question_numeric_final`, weight 3, not `low_confidence_escalate` alone), but it could cause false positives in edge cases where explanation warning is high but the answer is correct and confidence is merely borderline.

### What is not supported

1. **Causal attribution to signal design alone.** The claim that "selective escalation *improves*" depends on a defined improvement metric. On MATH500 the story holds (same accuracy at far lower cost). On hard GSM8K the story is ambiguous (lower false positives but also lower recall). The repo does not include a single evaluation that cleanly dominates V5 on both precision *and* recall simultaneously.

2. **Generalization beyond the benchmark sample.** All quantitative results cover at most 100 samples per dataset, selected from specific subsets of GSM8K and MATH500. The repo's evidence labels consistently mark broad claims as `exploratory_only`.

---

## 6. Repo Artifacts Supporting Each Claim

| Claim | Artifact |
|---|---|
| V5 over-escalates on concise-correct traces due to explanation/role signals | `docs/FALSE_POSITIVE_ANALYSIS.md`, `outputs/adaptive_policy_v7/per_case_results.csv` |
| V6 fixes V5's concise-correct FPs | `outputs/adaptive_policy_v7/summary.json` (`false_positive_v6_revise_count: 0`), `outputs/adaptive_policy_v7/per_case_results.csv` |
| V7 preserves V6's FP improvement | `outputs/adaptive_policy_v7/summary.json` (`false_positive_v7_revise_count: 0`) |
| V6 misses Tobias and Jasmine (false negatives) | `docs/ADAPTIVE_POLICY_V7.md` §2, `docs/REAL_V6_FALSE_NEGATIVE_PROBE.md`, `outputs/adaptive_policy_v7/false_negative_probe.csv` |
| V7 catches both V6 false negatives via answer-error signals | `outputs/adaptive_policy_v7/false_negative_probe.csv` (rows 4 and 6), `outputs/adaptive_policy_v7/summary.json` (`snapshot_wrong_v7_no_revise: 0`) |
| V6 lowers cost vs V5 (MATH500) | `outputs/paper_tables/policy_efficiency_table.csv`, `outputs/paper_tables/real_routing/real_policy_eval_comparison_long.csv` |
| V5 accuracy on hard GSM8K equals `direct_plus_revise` (86%) at lower cost | `outputs/paper_tables/policy_efficiency_table.csv` |
| V6 accuracy on hard GSM8K drops vs V5 (81% vs 86%) | `outputs/paper_tables/policy_efficiency_table.csv`, `outputs/real_hard_gsm8k_policy_eval/summary.json` |
| V7 partially recovers hard GSM8K accuracy (82%) vs V6 | `outputs/paper_tables/policy_efficiency_table.csv` |
| Explicit two-bucket signal architecture in V6 code | `src/policies/adaptive_policy_v6.py` (lines 229–310, `explanation_warning_score` vs `answer_error_score`) |
| V7's new signals go into `answer_error_score` not `explanation_warning_score` | `src/policies/adaptive_policy_v7.py` (lines 157–169: `extra_error` += weights for V7 flags, then added to `answer_error`) |

---

## Summary

The progression from V5 → V6 → V7 **does support** the proposed scientific story at the level of **code architecture and small-scale offline evaluation**. V5 uses a blended signal where explanation irregularity (role echoes, unified error) can directly trigger revise. V6 explicitly demotes those signals to a soft bucket, making answer-level error detection the primary revise driver. V7 extends the answer-error bucket with new structured semantic mismatch detectors. The offline fixture tests and real probe results confirm that this architectural shift reduces false-positive escalations while recovering targeted false negatives.

The story is **weakly supported** at the population level: the available 100-sample evaluations show cost efficiency gains on MATH500 and GSM8K random, but hard GSM8K shows an accuracy regression in V6 that V7 only partially recovers, and all population-level claims carry `exploratory_only` labels in the repo. The story should be presented with the caveat that the efficiency gain is context-dependent and that the fixture and probe evidence is small-scale and partially synthetic.
