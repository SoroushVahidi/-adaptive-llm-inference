# Adaptive Policy V7

## Evidence labels

| Claim | Label |
|-------|--------|
| False-positive fixture table (5 concise-correct traces) | **measured_now** (regenerate: `python3 scripts/run_adaptive_policy_v7_eval.py`) |
| Real probe snapshot (7 rows, committed JSONL) | **measured_now** for scoring; **exploratory_only** for routing quality at scale |
| Broad “V7 beats V6 on GSM8K” | **exploratory_only** until larger runs |

---

## 1. What V6 fixed

Concise **correct** traces no longer get **`direct_plus_revise`** just because role coverage flags missing question literals — **explanation_warning** stays weak for revise; **answer_error** drives escalation.

## 2. What V6 missed (real probe)

Bundled snapshot `src/datasets/bundled/real_v6_false_negative_probe_snapshot.jsonl` (from `scripts/run_real_v6_false_negative_probe.py`):

- **Tobias (`gsm8k_test_11`):** Wrong answer **95** (full shoe price) on “how much **more**”; **answer_error 0**, **confident true**.
- **Jasmine (`gsm8k_test_13`):** Body concludes **Sunday**, **Final answer: 7**; **confident false**, **answer_error 0**, **explanation_warning 0** — V6 combo rule never fired.

## 3. What V7 changes

V7 **starts from `compute_v6_scores`** then adds **V7-only flags** that increment **`answer_error_score`** and/or force **revise** when confidence is low.

### New signals (lightweight)

| Signal | Detection sketch |
|--------|------------------|
| **`weekday_question_numeric_final`** | Categorical “which day” question + final parses as **integer** (Jasmine-style). |
| **`need_more_answer_equals_list_price`** | Question asks “how much **more**” + parsed final equals **first `$N`** in question (Tobias-style misuse of list price). |
| **`tail_equals_disagrees_with_final`** | Last `= <number>` in body (last 800 chars) ≠ numeric final (numeric targets only). |
| **`low_confidence_escalate`** | V6 **final_answer_confident** false **and** (categorical **or** explanation_warning ≥ 2). |

### Weights (defaults, on top of V6)

- `weight_weekday_question_numeric_final` = **3**
- `weight_need_more_equals_list_price` = **3**
- `weight_tail_equals_disagrees` = **2**

Threshold **`answer_error_revise_threshold`** stays **2** (inherited from V6).

### Revise rules (exact)

1. **Revise** if `answer_error_score >= 2` (after V7 extras).
2. **Else revise** if `low_confidence_escalate` (V7 flag from V6 trust + categorical / explanation pressure).
3. **Else revise** if V6 combo: `explanation_warning >= explanation_warn_high` **and** not confident **and** `answer_error >= answer_error_moderate_for_combo`.

**Not revised:** explanation warnings **alone** when answer_error stays 0 and confidence stays high (concise-correct path preserved).

## 4. Did V7 preserve concise-correct trust?

**On the 5 `FALSE_POSITIVE_ANALYSIS` fixtures:** **yes** — `false_positive_v7_revise_count = 0` in offline eval (same as V6 on those traces; V5 still revises 5/5).

## 5. Did V7 improve the known false negatives?

**On the committed real probe snapshot (7 rows, 2 wrong):**

- V6: **2** wrong with **no** revise.
- V7: **0** wrong with no revise (`snapshot_wrong_v7_no_revise = 0`).

Both **Tobias** and **Jasmine** rows **revise** under V7 on that snapshot.

## 6. Limitations

- **Template heuristics:** “need more = first dollar” can misfire if question order differs; **weekday + numeric final** is narrow.
- **Tail `=` check** may false-positive if body has unrelated equals near the end (conservative tail window 800 chars).
- Snapshot is **one model run** (`gpt-4o-mini`); refresh JSONL to re-score after new API runs.

## Related files

- `src/policies/adaptive_policy_v7.py`
- `src/evaluation/adaptive_policy_v7_eval.py`
- `scripts/run_adaptive_policy_v7_eval.py`
- `configs/adaptive_policy_v7_offline.yaml`
- `src/datasets/bundled/real_v6_false_negative_probe_snapshot.jsonl`
