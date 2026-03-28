# Feature Gap Analysis — Revise-Helps Cases

> **Purpose:** Identify what our current cheap features fail to capture for
> queries where the `direct_plus_revise` strategy helps but the existing
> adaptive policies (v3/v4) did not trigger it.

---

## 1. Input Files Used

| File | Required? | Description |
|------|-----------|-------------|
| `outputs/oracle_subset_eval/oracle_assignments.csv` | Primary | Per-query oracle results: `cheapest_correct_strategy`, `direct_greedy_correct`, `direct_already_optimal` |
| `outputs/oracle_subset_eval/per_query_matrix.csv` | Primary | Per-query × per-strategy results: `question_id`, `strategy`, `correct`, `question_text` |
| `outputs/revise_case_analysis/case_table.csv` | Optional | Existing manual case categories; overrides auto-assigned groups when present |
| `outputs/revise_case_analysis/category_summary.csv` | Optional | Aggregate category summary (loaded but not required for analysis) |
| `outputs/adaptive_policy_v3/per_query_results.csv` | Optional | Policy v3 per-query chosen strategy |
| `outputs/adaptive_policy_v4/per_query_results.csv` | Optional | Policy v4 per-query chosen strategy |

All inputs are optional — missing files are skipped gracefully and produce
empty groups.  The analysis is most useful when oracle outputs are available.

---

## 2. Group Definitions

Three mutually exclusive groups are defined for each query:

| Group | Definition |
|-------|-----------|
| `revise_helps` | `direct_greedy` was **wrong**; `direct_plus_revise` was **correct** |
| `reasoning_enough` | `direct_greedy` was already **correct**, OR some non-revise strategy was cheapest-correct while revise result is unknown |
| `revise_not_enough` | No strategy was correct (oracle failed), OR `direct_plus_revise` was tried and still **wrong** |

**Assignment logic** (in `src/analysis/feature_gap_analysis.assign_group`):
1. If `direct_greedy_correct == 1` → `reasoning_enough`
2. Else if `direct_plus_revise correct == 1` → `revise_helps`
3. Else if `oracle_any_correct == 0` → `revise_not_enough`
4. Else if `direct_plus_revise correct == 0` → `revise_not_enough`
5. Else (revise result unknown, but something else was correct) → `reasoning_enough`

---

## 3. Where Current Features Succeed

The existing cheap feature set (`src/features/precompute_features.py`) works
well for:

- **Identifying question complexity** via `question_length_chars`,
  `question_length_tokens_approx`, and `num_sentences_approx`.
- **Detecting numeric density** via `num_numeric_mentions`,
  `max_numeric_value_approx`, `numeric_range_approx`.
- **Flagging broad multi-step structure** via `has_multi_step_cue` (which
  fires on keywords like _total_, _remaining_, _after_, _left_, etc.).
- **Signalling currency/percentage context** via `has_currency_symbol` and
  `has_percent_symbol`.

These features are sufficient for `reasoning_enough` queries, where
`direct_greedy` already produces the correct answer and the decision "no
escalation needed" is straightforward.

---

## 4. Where Current Features Fail

### 4.1 `has_multi_step_cue` is too coarse

The single boolean `has_multi_step_cue` fires on any of 15+ keywords
(total, remaining, after, left, difference, …).  It cannot distinguish:
- **Subtraction-final** problems (the answer requires computing a remainder
  after spending / selling / losing) — where `direct_greedy` tends to stop
  at the intermediate total.
- **Additive accumulation** problems (the answer is a sum over several items)
  — where `direct_greedy` usually succeeds.

Both problem types fire `has_multi_step_cue`, but only subtraction-final
problems systematically benefit from revision.

### 4.2 No target-quantity signal

Current features carry no information about _what the question is asking for_:
- "How many are **remaining**?" → answer is a remainder
- "How many did she **earn in total**?" → answer is a sum
- "How much does each cost **per** day?" → answer requires rate × time

These surface-form signals are absent from the feature vector entirely.

### 4.3 Wording-trap features are unimplemented

Five lightweight wording-trap signals are **not present** in the current
feature set, yet they directly correspond to the most common revise-help
patterns identified in this analysis:

| Missing feature | Pattern targeted |
|-----------------|-----------------|
| `has_remaining_left_cue` | "remaining / left / left over" → subtraction final step |
| `has_subtraction_trap_verb` | "spent / sold / gave away / lost" → hidden two-step structure |
| `has_total_earned_cue` | "total / altogether / earned / made" → can be intermediate or final |
| `has_unit_per_cue` | "per / each / every" → rate × quantity multiply step |
| `has_intermediate_quantity_ask` | "how many does/did" → multiple plausible answer positions |

### 4.4 No answer-echo detection

A strong signal for "the model returned a question value, not a computed
answer" — checking whether the parsed first-pass answer is a verbatim numeric
token in the question — is not implemented.

### 4.5 No semantic depth

All current features are purely syntactic (regex / character counts).  They
carry no information about the number of arithmetic steps, the dependency
chain between quantities, or whether the solution requires tracking which
quantity is the final target.

---

## 5. Candidate Next-Step Signal Ideas

The following five lightweight signals are grounded in the patterns above and
can be implemented as pure regex over the question string (no model call
needed):

### Signal 1 — `has_remainder_ask`
- **What it captures:** True when the question explicitly asks for what
  "remains", "is left", or "is still needed".
- **Implementation:** Regex match on `remaining|left over|left|have left`.
- **Why it helps:** Directly identifies the most common subtraction-final
  problem form that direct_greedy gets wrong.

### Signal 2 — `subtraction_verb_count`
- **What it captures:** Integer count of spend/gave/lost/sold verbs.
- **Implementation:** Regex count of the subtraction verb class.
- **Why it helps:** A higher count correlates with hidden multi-step
  subtraction chains that direct_greedy short-circuits.

### Signal 3 — `per_unit_rate_flag`
- **What it captures:** True when "per", "each", or "every" appears alongside
  a numeric token within a short window.
- **Implementation:** Regex proximity check: `\d+\s*(?:per|each|every)`.
- **Why it helps:** Catches rate×quantity multiplication steps that the direct
  pass may skip, producing the rate instead of the total.

### Signal 4 — `first_pass_answer_in_question_flag`
- **What it captures:** True when the parsed first-pass answer appears
  verbatim as a numeric token in the question text.
- **Implementation:** Check `extracted_answer ∈ numeric_tokens(question_text)`.
- **Why it helps:** A strong oracle-free signal that the model echoed a given
  value instead of computing the target quantity.

### Signal 5 — `sentence_count_vs_numeric_token_ratio`
- **What it captures:** `num_sentences / max(num_numeric_mentions, 1)`.
- **Implementation:** Derived from existing `num_sentences_approx` and
  `num_numeric_mentions`.
- **Why it helps:** A high ratio (many sentences, few numbers) often
  indicates a narrative problem requiring careful tracking of which
  quantity is the final answer.

---

## 6. Conclusion: Do We Need Better Hand-Crafted Signals, Learned Routing, or a Stronger Verifier?

The answer depends on how many `revise_helps` cases are **missed** by the
current v3/v4 policies after oracle outputs become available.

| Coverage of existing policy | Recommended next step |
|----------------------------|-----------------------|
| < 50 % of revise_helps caught | **Better hand-crafted signals** — add the 5 candidate signals above; they are cheap (regex, no model call) and directly address the dominant failure patterns. |
| 50–90 % caught | **Stronger verifier** — use `first_pass_answer_in_question_flag` as an additional escalation trigger; no retraining needed. |
| > 90 % caught | **Learned routing** — current recall is good; the bottleneck is precision (false positive escalations). Train a lightweight logistic regression or decision tree on `routing_dataset.csv` to reduce unnecessary revise calls. |

Until oracle outputs are available the analysis runs on empty groups and
reports "no data". Re-run after `run_oracle_subset_eval.py` completes
to get meaningful numbers.

---

## 7. How to Reproduce

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the analysis (all inputs optional — missing files are skipped)
python3 scripts/run_feature_gap_analysis.py

# Run unit tests
pytest tests/test_feature_gap_analysis.py -v
```

Output files are written to `outputs/feature_gap_analysis/`:
- `group_feature_summary.csv` — mean/rate of each feature per group
- `missed_revise_cases.csv` — queries where revise would help but v3/v4 missed
- `pattern_notes.json` — structured qualitative pattern notes

---

*Last updated: 2026-03-28 — analysis module at `src/analysis/feature_gap_analysis.py`*
