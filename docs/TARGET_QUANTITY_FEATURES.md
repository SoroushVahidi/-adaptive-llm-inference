# Target-Quantity and Wording-Trap Features

> **Purpose:** Capture the *type of quantity* a math word problem asks for
> and surface common wording traps that cause a direct-greedy pass to return
> the wrong answer.  These are the last hand-crafted features before moving
> to learned routing.

---

## 1. Overview

`src/features/target_quantity_features.py` provides a single public function:

```python
from src.features import extract_target_quantity_features

feats = extract_target_quantity_features(question_text)  # → dict[str, bool]
```

All 11 features are **boolean** and derived from pure regex/string operations.
No model calls are needed.  The output dictionary has no key overlap with
`extract_query_features`, so the two can be merged safely:

```python
from src.features import extract_query_features, extract_target_quantity_features

merged = {**extract_query_features(q), **extract_target_quantity_features(q)}
```

---

## 2. Full Feature List and Keyword Rules

### A — Target-type cues

| Feature | Type | Keyword / rule |
|---------|------|---------------|
| `asks_remaining_or_left` | bool | `remaining`, `left over`, `left`, `have left`, `are left` |
| `asks_total` | bool | `total`, `altogether`, `in all`, `combined`, `grand total` |
| `asks_difference` | bool | `difference`, `how much more`, `how many more`, `more than`, `less than`, `fewer than`, `how much less` |
| `asks_rate_or_unit` | bool | `per`, `each`, `every`, `apiece`, `a piece`, `per day/week/hour/month/year/minute`, `rate` |
| `asks_money` | bool | `$`, `€`, `£`, `¥`, `₹`, `dollar(s)`, `cent(s)`, `euro(s)`, `pound(s)`, `rupee(s)`, `yen` |
| `asks_time` | bool | `minute(s)`, `hour(s)`, `day(s)`, `week(s)`, `month(s)`, `year(s)`, `second(s)` |

### B — Wording-trap signals

| Feature | Type | Keyword / rule |
|---------|------|---------------|
| `has_subtraction_trap_verb` | bool | `spent`, `lost`, `gave away`, `sold`, `used`, `ate`, `eaten`, `consumed`, `donated`, `gave`, `given` |
| `has_addition_trap_structure` | bool | `also`, `then`, `together`, `as well`, `additionally`, `on top of`, `plus`, `added`, `add`, `combined with` |
| `has_multi_operation_hint` | bool | ≥ 2 distinct operation verbs from a 30+ verb list (bought, sold, earned, spent, received, gave, took, found, saved, paid, …) |

### C — Answer-risk signals

| Feature | Type | Rule |
|---------|------|------|
| `likely_intermediate_quantity_ask` | bool | ≥ 3 sentences **AND** ≥ 3 numeric tokens **AND** no `total`/`remaining` anchor present |
| `potential_answer_echo_risk` | bool | ≥ 4 numeric tokens **AND** final sentence ≤ 12 tokens |

---

## 3. Examples

### Example 1 — "remaining" problem

**Question:**
> "Janet had 20 apples. She gave 5 to her friend and ate 3 herself.
>  How many apples does she have left?"

**Features fired:**

| Feature | Value | Reason |
|---------|-------|--------|
| `asks_remaining_or_left` | `True` | "left" in final sentence |
| `has_subtraction_trap_verb` | `True` | "gave" |
| `has_multi_operation_hint` | `True` | "gave" + "ate" = 2 operation verbs |
| `asks_total` | `False` | — |
| `likely_intermediate_quantity_ask` | `False` | "left" anchor suppresses it |

**Why this helps detect revise-help cases:**
`asks_remaining_or_left=True` + `has_subtraction_trap_verb=True` is the
strongest composite signal that a direct-greedy pass may return the pre-
subtraction total rather than the remainder.  Revision explicitly re-asks
for the final quantity, which is why `direct_plus_revise` fixes these cases.

---

### Example 2 — "total" problem

**Question:**
> "Maria earns $15 per hour. She works 8 hours a day for 5 days.
>  What are her total earnings?"

**Features fired:**

| Feature | Value | Reason |
|---------|-------|--------|
| `asks_total` | `True` | "total" |
| `asks_rate_or_unit` | `True` | "per" |
| `asks_money` | `True` | "$" |
| `asks_time` | `True` | "hours", "days" |
| `has_multi_operation_hint` | `True` | "earns" + "works" = 2 verbs |

**Why this helps:**
`asks_rate_or_unit=True` + `asks_total=True` signals a rate×time
multiplication problem.  If direct greedy returns the hourly rate or the
daily total instead of the 5-day total, revision forces reconsideration of
which product is the final answer.

---

### Example 3 — "rate" problem

**Question:**
> "A machine produces 120 widgets in 4 hours.
>  How many widgets does it produce per hour?"

**Features fired:**

| Feature | Value | Reason |
|---------|-------|--------|
| `asks_rate_or_unit` | `True` | "per" |
| `asks_time` | `True` | "hours" |
| `asks_remaining_or_left` | `False` | — |
| `asks_total` | `False` | — |
| `has_subtraction_trap_verb` | `False` | — |

**Why this helps:**
`asks_rate_or_unit=True` with no `asks_total` or `asks_remaining_or_left`
flags a pure division / rate problem.  Direct greedy sometimes returns the
total count (120) rather than the per-hour rate (30), particularly when
the total is the only prominent number.  The rate signal can trigger a
targeted revision prompt.

---

## 4. Why These Features Help Detect Revise-Help Cases

The feature gap analysis (`docs/FEATURE_GAP_ANALYSIS.md`) showed that the
existing `has_multi_step_cue` boolean is too coarse: it fires on both
subtraction-final and additive-accumulation problems, which require
*different* strategies.

These 11 features add precision by:

1. **Naming the target type.**  `asks_remaining_or_left` isolates the most
   common revise-help pattern (subtraction-final problems).  `asks_total`
   separates accumulation problems where direct is usually fine.

2. **Flagging dangerous verb patterns.**  `has_subtraction_trap_verb`
   directly identifies the class of verbs that cause a direct pass to stop
   at an intermediate total.

3. **Detecting multi-step chains.**  `has_multi_operation_hint` catches
   chained-operation problems without relying on keyword co-occurrence.

4. **Raising echo risk.**  `potential_answer_echo_risk` flags dense-number
   problems where the model may copy a given value — a subtle but common
   source of incorrect direct answers.

Together with the base features from `extract_query_features`, these signals
form a richer feature vector that a lightweight router (logistic regression,
decision tree) can use to decide when to escalate to `direct_plus_revise`.

---

## 5. Integration

```python
# Standalone
from src.features.target_quantity_features import extract_target_quantity_features
feats = extract_target_quantity_features(question)

# Via package
from src.features import extract_target_quantity_features
feats = extract_target_quantity_features(question)

# Merged with base features
from src.features import extract_query_features, extract_target_quantity_features
merged = {**extract_query_features(q), **extract_target_quantity_features(q)}
```

---

## 6. Limitations

- All features are **lexical** — they fire on surface-form keywords, not
  semantic meaning.  A problem phrased as "what quantity remains?" will
  trigger `asks_remaining_or_left`; a problem that implies subtraction
  without these words will not.
- `likely_intermediate_quantity_ask` and `potential_answer_echo_risk` are
  **heuristics** with fixed thresholds (≥ 3 sentences, ≥ 3 numbers, etc.).
  These thresholds were set for GSM8K-style 2–5 sentence problems and may
  need tuning for other datasets.
- `has_multi_operation_hint` deduplicates verb forms so "gave" and "give"
  count as one verb; however morphological variants not in the list will
  not be counted.
- No feature captures **negation** (e.g. "did *not* sell").

---

*Module:* `src/features/target_quantity_features.py`  
*Tests:* `tests/test_target_quantity_features.py`  
*Diagnostic:* `scripts/inspect_target_features.py`
