# False Positive Analysis — Correct Answers Flagged as Suspicious

**Date:** 2026-03-29  
**Method:** Offline only — no APIs or model calls.  
**Data:** Real problem text and gold answers from `src/datasets/bundled/gsm8k_test_sample.json`.  
**Simulated outputs:** Short, **correct** reasoning traces were hand-written to mimic a model that states the right answer with **minimal intermediate exposition** (a common `reasoning_greedy`-style success path). All feature values were computed with the current code (`compute_unified_error_signal`, `extract_constraint_violation_features`, `compute_calibrated_role_decision`, `extract_target_quantity_features`).

**Honest summary:** On these five real questions, the **unified error score is high (0.39–0.57)** and **multiple subsystems fire** even when the parsed answer matches gold. The stack **over-triggers** on **compressed correct reasoning** and on **non-numeric answers**, relative to its own “suspicion” semantics.

---

## Summary table (5 cases)

| ID | Topic | Gold | Unified error | Strongest escalation | Main fired families |
|----|-------|------|-----------------|----------------------|---------------------|
| `gsm8k_test_8` | Budget / receipt (shoes price) | 41 | **0.368** | `strong_escalation_candidate` | Role coverage + target wording-trap flags |
| `gsm8k_test_11` | Money still needed for shoes | 65 | **0.400** | `strong_escalation_candidate` | Role coverage + target wording-trap flags |
| `gsm8k_test_18` | Apples left in basket | 60 | **0.400** | `strong_escalation_candidate` | Role coverage + target wording-trap flags |
| `gsm8k_test_0` | Clips April + May | 72 | **0.388** | `strong_escalation_candidate` | Role coverage (`sold` → subtract role) |
| `gsm8k_test_13` | Day first >100 paperclips | Sunday | **0.566** | (role noise; constraint fires) | **answer_type_mismatch** + role + self/calibration |

**Dataset caveat:** `gsm8k_test_3` in the same bundled JSON (Julie’s book) **strongly fires** `target_quantity_mismatch_suspected` on minimal correct-form traces, but its bundled gold `54` is **arithmetically inconsistent** with the usual reading (half of remaining after 12 + 24 pages read would be **42**, not 54). That item is omitted here to avoid mixing **signal false positives** with **label confusion**.

---

## Case 1 — `gsm8k_test_8` (Alexis’s shopping budget)

**A. Problem text**  
Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?

**B. Gold answer**  
`41`

**C. Correct reasoning (short)**  
Total budget 200; known purchases plus remaining: 30 + 46 + 38 + 11 + 18 + 16 = 159; shoes = 200 − 159 = **41**.

**D. Simulated model output (minimal but correct)**  
Use a line that avoids the substring **`so`** before `Final answer` (because `FINAL_STATEMENT_RE` matches `so` inside words like “clothes” and truncates the captured tail — an implementation sharp edge). Example:

```
Itemized spending leaves 41 for shoes.
Final answer: 41
```

**E. Feature breakdown (computed)**

| Family | Key outputs |
|--------|-------------|
| `target_quantity_features` | `asks_remaining_or_left`, `asks_money`, `has_subtraction_trap_verb` (“spent”), `has_addition_trap_structure`, `has_multi_operation_hint`, `potential_answer_echo_risk` |
| `constraint_violation_features` | **No** flags in `triggered_constraint_signals` (with the trace above) |
| `number_role_features` / calibrated role | **Triggers:** `missing_required_number`, `required_subtractive_number_missing`, `possible_intermediate_stop_suspected` → `strong_escalation_candidate`, `role_strong_error_score: 5`, `escalation_recommended: True` |
| `unified_error_signal` | `unified_error_score ≈ 0.368`, `unified_confidence_score ≈ 0.715`; `role_error: 1.0`, `self_error ≈ 0.1`, `step_error: 0` |

**F. Why the system flags it**

- **Role coverage:** Many dollar amounts are tied to **spent** / **subtract** roles and marked **strongly required**; the two-line trace does not echo **30, 46, 38, 11, 18, 16, 200** → **missing_required_number** and **possible_intermediate_stop_suspected**.
- **Target cues:** “**left** from her budget” turns on **asks_remaining_or_left** and **subtraction_trap** vocabulary even though the target is “price of shoes,” not a simple remainder phrase in the answer line.

**G. Why it is actually correct**

- **41** satisfies 200 − (30+46+38+11+18+shoes) = 16 ⇒ shoes = **41**; matches gold.

**H. Root cause**

- **Literal-echo-as-proof:** Same pattern as other cases — correct **compressed** solution without re-listing every receipt line.
- **Regex sharp edge:** If the reasoning contains **`so`** inside ordinary words, **final-statement extraction** can truncate and **inflate** other signals; this is a separate **implementation** footgun on top of the conceptual false positive.

**I. Conceptual fix**

- **Receipt problems:** If the answer **closes the budget** (sum of known + unknown + remaining = budget), treat as **consistent** and **suppress** subtractive “missing literal” penalties.
- **Final-statement regex:** Prefer word-boundary `\\bso\\b` (conceptually) so “clothes” does not start the “final statement” at `so`.

---

## Case 2 — `gsm8k_test_11` (Tobias’s shoes)

**A. Problem text**  
Tobias is buying a new pair of shoes that costs $95. He has been saving up his allowance for several weeks. He gets a $5 allowance per week. He has already spent $15 out of his savings. If he has been saving for 3 weeks, how much more money does he need to buy the shoes?

**B. Gold answer**  
`65`

**D. Simulated model output**  
```
Computed savings and need.
Final answer: 65
```

**E. Feature breakdown**

- **Target quantity:** `asks_difference`, `asks_money`, `asks_rate_or_unit`, `asks_time`, `has_subtraction_trap_verb` (“spent”), `has_multi_operation_hint`, `likely_intermediate_quantity_ask` — many **true** (surface cues).
- **Constraint:** No constraint flags in `triggered_constraint_signals`.
- **Role:** `missing_required_number`, `required_subtractive_number_missing`, `required_additive_number_missing`, `possible_intermediate_stop_suspected` → **`strong_escalation_candidate`**, `role_strong_error_score: 5`.
- **Unified:** `unified_error_score ≈ 0.400`, `role_error: 1.0`, `target_error: 0.15`, `step_error: 0.5`.

**F. Why flagged**  
Multi-step money story → many numbers marked **required**; **“spent”** triggers **subtract** role; short reasoning omits literals **95, 5, 15, 3** → mass **missing** + **possible_intermediate_stop**.

**G. Why correct**  
Answer **65** matches gold; the model need not repeat every bill and week count if the computation is right.

**H. Root cause**  
**Over-counting “required” mentions** for rich money problems + **“spent” → subtract** without requiring evidence that the *final* answer is an intermediate subtotal.

**I. Conceptual fix**  
**Discount role-coverage penalties** when **numeric answer matches** a **feasibility envelope** derived from question (e.g. need ∈ [0, 95]) and **finalization cue** is present.

---

## Case 3 — `gsm8k_test_18` (Apples in basket)

**A. Problem text**  
At the beginning of the day there were 74 apples in a basket. During the day, 17 more apples were added to the basket and 31 apples were removed. How many apples are in the basket at the end of the day?

**B. Gold answer**  
`60`

**D. Simulated model output (minimal — intentional stress test)**  
```
Worked it out.
Final answer: 60
```

**E. Feature breakdown**

- **Target:** `asks_time` (“end of the day”), `has_addition_trap_structure`, `likely_intermediate_quantity_ask`.
- **Role:** Same pattern as Case 2 — **missing** 74/17/31, subtract/add roles, **`strong_escalation_candidate`**, `unified_error ≈ 0.400`.

**F. Why flagged**  
No digits in reasoning → **role coverage collapse**; “added/removed” vocabulary creates **required** roles for multiple literals.

**G. Why correct**  
Gold **60**; answer **60**.

**H. Root cause**  
**Literal-echo = proof** assumption is false for **correct mental math** or **truncated logs**.

**I. Conceptual fix**  
Treat **high unified_confidence** + **correct parsed number** + **simple additive structure** as a **suppressor** for role-based escalation (calibrated “trust compression”).

**Important caveat (same item, different trace):** If the reasoning is `End count = 74 + 17 - 31 = 60.\nFinal answer: 60`, **role triggers empty**, `unified_error` drops to **~0.094**, `calibrated_decision: no_escalation`. So the false positive is **not the question** — it is **brittleness to reasoning length**.

---

## Case 4 — `gsm8k_test_0` (Natalia’s clips)

**A. Problem text**  
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

**B. Gold answer**  
`72`

**D. Simulated model output**  
```
Worked it out.
Final answer: 72
```

**E. Feature breakdown**

- **Target:** `asks_total`, `has_subtraction_trap_verb` (“sold”), `has_addition_trap_structure`, `has_multi_operation_hint`.
- **Role:** `missing_required_number`, `required_subtractive_number_missing`, `possible_intermediate_stop_suspected` → **`strong_escalation_candidate`**, `unified_error ≈ 0.388`.
- **Constraint:** none triggered.

**F. Why flagged**  
**“sold”** is in `ROLE_CFG.sub_words` → numbers near it get **subtract** role and **strong_required**; **48** and **half** may be required but **not echoed** in minimal text → **required_subtractive_number_missing** + **intermediate stop**.

**G. Why correct**  
48 + 24 = 72; answer **72** matches gold.

**H. Root cause**  
**Verb lexicon false positive:** “sold” signals **subtraction** in the role engine even when the problem is **summing two months’ sales**, not subtracting inventory.

**I. Conceptual fix**  
**Contextualize verbs** with question **target relation** (`contributes_to_total` here) — **do not treat “sold” as subtract** when the global ask is **total/altogether**.

---

## Case 5 — `gsm8k_test_13` (Paperclips — day of week)

**A. Problem text**  
Jasmine had 3 paperclips on Monday, then she had 6 on Tuesday, and her number of paperclips proceeded to double on each subsequent day. On what day of the week did she first have more than 100 paperclips?

**B. Gold answer**  
`Sunday` (non-numeric)

**D. Simulated model output**  
```
Doubling each day passes 100 on Sunday.
Final answer: Sunday
```

**E. Feature breakdown**

- **Constraint:** **`answer_type_mismatch_suspected`** — `asks_when` is true and any **parsed_answer** triggers mismatch in `extract_constraint_violation_features` (numeric *or not*).
- **Role:** `missing_required_number`, `required_rate_number_missing`; `role_strong_error_score: 2` (below v4/v5 strong threshold alone but still noisy).
- **Unified:** **`unified_error_score ≈ 0.566`** (highest of the five) — `constraint_error`, `self_error`, `calibration_uncertainty`, `step_error` all elevated; `unified_confidence_score ≈ 0.41`.

**F. Why flagged**  
Pipeline is **numeric-first**; **weekday string** answers look like type errors. Self-verification and calibration **penalize** non-numeric “final” formats.

**G. Why correct**  
Gold is **Sunday**; output states **Sunday**.

**H. Root cause**  
**Answer-type ontology** excludes **valid categorical math answers** (day names, ordering labels).

**I. Conceptual fix**  
**Branch `asks_when` / ordinal questions** to a **non-numeric consistency channel** (allow list of weekdays, match against last capitalized token, etc.) instead of treating as numeric mismatch.

---

## Cross-case synthesis

### 2. Most common failure pattern

**Correct but compressed reasoning** → **role coverage** declares **required literals missing** and often **`possible_intermediate_stop_suspected`**, which **dominates** `unified_error` via **`role_error = 1.0`**. Where the question has delicate **target language** (half of remaining) or **non-numeric answers**, **constraint** and **calibration/self** layers **add** false positives.

### 3. Which feature family is most responsible

**Primary:** `number_role_features` (**role coverage** + **calibrated_role_decision**) — fires **strong escalation** on **all five** traces.  
**Secondary:** `constraint_violation_features` for **answer_type** (Case 5). *(On `gsm8k_test_3`, **target_quantity_mismatch_suspected** also fires heavily; we omitted that row from the five-case set because bundled gold conflicts with standard arithmetic.)*  
**Tertiary:** `unified_error_signal` **fixed convex combination** — once `role_error` saturates, the **unified score stays high** even when the answer is gold-correct.

### 4. Implication for the current bottleneck

This reinforces the **marginal-value / estimand** issue: signals measure **“does the trace look like my template for a full solution?”** not **“is the next compute step likely to change correctness?”** High **false positive rate on cheap correct traces** means **routing will burn budget** or **train learners on wrong labels** unless **outcome-aligned gating** is added.

### 5. Three concrete ideas to reduce false positives without killing recall

1. **Literal-echo suppressor:** If `parsed_answer` matches gold or passes **bound / parity checks** from the question, **cap** `role_strong_error_score` (do not let “missing mention of 120” alone trigger max escalation).

2. **Verb–target coupling:** When `_global_target_relation` is **`contributes_to_total`**, **downgrade** “sold/spent” **subtract** roles unless the question explicitly asks **remaining/left**.

3. **Split answer modalities:** For `asks_when` (and similar), **skip** `answer_type_mismatch_suspected` for numeric expectation and use a **small allowlist** (weekdays, month names) so correct **Sunday** does not inflate **unified_error**.

---

## Limitations

- **Simulated reasoning only** — traces are plausible, not logged model outputs.  
- **Bundled sample** — 20 GSM8K items; not full test set.  
- **No empirical FP rate** — this documents **mechanistic** false positives on chosen items, not population frequencies.  
- **`gsm8k_test_3` (Julie)** — bundled gold **54** vs typical arithmetic **42** for “half of remaining”; excluded from the five cases to avoid conflating **label noise** with **routing false positives**. The **constraint** half-of-remaining heuristic still **over-fires** on minimal traces that state **either** number; verify against HF GSM8K if needed.

---

## Reproduction

```bash
cd /workspace
python3 -c "
import json
from pathlib import Path
from src.features.unified_error_signal import compute_unified_error_signal
from src.features.constraint_violation_features import extract_constraint_violation_features
from src.features.number_role_features import compute_calibrated_role_decision
from src.features.target_quantity_features import extract_target_quantity_features

# Load bundled GSM8K, set question/reasoning/gold as in this doc, then print
# extract_constraint_violation_features, compute_calibrated_role_decision,
# compute_unified_error_signal, extract_target_quantity_features.
"
```

Use the exact strings from sections A–D for each case to replicate numbers.
