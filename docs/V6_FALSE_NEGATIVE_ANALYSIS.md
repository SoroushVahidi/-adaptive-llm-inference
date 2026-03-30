# V6 False-Negative Analysis — Wrong Answers That Slip Through

**Date:** 2026-03-29  
**Method:** Offline only — no APIs or live model calls.  
**Tooling:** `compute_v6_scores` and `choose_strategy` from `src/policies/adaptive_policy_v6.py` with default `AdaptivePolicyV6Config`.  
**Traces:** **Simulated** plausible `reasoning_greedy`-style outputs (not logged LLM runs).  
**Problems:** Real **GSM8K** question text from `src/datasets/bundled/gsm8k_test_sample.json` where noted; one **synthetic** bus problem for a clean off-by-one pattern.

---

## Executive summary

V6 **does not revise** when **`answer_error_score == 0`** and the trace passes **`final_answer_confident`** (parse + finalization cue + no hard constraint flags). **`evaluate_candidate`** only catches a **narrow** set of structural mismatches (echo, remaining vs max, etc.). **Silent arithmetic errors**, **off-by-one**, **wrong rates**, and **wrong weekdays** often produce **plausible finals** that stay **below** the revise threshold. **`explanation_warning_score` can be large** (e.g. missing literals) and V6 **still** trusts the answer if **`answer_error_score` stays 0** — the **combo rule** requires **low** confidence, but **`answer_error_score == 0` forces confidence true** in the primary path.

---

## Summary table (5 cases)

| # | Problem source | Gold | Wrong answer | `explanation_warning` | `answer_error` | `final_answer_confident` | Chosen strategy |
|---|----------------|------|--------------|----------------------:|---------------:|:-------------------------:|----------------|
| 1 | GSM8K bundled (`gsm8k_test_0`) | 72 | 62 | 0 | 0 | Yes | `reasoning_greedy` |
| 2 | GSM8K-style (simple) | 5 | 4 | 1 | 0 | Yes | `reasoning_greedy` |
| 3 | Synthetic bus | 14 | 15 | 1 | 0 | Yes | `reasoning_greedy` |
| 4 | GSM8K bundled (`gsm8k_test_1`) | 10 | 12 | 1 | 0 | Yes | `reasoning_greedy` |
| 5 | GSM8K bundled (`gsm8k_test_13`) | Sunday | Saturday | 6 | 0 | Yes | `reasoning_greedy` |

All five: **`revise_recommended: false`**, **`revise_reason: no_escalation`**.

Cases 1–4 keep **`explanation_warning` ≤ 1** (low/moderate). Case 5 uses **high** explanation_warning to illustrate that **explanation pressure alone** still may not trigger revise under V6 when **`answer_error_score` is 0**.

---

## Case 1 — Wrong total with verbose (but incorrect) arithmetic

**A. Problem text**  
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

**B. Gold answer**  
`72` (48 + 24)

**C. Wrong but plausible answer**  
`62`

**D. Simulated reasoning trace**  
```
Step one: In April Natalia sold 48 clips to her friends. Step two: In May she sold half as many as April which is twenty four clips. Step three: Adding April and May gives forty eight plus twenty four equals sixty two clips total. Final answer: 62
```

**E. V6 breakdown (default config)**  
- `explanation_warning_score`: **0**  
- `answer_error_score`: **0**  
- `final_answer_confident`: **true**  
- Route: **`reasoning_greedy`** (no revise)

**F. Why V6 trusts it**  
- Parsed numeric final **62**; **Final answer** cue present.  
- **`answer_error_score == 0`** → trust path **`zero_answer_error_with_parse_and_finalization`**.  
- **`evaluate_candidate`**: 62 avoids **intermediate_echo_risk** (not equal to 48 or 24); **total_conflict** does not fire (asks “altogether”; 62 is not below min(48, 24)).  
- **Constraint** features: no fired `triggered_constraint_signals`.

**G. Why it is actually wrong**  
**Arithmetic slip** in the narrative (48 + 24 miscomputed as 62) while **surface structure** looks like a full solution.

**H. Root cause (method)**  
V6 treats **“zero cheap error signals”** as **strong evidence of correctness**. It has **no check that intermediate steps are mutually consistent** (e.g. 48 + 24 = 72).

**I. Conceptual fix**  
**Lightweight arithmetic consistency** on extracted numbers in the reasoning (e.g. verify stated subtotals) *without* requiring every question literal to appear — narrower than old role coverage, stronger than “echo” rules.

---

## Case 2 — Simple addition error, short trace

**A. Problem text**  
Tom has 3 apples and buys 2 more. How many apples does he have now?

**B. Gold answer**  
`5`

**C. Wrong answer**  
`4`

**D. Simulated trace**  
```
3 + 2 = 4.
Final answer: 4
```

**E. V6 breakdown**  
- `explanation_warning_score`: **1** (`short_reasoning`)  
- `answer_error_score`: **0**  
- `final_answer_confident`: **true**  
- Route: **`reasoning_greedy`**

**F. Why V6 trusts it**  
Same as Case 1: **no constraint/consistency hit**; **answer_error 0** ⇒ confident.

**G. Why wrong**  
Classic **one-step arithmetic error**.

**H. Root cause**  
**Trivial wrong answers** in **simple** problems are **invisible** to surface heuristics.

**I. Conceptual fix**  
**Difficulty-aware escalation**: for **very small** question literals and **single-operation** structure, use **lower bar** for second-pass (e.g. optional verify pass) — cost is low.

---

## Case 3 — Off-by-one on subtraction (synthetic)

**A. Problem text**  
A bus has 40 seats. 26 are occupied. How many seats are left?

**B. Gold answer**  
`14`

**C. Wrong answer**  
`15`

**D. Simulated trace**  
```
40 - 26 = 15.
Final answer: 15
```

**E. V6 breakdown**  
- `explanation_warning_score`: **1**  
- `answer_error_score`: **0**  
- `final_answer_confident`: **true**  
- Route: **`reasoning_greedy`**

**F. Why V6 trusts it**  
**15** is **not** exactly **26** or **40**, so **no intermediate_echo_risk**. **remaining_conflict** requires `value >= max(nums)` → 15 ≥ 40 is false.

**G. Why wrong**  
**Off-by-one / subtraction mistake** (40 − 26 = 14).

**H. Root cause**  
**evaluate_candidate** uses **coarse** numeric checks; **small errors** inside the feasible range **pass**.

**I. Conceptual fix**  
**Independent recompute** channel (e.g. second model pass with “double-check subtraction only” or rule-based re-evaluation of the **single** arithmetic claim in the trace).

---

## Case 4 — Rate / unit time — wrong but plausible dollar amount

**A. Problem text**  
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

**B. Gold answer**  
`10` (proportional to 50/60 hour)

**C. Wrong answer**  
`12`

**D. Simulated trace**  
```
She makes 12 dollars per hour, so for this job she earned $12.
Final answer: 12
```

**E. V6 breakdown**  
- `explanation_warning_score`: **1**  
- `answer_error_score`: **0**  
- `final_answer_confident`: **true**  
- Route: **`reasoning_greedy`**

**F. Why V6 trusts it**  
Answer **matches the hourly rate** in the prompt; **no** “rate_vs_total_conflict” trigger (needs both **per/each** and **total/altogether** style cues in specific combination). **Constraints** do not flag **12** as impossible.

**G. Why wrong**  
**Confused hourly rate with earnings** for **partial hour**.

**H. Root cause**  
V6 **does not encode** “earnings for **N minutes** must scale with **N/60**” without **stronger domain rules** or a **verifier**.

**I. Conceptual fix**  
**Template-specific checks** when question contains **“per hour”** + **minutes** + **“how much”**: flag answers **equal to the rate** as **suspected** (soft score), not automatic revise — tune to limit false positives.

---

## Case 5 — Wrong weekday (categorical)

**A. Problem text**  
Jasmine had 3 paperclips on Monday, then she had 6 on Tuesday, and her number of paperclips proceeded to double on each subsequent day. On what day of the week did she first have more than 100 paperclips?

**B. Gold answer**  
`Sunday` (per GSM8K)

**C. Wrong answer**  
`Saturday`

**D. Simulated trace**  
```
Final answer: Saturday
```

**E. V6 breakdown**  
- `explanation_warning_score`: **6** (missing literals, rate-related role flags, short reasoning) — **not** “low/moderate”; included deliberately to show **high explanation pressure still does not revise** when **`answer_error_score` stays 0** and trust holds.  
- `answer_error_score`: **0** (**categorical** question → **no** `answer_type_mismatch` on non-numeric final)  
- `final_answer_confident`: **true** (**Saturday** matches weekday regex)  
- Route: **`reasoning_greedy`**

**F. Why V6 trusts it**  
V6 **explicitly** avoids treating **valid weekday tokens** as type errors. **No** numeric **consistency** path applies. **Combo revise** needs **low** trust, but trust stays **high** because **`answer_error_score == 0`**.

**G. Why wrong**  
**Calendar / doubling sequence** error: model picks **adjacent** wrong day with **high confidence**.

**H. Root cause**  
**Categorical answers** have **almost no** automated structural checks in V6 beyond **lexical validity**.

**I. Conceptual fix**  
**Optional shallow enumeration**: for “which day” problems, **recompute** the day index from **stated recurrence** in a **tiny deterministic simulator** (no LLM) when the pattern is **double each day** — flag mismatch with stated final.

---

## Cross-case synthesis

### Dominant “V6 miss” pattern

**Confident wrong finals** where:

1. The wrong number **looks structurally fine** (integer, in range, not an obvious echo of the “remaining” failure mode).  
2. **`answer_error_score` stays 0**, so **`final_answer_confident` is true** under the **primary trust rule**.  
3. **Explanation pressure does not force revise** because the **combo rule** is blocked by **high** confidence.

So the dangerous class is: **wrong but locally coherent** — not **missing numbers**, not **obvious constraint violations**, not **caught by `evaluate_candidate`**.

### Are these mostly …?

- **Confident wrong finals:** **Yes** (all five).  
- **Target mismatches:** **Sometimes** (Case 4 is rate-vs-amount; Case 5 is wrong target day).  
- **Hidden arithmetic slips:** **Yes** (Cases 1–3).  
- **Wrong compressed reasoning:** **Partially** — Case 1 is **verbose but wrong**; short traces are Cases 2–4.

### What this implies for the next bottleneck

The bottleneck shifts from **false positives on concise correct traces** to **false negatives on plausible wrong finals** when **cheap syntactic/structural checks are insufficient**. Progress needs either:

- **Stronger semantic or arithmetic verification** (still lightweight), or  
- **Probabilistic / learned** “marginal error” scores **calibrated** not to recreate mass escalation on concise correct answers, or  
- **Budgeted second checks** (sample, verify, or re-ask) **targeted** at **high-leverage** patterns (simple ops, rate×time, weekdays).

---

## V6 assumptions that are too weak (for this failure mode)

1. **`answer_error_score == 0` ⇒ high trust** — treats absence of **heuristic** flags as evidence of **correctness**.  
2. **`evaluate_candidate` on the final number only** — ignores **consistency between** written steps and **final**.  
3. **Categorical trust = regex on weekdays** — no **semantic** check against the **story**.  
4. **Combo rule rarely fires** when **`answer_error_score` is 0** — **explanation_warning** can be **large** (Case 5: 6) and still **not** revise.

---

## 2–3 candidate improvements (conceptual, recall without full FP regression)

1. **Step–answer consistency (narrow):** Parse **one** explicit arithmetic line in the trace (e.g. `a + b = c`) and **verify** `c` equals `a+b` for **small integers** — catches Case 1–3 **without** demanding all question numbers in the text.

2. **Soft “suspicious template” scores:** Add **small** `answer_error` mass (below single-shot revise threshold alone, or only with **moderate** explanation_warning) for patterns like **answer == hourly rate** under **minutes** + **earn** (Case 4) — tune on held-out data.

3. **Deterministic micro-models for recurring GSM8K motifs:** Doubling from Monday, **linear** rate problems — **closed-form** check vs final when structure matches; **skip** when parse is ambiguous (stay conservative).

---

## Limitations / caveats

- All reasoning traces are **hand-simulated**; **real** model errors may **correlate** differently with features.  
- Counts and scores are **for default** `AdaptivePolicyV6Config`; thresholds change behavior.  
- **No** large-scale **empirical false-negative rate** — five **illustrative** cases only.  
- Tighter checks risk **reintroducing** some **false positives**; any addition needs **paired** evaluation on **`FALSE_POSITIVE_ANALYSIS`** fixtures.

---

## Reproduction

```bash
cd /workspace
python3 -c "
from src.policies.adaptive_policy_v6 import compute_v6_scores, AdaptivePolicyV6Config, choose_strategy, extract_question_features_v6

cfg = AdaptivePolicyV6Config()
# Paste question + trace from any case above, then:
s = compute_v6_scores(question, trace, cfg)
feats = extract_question_features_v6(question)
print(s['explanation_warning_score'], s['answer_error_score'], s['final_answer_confident'], s['revise_recommended'])
print(choose_strategy(question, feats, trace, cfg))
"
```
