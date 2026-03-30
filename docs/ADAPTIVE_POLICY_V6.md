# Adaptive Policy V6 — Explanation vs Answer Error

## What was wrong with earlier policies

**V4** escalates on weighted **constraint** violations only; it often leaves **role-coverage** false positives to other layers.

**V5** (with unified signals enabled) treats **calibrated role** and **unified error** as revise drivers. **Missing intermediate numbers** in the chain-of-thought inflate **role_strong_error_score** and **unified_error**, so **correct but concise** traces get **`direct_plus_revise`** even when the **parsed final answer** is clean and constraint checks are quiet. See `docs/FALSE_POSITIVE_ANALYSIS.md`.

## Core distinction

| Concept | Meaning | Policy use |
|--------|---------|------------|
| **explanation_incomplete** | Short trace, literals from the question not echoed, `possible_intermediate_stop`-style cues | **`explanation_warning_score`** — log / soft signal only |
| **answer_likely_wrong** | Parse failure, type mismatch, target mismatch, bounds, or **parsed-answer** consistency flags | **`answer_error_score`** — primary revise driver |

**Invariant:** `explanation_incomplete ≠ answer_likely_wrong`.

## Why concise correct reasoning should not be over-penalized

Many valid solutions omit re-stating every number from the prompt. Role coverage was designed as a **proxy for thorough reasoning**, not as a **standalone correctness test**. V6 **does not** map missing-role evidence to revise unless **answer_error** crosses a threshold or a **combo rule** fires with **low trust**.

## Score structure (implementation)

**File:** `src/policies/adaptive_policy_v6.py`

- **`explanation_warning_score`** — weighted sum of: `missing_required_number` (per count), `possible_intermediate_stop_suspected`, optional role-missing flags, **`short_reasoning`** (word count ≤ `short_reasoning_max_words`).
- **`answer_error_score`** — weighted **constraint** booleans from `extract_constraint_violation_features`, plus **`consistency_*`** contributions from `src/analysis/consistency_benchmark.evaluate_candidate` on the **parsed** answer (e.g. `intermediate_echo_risk`, `remaining_conflict`, …).
- **`final_answer_confident`** — true when: no hard constraint violation; parse matches expected modality (numeric vs categorical weekday/month/yes-no); finalization cue present; **`answer_error_score == 0`** **or** coherent answer + sufficient role coverage (fallback path).

## Revise rules (summary)

1. Default first pass: **`reasoning_greedy`** (same as v2/v4/v5 for non-simple questions).
2. **Revise** if `answer_error_score ≥ answer_error_revise_threshold` (default **2**).
3. **Revise** if `explanation_warning_score ≥ explanation_warn_high` **and** `final_answer_confident` is **false** **and** `answer_error_score ≥ answer_error_moderate_for_combo` (default **1**).
4. **Categorical** questions (`asks_when`, etc.): **do not** add `answer_type_mismatch_suspected` into `answer_error_score` for non-numeric finals; weekday names are accepted.

## What improved (measured offline)

**Evidence:** `outputs/adaptive_policy_v6/` produced by:

`python3 scripts/run_adaptive_policy_v6_eval.py --config configs/adaptive_policy_v6_offline.yaml`

On the **five** traces in `docs/FALSE_POSITIVE_ANALYSIS.md` (fixtures in `src/evaluation/adaptive_policy_v6_eval.py`):

- **V5:** `direct_plus_revise` on **5/5** (100% false-positive revise rate on this set).
- **V6:** `direct_plus_revise` on **0/5** — keeps **`reasoning_greedy`** while **explanation_warning_score** can still be high.

**Recall proxies** (two synthetic wrong first passes with echo/intermediate failure): **V6** selects **`direct_plus_revise` on 2/2**.

## What remains unresolved

- **Population metrics** on full GSM8K/MATH500 with live models are still **blocked**; offline fixtures are **small** and hand-picked.
- **`evaluate_candidate`** can **misfire** on some correct numeric answers that coincide with a question literal (residual FPR).
- **V4** on the same five fixtures often stays on **`reasoning_greedy`** already; the main regression fixed vs **v5/unified** is **role+unified over-triggering**.

## Related files

- `src/policies/adaptive_policy_v6.py`
- `src/evaluation/adaptive_policy_v6_eval.py`
- `scripts/run_adaptive_policy_v6_eval.py`
- `configs/adaptive_policy_v6_offline.yaml`
- `tests/test_adaptive_policy_v6.py`
