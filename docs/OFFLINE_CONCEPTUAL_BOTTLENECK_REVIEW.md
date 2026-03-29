# Offline Conceptual Bottleneck Review

**Date:** 2026-03-29  
**Method:** Repository-only inspection (code, tests, configs, docs). No live APIs or model calls.  
**Evidence legend:** **implemented** = in code/tests; **measured** = backed by committed artifacts or tests that assert a quantity; **blocked** = designed to run but requires keys/network and has no committed outputs; **speculative** = narrative or design intent without verification.

---

## 1. Project goal (intended scientific claim)

**Stated aim** (from `docs/PROJECT_CONTEXT.md`): adaptive allocation of test-time compute across reasoning queries under a budget, maximizing answer quality—framed for EAAI as applied AI with knapsack-style structure (MCKP) as *known* background, not novelty.

**Implied contribution the codebase is set up to support:** show that **prediction-based routing** (query features ± first-pass output) can improve **budgeted utility** (accuracy per unit cost) versus strong fixed baselines—especially **`reasoning_greedy`** at one call—by selectively spending extra compute only where marginal benefit is high.

---

## 2. Strongest current ideas (what is already solid)

| Area | Why it is strong | Evidence type |
|------|------------------|----------------|
| **End-to-end experiment plumbing** | Datasets (GSM8K, MATH500), strategy runners, oracle-subset harness, adaptive policies v1–v5, logging | **implemented** |
| **Clear cheap baseline anchor** | Policies and docs repeatedly center **`reasoning_greedy`** as the bar adaptive routing must beat | **implemented** + **docs** |
| **MCKP + synthetic TTC** | Exact DP allocator, simulated sweeps, tests that MCKP ≥ equal on nontrivial synthetic instances | **implemented** + **measured** (test assertions) |
| **Interpretable signal families** | Regex/lightweight features (target quantity, constraints, roles, unified aggregation) with unit tests | **implemented** |
| **Offline consistency benchmark** | `src/analysis/consistency_benchmark.py` defines failure types (`wrong_target_quantity`, `intermediate_as_final`, etc.) and compares checker variants without API | **implemented** |
| **Evidence discipline docs** | `docs/STATE_OF_EVIDENCE.md`, `docs/UPDATED_EVALUATION_PRINCIPLES.md`, `docs/RESULTS_INVENTORY.md` correctly separate code from committed results | **implemented** (meta) |

---

## 3. Weakest or most fragile ideas (critical)

| Issue | Detail | Evidence type |
|-------|--------|----------------|
| **Profit / utility model for allocation** | MCKP solves allocation *given* per-query profit tables. There is no first-class, validated module that maps `(question, first_pass)` → calibrated **marginal** success probabilities for each discrete action under your cost proxy. Heuristic scores (v4 weighted violations, v5 unified_error) are **not** the same as estimated utility increments. | **implemented** (partial heuristics) / **gap** |
| **Selective routing vs strong default** | The open scientific question in `docs/OPEN_QUESTIONS_NOW.md` and `docs/STATE_OF_EVIDENCE.md`—beat `reasoning_greedy` on quality-per-cost—has **no committed empirical resolution**. | **blocked** + **docs** |
| **Overlapping policy generations** | v1–v5 stack increases maintenance surface; thresholds and weights in `unified_error_signal.py` look **hand-tuned** without documented calibration protocol tied to a loss. | **implemented** / **speculative** |
| **Staleness inside the repo story** | `docs/FEATURE_GAP_ANALYSIS.md` §4.3 still says wording-trap features are “unimplemented,” but `src/features/target_quantity_features.py` now implements overlapping cues—internal narrative drift. | **docs vs code** |
| **Oracle narrative vs BLOCKED logs** | `docs/ORACLE_ANALYSIS_SUMMARY.md` contains rich numbers; `docs/RESULTS_ORACLE_SUBSET.md` and `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md` say BLOCKED; `docs/RESULTS_INVENTORY.md` marks oracle table **UNVERIFIED**. Treat oracle numbers as **non-repo-grounded** until artifacts exist. | **blocked** + **conflicting docs** |

---

## 4. Most important unresolved conceptual bottleneck (method-internal)

**Bottleneck:** The project optimizes *allocation* and *routing* in separate mental models without a single **identified estimand** for “value of one more unit of compute **conditional on** what we already observed.”

Concretely:

1. **Routing** asks: after one pass, should we pay for revise / best-of-3 / etc.?
2. **Knapsack** asks: given predicted utilities at each level for every query, how do we split a batch budget?

The codebase has many **error-ish proxies** (constraint mismatch, role warnings, unified error score) but does not yet close the loop: **P(correct | action, observable state) − P(correct | default action, observable state)** at fixed cost, nor **robustness** of that difference to parser noise and prompt variance. Without that object, extra feature families mostly **re-shuffle heuristics** rather than guarantee movement on the **accuracy–cost frontier**.

This is deeper than “need labels”: even with oracle CSVs, the method question remains whether signals are **aligned with marginal value** or only with **plausible-looking inconsistency**.

---

## 5. Best evidence supporting that conclusion

| Evidence | What it shows |
|----------|----------------|
| `docs/STATE_OF_EVIDENCE.md` §6–7 | Central unknown is selective reliability vs `reasoning_greedy`; empirical router superiority unproven in committed artifacts. |
| `docs/OPEN_QUESTIONS_NOW.md` | Ranks “policy value” and “signal adequacy” ahead of more strategies. |
| `src/features/unified_error_signal.py` | Fixed convex combination of sub-scores; no derivation from data or explicit link to cost-normalized utility. |
| `src/evaluation/oracle_subset_eval.py` + `STRATEGY_COST_PROXY` | Cost is invocation count; **routing** logic elsewhere does not consume the same profit-table interface as `MCKPAllocator.allocate`. |
| `docs/RESULTS_INVENTORY.md` | Only synthetic MCKP-vs-equal and offline feature firing rates are clearly reproducible without API; headline oracle story is unverified in-repo. |

---

## 6. Two to three candidate improvements (idea level, minimal new model dependency tonight)

1. **Define the estimand on paper:** For each query state \(s\) (question-only or question + first output), write the **marginal value** of action \(a\) vs baseline \(b\) under your cost proxy: \(\Delta(s) = \mathbb{E}[\text{correct}|a,s] - \mathbb{E}[\text{correct}|b,s]\) minus cost penalty. Require any router feature set to be judged by how well it ranks \(\Delta\) (e.g. ROC for “benefits from revise”), not by raw error score.

2. **Stop expanding features until a calibration story exists:** Either (i) discrete buckets with held-out empirical frequencies, or (ii) a tiny convex program that fits weights to maximize a **ranking** loss on offline failure-type labels from `consistency_benchmark.py` + gold answers—**no** new LLM calls if you use benchmark labels on public questions.

3. **Unify allocator and router interfaces:** Specify how hypothetical “level profits” for MCKP would be **filled** from the same \(s\) used by adaptive_policy_v5—today the knapsack solver is **implemented** but **decoupled** from the router’s heuristic scores (`src/allocators/mckp_allocator.py` vs `src/policies/adaptive_policy_v5.py`).

---

## 7. What to think about tonight (one focus)

**Tonight’s focus:** **Marginal value alignment**—whether your escalation triggers approximate **incremental** success probability for the *next* expensive action versus **`reasoning_greedy` alone**, and what would falsify the current unified-score design (e.g. high score but revise rarely helps, or low score but revise would often fix).

See `docs/TONIGHT_THINKING_QUESTION.md` for the exact question, subquestions, and failure-mode anchors.

---

## Files consulted (representative)

`docs/PROJECT_CONTEXT.md`, `docs/STATE_OF_EVIDENCE.md`, `docs/OPEN_QUESTIONS_NOW.md`, `docs/DECISION_POINT_SUMMARY.md`, `docs/FEATURE_GAP_ANALYSIS.md`, `docs/RESULTS_INVENTORY.md`, `docs/ORACLE_ANALYSIS_SUMMARY.md`, `docs/RESULTS_ORACLE_SUBSET.md`, `docs/ADAPTIVE_POLICY_V5.md`, `docs/CONSISTENCY_FAILURE_EXAMPLES.md`, `src/policies/adaptive_policy_v4.py`, `src/policies/adaptive_policy_v5.py`, `src/features/unified_error_signal.py`, `src/features/target_quantity_features.py`, `src/datasets/routing_dataset.py`, `src/evaluation/oracle_subset_eval.py`, `src/analysis/consistency_benchmark.py`, `src/allocators/mckp_allocator.py`, `tests/test_simulated_allocation.py`, `configs/` (oracle / routing / simulated).
