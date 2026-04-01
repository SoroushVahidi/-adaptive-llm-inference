# Five-Action Implementation Audit

Structured per-action audit for the canonical 5-action routing target space.
Each action is assessed against seven criteria.

---

## Audit Legend

| Field | Meaning |
|-------|---------|
| **Status** | `implemented` / `partial` / `placeholder` (matches `configs/action_space_catalog.yaml`) |
| **Runnable** | Whether the action can be executed end-to-end today with an API key |
| **Outcome data** | Whether per-query `{action}_correct` columns exist in committed routing CSVs |
| **Cost estimate** | Whether a cost-proxy value exists in `STRATEGY_COST_PROXY` |
| **Oracle support** | Whether oracle-label construction code handles this action |
| **Router-ready** | Whether the learned-router training pipeline can use this action as a label |
| **Blocker** | Exact issue preventing full readiness (empty = none) |

---

## A0 — `reasoning_greedy` (RG)

| Criterion | Value |
|-----------|-------|
| **Description** | Single chain-of-thought pass; no revision or resampling. Cheapest action. |
| **Expected cost** | 1 (one cheap-model call) |
| **Expected role** | Cheap fallback for easy queries; anchor of the cost-accuracy Pareto curve |
| **Status** | `implemented` |
| **Runnable** | ✅ Yes |
| **Outcome data** | ✅ `reasoning_correct` column present in all four enriched routing CSVs |
| **Cost estimate** | ✅ `STRATEGY_COST_PROXY["reasoning_greedy"] = 1` |
| **Oracle support** | ✅ Handled by `best_accuracy_action()` / `best_utility_action()` |
| **Router-ready** | ✅ Binary `escalate_label` already uses this action as the cheap baseline |
| **Blocker** | — |

**Code paths:**
- `src/baselines/greedy.py` — `GreedyBaseline.solve()`
- `src/evaluation/strategy_expansion_eval.py` — `run_reasoning_greedy()`
- `src/evaluation/oracle_subset_eval.py` — `_ORACLE_RUNNERS["reasoning_greedy"]`

**Committed artifacts:**
- `data/real_gsm8k_routing_dataset_enriched.csv` — column `reasoning_correct`, `reasoning_cost`
- `data/real_hard_gsm8k_routing_dataset_enriched.csv` — same
- `data/real_hard_gsm8k_b2_routing_dataset_enriched.csv` — same
- `data/real_math500_routing_dataset_enriched.csv` — same

---

## A1 — `direct_plus_revise` (DPR)

| Criterion | Value |
|-----------|-------|
| **Description** | Direct first-pass answer followed by an explicit self-revision stage. The "expensive" action in the existing binary routing setup. |
| **Expected cost** | 2 (direct call + revise call) |
| **Expected role** | Sequential self-correction for queries where a single pass is likely wrong but the model can recover via revision |
| **Status** | `implemented` |
| **Runnable** | ✅ Yes |
| **Outcome data** | ✅ `revise_correct` column present in all four enriched routing CSVs |
| **Cost estimate** | ✅ `STRATEGY_COST_PROXY["direct_plus_revise"] = 2` |
| **Oracle support** | ✅ Handled by `best_accuracy_action()` / `best_utility_action()` |
| **Router-ready** | ✅ Binary `escalate_label` already uses this action as the expensive target |
| **Blocker** | — |

**Code paths:**
- `src/evaluation/strategy_expansion_eval.py` — `run_direct_plus_revise()`
- `src/evaluation/oracle_subset_eval.py` — `_ORACLE_RUNNERS["direct_plus_revise"]`

**Committed artifacts:**
- `data/real_gsm8k_routing_dataset_enriched.csv` — column `revise_correct`, `revise_cost`
- `data/real_hard_gsm8k_routing_dataset_enriched.csv` — same
- `data/real_hard_gsm8k_b2_routing_dataset_enriched.csv` — same
- `data/real_math500_routing_dataset_enriched.csv` — same

---

## A2 — `reasoning_then_revise` (RTR)

| Criterion | Value |
|-----------|-------|
| **Description** | Chain-of-thought first pass followed by a revision stage conditioned on the full reasoning trace. Combines reasoning depth with iterative self-correction. |
| **Expected cost** | 2 (reasoning call + revise call) |
| **Expected role** | Queries requiring both deeper reasoning AND a correction pass; behaviorally distinct from DPR because the first stage is CoT, not a direct answer |
| **Status** | `implemented` |
| **Runnable** | ✅ Yes (API calls succeed; runner is tested) |
| **Outcome data** | ❌ Outcome columns NOT in committed routing CSVs |
| **Cost estimate** | ✅ `STRATEGY_COST_PROXY["reasoning_then_revise"] = 2` |
| **Oracle support** | ✅ Handled by `best_accuracy_action()` once outcome data exists |
| **Router-ready** | ❌ No outcome data → no training label |
| **Blocker** | **B1**: Per-query `reasoning_then_revise__correct` / `reasoning_then_revise__cost` columns are absent from all routing CSVs.  Must be generated via `scripts/run_build_multi_action_dataset.py`. |

**Code paths:**
- `src/evaluation/strategy_expansion_eval.py` — `run_reasoning_then_revise()`
- `src/evaluation/oracle_subset_eval.py` — `_ORACLE_RUNNERS["reasoning_then_revise"]`, listed in `MULTI_ACTION_ORACLE_STRATEGIES`

**Committed artifacts:** None for this action specifically.

**Ambiguity note:** `reasoning_then_revise` uses `sample_count: 2` in
`configs/action_space_catalog.yaml`, reflecting the 2-call structure, but it
is not a parallel-vote strategy.  The cost proxy in `oracle_subset_eval.py`
is correctly set to `2` (not `3`).  No ambiguity with `self_consistency_3`.

---

## A3 — `self_consistency_3` (SC3)

| Criterion | Value |
|-----------|-------|
| **Description** | Three independent chain-of-thought samples, majority vote selects the final answer. Diversity-based approach rather than iterative revision. |
| **Expected cost** | 3 (three parallel CoT calls) |
| **Expected role** | Queries where a single CoT occasionally errs but three samples converge reliably; complementary to A1/A2 because it exploits sampling diversity rather than self-correction |
| **Status** | `implemented` |
| **Runnable** | ✅ Yes |
| **Outcome data** | ❌ Outcome columns NOT in committed routing CSVs |
| **Cost estimate** | ✅ `STRATEGY_COST_PROXY["self_consistency_3"] = 3` |
| **Oracle support** | ✅ Handled by `best_accuracy_action()` once outcome data exists |
| **Router-ready** | ❌ No outcome data → no training label |
| **Blocker** | **B1**: Same as A2 — covered by the same generation run. |

**Code paths:**
- `src/evaluation/strategy_expansion_eval.py` — `run_self_consistency_3()`
- `src/baselines/self_consistency.py` — `SelfConsistencyBaseline` (5-sample version; `run_self_consistency_3` uses 3 samples)
- `src/evaluation/oracle_subset_eval.py` — `_ORACLE_RUNNERS["self_consistency_3"]`, listed in `MULTI_ACTION_ORACLE_STRATEGIES`

**Committed artifacts:** None for this action specifically.

**Ambiguity note:** The full `self_consistency` strategy in the catalog uses
5 samples; `self_consistency_3` is the 3-sample variant.  They share the same
majority-vote logic but differ in cost (5 vs. 3 calls).  The 3-sample variant
is the correct choice for the 5-action space because: (a) cost=3 keeps it
meaningfully separated from cost=2 without over-spending, and (b) the runner
`run_self_consistency_3` is already in `MULTI_ACTION_ORACLE_STRATEGIES`.

---

## A4 — `direct_plus_critique_plus_final` (DCPF)

| Criterion | Value |
|-----------|-------|
| **Description** | Three-stage pipeline: direct answer → structured critique of that answer → final revised answer conditioned on the critique. The critique stage explicitly evaluates the initial response before regenerating. |
| **Expected cost** | 3 (direct call + critique call + final call) |
| **Expected role** | Hardest queries where structured critical evaluation — not just revision — is needed; the intermediate critique stage reasons about the error before regenerating a final answer |
| **Status** | `implemented` |
| **Runnable** | ✅ Yes (runner exists and is tested) |
| **Outcome data** | ❌ Outcome columns NOT in committed routing CSVs |
| **Cost estimate** | ✅ `STRATEGY_COST_PROXY["direct_plus_critique_plus_final"] = 3` |
| **Oracle support** | ✅ Handled by `best_accuracy_action()` once outcome data exists |
| **Router-ready** | ❌ No outcome data → no training label |
| **Blocker** | **B1 + B2**: Outcome data missing (B1); additionally, this action was NOT in `MULTI_ACTION_ORACLE_STRATEGIES` before this branch (B2).  Both blockers resolved: B2 fixed on this branch by adding the action to `MULTI_ACTION_ORACLE_STRATEGIES` and `MULTI_ACTION_ORDER`. |

**Code paths:**
- `src/evaluation/expanded_strategy_eval.py` — `run_direct_plus_critique_plus_final()`
- `src/evaluation/oracle_subset_eval.py` — `_ORACLE_RUNNERS["direct_plus_critique_plus_final"]`, now added to `MULTI_ACTION_ORACLE_STRATEGIES`

**Committed artifacts:** None for this action specifically.

**Ambiguity note:** `direct_plus_critique_plus_final` is sometimes described
as a 3-stage "critique-then-finalization" variant.  It is distinct from
`direct_plus_revise` (A1): the intermediate stage issues a structured _critique_
prompt, not a bare revision prompt, so the model is explicitly instructed to
identify errors before producing a final answer.  This is a meaningfully
different behaviour, not a near-duplicate.

---

## Summary Table

| Action | Status | Runnable (API) | Outcome data | Cost proxy | Oracle support | Router-ready | Blockers |
|--------|--------|---------------|--------------|------------|----------------|--------------|----------|
| A0 `reasoning_greedy` | implemented | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| A1 `direct_plus_revise` | implemented | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| A2 `reasoning_then_revise` | implemented | ✅ | ❌ | ✅ | ✅* | ❌ | B1 |
| A3 `self_consistency_3` | implemented | ✅ | ❌ | ✅ | ✅* | ❌ | B1 |
| A4 `direct_plus_critique_plus_final` | implemented | ✅ | ❌ | ✅ | ✅* | ❌ | B1 (B2 fixed) |

> *Oracle support logic is in place; blocked only on the outcome data not yet existing.

---

## Blocker Resolution Checklist

- [x] **B2 (code)**: Add `direct_plus_critique_plus_final` to `MULTI_ACTION_ORACLE_STRATEGIES` *(done on this branch)*
- [x] **B2 (code)**: Add `direct_plus_critique_plus_final` to `MULTI_ACTION_ORDER` *(done on this branch)*
- [ ] **B1 (data)**: Run `scripts/run_build_multi_action_dataset.py` on all 4 regimes *(requires OpenAI API key)*
- [ ] **B3 (code)**: Extend `src/routing/learned_router/features.py` with 5-way label construction
- [ ] **Step 4 (config)**: Create `configs/five_action_router.yaml` for the 5-way router training run
