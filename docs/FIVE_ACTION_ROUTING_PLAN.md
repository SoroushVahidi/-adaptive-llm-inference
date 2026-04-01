# Five-Action Routing Plan

This document defines the canonical 5-action routing target space for the
learned routing model that maps each prompt to one of five inference actions.
It covers the rationale, implementation status, compatibility analysis, and
the exact minimal engineering steps required before training can begin.

---

## 1. Canonical 5-Action Set

All five actions are **fully implemented** in the repository (runners exist,
unit-tested, callable via existing evaluation scripts).  The blocker for
actions A2–A4 is that per-query outcome data does not yet exist in the
committed routing CSV files — a generation step is required.

| ID | Action name | Short | Cost proxy¹ | Cost tier | Mechanism |
|----|-------------|-------|-------------|-----------|-----------|
| A0 | `reasoning_greedy` | RG | 1 | Lowest | Single chain-of-thought pass |
| A1 | `direct_plus_revise` | DPR | 2 | Low-medium | Direct answer → self-revision |
| A2 | `reasoning_then_revise` | RTR | 2 | Low-medium | CoT reasoning → revision |
| A3 | `self_consistency_3` | SC3 | 3 | Medium | 3 parallel CoT samples → majority vote |
| A4 | `direct_plus_critique_plus_final` | DCPF | 3 | Medium | Direct → structured critique → final answer |

> ¹ Cost proxy = number of model calls (cheap_model tier).

### Why these five?

- **Clear cost ordering**: 1, 2, 2, 3, 3 covers the low-to-medium cost range
  that is experimentally relevant for the paper's math benchmarks.
- **Distinct mechanisms**: each action differs in at least one of {generation
  strategy, number of stages, degree of self-correction, source of diversity}.
- **No near-duplicates**: A1 and A2 both cost 2 but differ in the first stage
  (direct vs. CoT); A3 and A4 both cost 3 but differ fundamentally (parallel
  sampling + vote vs. sequential critique pipeline).
- **Repo-backed**: all five have `status: implemented` in
  `configs/action_space_catalog.yaml` and runnable code in `src/evaluation/`.
- **Builds on current 4-action multi-action setup**: the existing
  `MULTI_ACTION_ORACLE_STRATEGIES` list in
  `src/evaluation/oracle_subset_eval.py` already contains A0–A3; A4 is the
  only net-new addition required.

---

## 2. Runnable / Non-Runnable Status Table

| Action | Runner exists? | Data exists? | Runnable for training? |
|--------|---------------|--------------|------------------------|
| A0 `reasoning_greedy` | ✅ Yes | ✅ Yes (`reasoning_correct`) | ✅ **Runnable** |
| A1 `direct_plus_revise` | ✅ Yes | ✅ Yes (`revise_correct`) | ✅ **Runnable** |
| A2 `reasoning_then_revise` | ✅ Yes | ❌ No outcome columns in routing CSVs | ❌ **Blocked** |
| A3 `self_consistency_3` | ✅ Yes | ❌ No outcome columns in routing CSVs | ❌ **Blocked** |
| A4 `direct_plus_critique_plus_final` | ✅ Yes | ❌ No outcome columns in routing CSVs; not in `MULTI_ACTION_ORACLE_STRATEGIES` | ❌ **Blocked** |

---

## 3. Exact Blockers

### Blocker B1 — A2, A3, A4: per-query outcome data missing

**What is blocked:** training labels for a 5-way classifier require per-query
correctness values for every action.  The current enriched routing datasets
(`data/real_*_enriched.csv`) only contain `reasoning_correct` (A0) and
`revise_correct` (A1).

**Root cause:** the routing datasets were built when the project used a binary
(RG vs. DPR) routing formulation.  No API calls have been made for A2, A3, or
A4 on those query sets.

**Required fix:** run `scripts/run_build_multi_action_dataset.py` against all
four main regimes.  The script calls the full set of strategies listed in
`MULTI_ACTION_ORACLE_STRATEGIES` and writes per-action `{name}__correct` and
`{name}__cost` columns to output CSVs.

### Blocker B2 — A4: missing from `MULTI_ACTION_ORACLE_STRATEGIES`

**What is blocked:** even after resolving B1, the generation script will not
produce outcome data for `direct_plus_critique_plus_final` because that action
is not listed in `MULTI_ACTION_ORACLE_STRATEGIES` in
`src/evaluation/oracle_subset_eval.py` (current list has only A0–A3).

**Root cause:** the 4-action multi-action experiment was defined before the
5-action design was finalised.

**Required fix:** add `"direct_plus_critique_plus_final"` to
`MULTI_ACTION_ORACLE_STRATEGIES` in `src/evaluation/oracle_subset_eval.py`
and to `MULTI_ACTION_ORDER` in `src/evaluation/multi_action_routing.py`.
Both changes have already been made in this branch.

### Blocker B3 — Learned router: binary label, not 5-way

**What is blocked:** the learned router (`src/routing/learned_router/`) uses
a binary escalation label (`escalate_label = revise_correct & ~reasoning_correct`).
A 5-way classifier needs a multiclass label column.

**Root cause:** the router was designed for the binary routing task.

**Required fix:** extend `src/routing/learned_router/features.py` to construct
a 5-class label using the oracle assignment logic from
`src/evaluation/multi_action_routing.py:best_accuracy_action()`.  The binary
label must be kept for backward compatibility with existing experiments.

---

## 4. Minimal Engineering Steps

The following four steps, in order, are sufficient to make all five actions
usable for a learned 5-way router.

### Step 1 (Already done on this branch)

Add `"direct_plus_critique_plus_final"` to `MULTI_ACTION_ORACLE_STRATEGIES`
in `src/evaluation/oracle_subset_eval.py` and to `MULTI_ACTION_ORDER` in
`src/evaluation/multi_action_routing.py`.  This unblocks B2.

### Step 2 — Generate 5-action outcome data  *(requires OpenAI API key)*

```bash
# Run on each of the four main regimes.
# The script calls all strategies in MULTI_ACTION_ORACLE_STRATEGIES and
# writes per-action outcome CSVs.

python scripts/run_build_multi_action_dataset.py \
    --dataset gsm8k_hard \
    --output-dir outputs/five_action_oracle

python scripts/run_build_multi_action_dataset.py \
    --dataset math500 \
    --output-dir outputs/five_action_oracle

# … repeat for gsm8k100 and aime2024 as needed
```

The output CSVs will have columns such as:
- `reasoning_greedy__correct`, `reasoning_greedy__cost`
- `direct_plus_revise__correct`, `direct_plus_revise__cost`
- `reasoning_then_revise__correct`, `reasoning_then_revise__cost`
- `self_consistency_3__correct`, `self_consistency_3__cost`
- `direct_plus_critique_plus_final__correct`, `direct_plus_critique_plus_final__cost`
- `best_accuracy_action` (oracle 5-way label)
- `best_utility_action_lambda_*` (cost-penalised oracle labels)

### Step 3 — Extend the learned router to 5-way classification

In `src/routing/learned_router/features.py`:

1. Add a `FIVE_ACTION_COLS` constant listing the five
   `{name}__correct` columns from the multi-action output CSV.
2. Add a `build_five_action_label(df)` function that reads the
   `best_accuracy_action` column (which `build_multi_action_rows` already
   writes) and encodes it as an integer in `{0, 1, 2, 3, 4}`.
3. Keep the existing binary `escalate_label` path unchanged.

### Step 4 — Add 5-way router config

Copy `configs/learned_router_default.yaml` to
`configs/five_action_router.yaml` and change:
- `label_type: multiclass` (new key)
- `n_classes: 5`
- `action_names: [reasoning_greedy, direct_plus_revise, reasoning_then_revise,
   self_consistency_3, direct_plus_critique_plus_final]`
- Point the regime files at the new 5-action output CSVs from Step 2.

---

## 5. Compatibility Report

### 5.1 prompt → one of 5 actions (training labels)

| Requirement | Current state |
|-------------|---------------|
| Feature columns available | ✅ All 47 features in `FEATURE_COLS` are query-level or cheap-first-pass features; they do not depend on which action is eventually executed. |
| Label column available | ❌ `best_accuracy_action` column exists only after the 5-action generation step (Step 2 above). |
| Label encoding | ❌ Not yet implemented in `src/routing/learned_router/features.py`. |

**Verdict:** features are ready; labels need Step 2 + Step 3.

### 5.2 Per-action cost estimation

| Requirement | Current state |
|-------------|---------------|
| Cost proxy table | ✅ `STRATEGY_COST_PROXY` in `src/evaluation/oracle_subset_eval.py` already lists all 5 actions with call-count proxies. |
| Per-query cost columns | ❌ Only `reasoning_cost` and `revise_cost` are in existing CSVs; A2–A4 costs will be written by the generation step. |
| Budget allocator integration | ✅ `src/allocators/mckp_allocator.py` is action-agnostic; it operates on a cost vector which can be populated from `STRATEGY_COST_PROXY`. |

**Verdict:** cost infrastructure is in place; per-query cost columns for A2–A4
require the generation step.

### 5.3 Per-action correctness outcomes

| Requirement | Current state |
|-------------|---------------|
| Correctness recording | ✅ `build_multi_action_rows()` writes `{action}__correct` per query. |
| Actions A0, A1 | ✅ Columns `reasoning_correct`, `revise_correct` already in enriched CSVs. |
| Actions A2, A3, A4 | ❌ Require generation step. |

**Verdict:** blocked on Step 2 for A2–A4.

### 5.4 Multi-action oracle construction

| Requirement | Current state |
|-------------|---------------|
| Oracle logic | ✅ `best_accuracy_action()` and `best_utility_action()` in `multi_action_routing.py` support an arbitrary action set. |
| 5-action oracle | ❌ Requires outcome data for all 5 actions (Step 2). |
| Cost-penalised oracle | ✅ Already supports `lambda` weighting for any action set. |

**Verdict:** oracle construction code is ready; blocked on outcome data.

---

## 6. Files Created by This Branch

| File | Purpose |
|------|---------|
| `configs/five_action_space.yaml` | Canonical 5-action registry (single source of truth) |
| `docs/FIVE_ACTION_ROUTING_PLAN.md` | This document |
| `docs/FIVE_ACTION_IMPLEMENTATION_AUDIT.md` | Detailed per-action audit table |
| `scripts/validate_five_action_space.py` | Runnable validation/status script |

### Minimal code changes (same branch)

| File | Change |
|------|--------|
| `src/evaluation/oracle_subset_eval.py` | Added `direct_plus_critique_plus_final` to `MULTI_ACTION_ORACLE_STRATEGIES` |
| `src/evaluation/multi_action_routing.py` | Added `direct_plus_critique_plus_final` to `MULTI_ACTION_ORDER` |

---

## 7. Final Summary

The repository is **not yet ready** to train a 5-way routing model end-to-end,
because per-query outcome columns for actions A2, A3, and A4 do not exist in
the committed datasets.  However:

- All five actions are **fully implemented** (code paths exist and are tested).
- The feature columns are **already available** in the enriched routing CSVs.
- The oracle construction and cost estimation code is **already in place**.
- The two required code-side changes (adding A4 to the multi-action lists)
  have been made in this branch.

The only remaining work is a **single data-generation run** (Step 2) with an
OpenAI API key, followed by a **small extension** of the learned router's
label-construction logic (Step 3) and a **new config file** (Step 4).
Total estimated engineering effort: ≈ 1–2 days (excluding API call time).
