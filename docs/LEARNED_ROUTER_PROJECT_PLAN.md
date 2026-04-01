# Learned Router Project Plan (Repository-Grounded)

Date: 2026-03-31

## Phase 1 audit summary

- Canonical main regimes (from `docs/CANONICAL_MANUSCRIPT_DECISIONS.md`):
  - `gsm8k_random_100`
  - `hard_gsm8k_100`
  - `hard_gsm8k_b2`
  - `math500_100`
- Available committed per-prompt outcomes:
  - `reasoning_correct`, `revise_correct`
  - `reasoning_cost`, `revise_cost`
  - Prompt-level routing features in each enriched CSV
- Existing learned-router artifacts:
  - `scripts/run_learned_router_baseline.py`
  - `outputs/baselines/learned_router/learned_router_summary.csv`
  - These are CV summaries, not explicit train/val/test split runs.

## Practical action space in committed artifacts

- In practice this repo has complete per-prompt outcomes for **2 primitive actions**:
  1. `reasoning_greedy`
  2. `direct_plus_revise`
- Additional methods (adaptive v5/v6/v7, confidence-threshold) are available as policy outputs/decisions, but not full independent action-outcome columns for a larger multi-action label space in the same canonical datasets.

## Needed implementation to reach end-to-end learned routing

1. Build a dedicated per-prompt routing ML dataset from committed enriched CSVs.
2. Create deterministic prompt-level train/validation/test splits with no leakage.
3. Train lightweight ML routers (logistic regression, decision tree, random forest).
4. Evaluate on held-out test split with:
   - classification metrics
   - routing accuracy/cost metrics
   - baseline/oracle comparison
5. Produce a manuscript-facing report with exact limitations and whether contribution is strengthened.

## Expansion decision

- No API/Wulver expansion is required for this first end-to-end learned-router pass because committed canonical data already contains 400 rows (4x100) with complete 2-action outcomes and routing-time features.
