# Routing ML Dataset Expansion Report

## Objective
Expand the canonical 4-regime learned-routing dataset beyond 100 prompts/regime while preserving:
- regime definitions,
- action semantics,
- feature philosophy,
- label policy.

## Canonical regimes and action protocol used
- Regimes: `gsm8k_random_100`, `hard_gsm8k_100`, `hard_gsm8k_b2`, `math500_100`.
- Actions (same as current runnable protocol):
  - `reasoning_greedy`
  - `direct_plus_revise`
  - `reasoning_then_revise`
- Label rule preserved: **cheapest correct action**; if none correct, **cheapest action**.

## Expansion attempts and repo-supported paths checked
1. Checked committed canonical/enriched datasets and per-query outputs.
   - All canonical routing datasets in `data/` and `outputs/real_*_routing/per_query_outputs.csv` contain **100 rows per regime**.
2. Attempted fresh expansion runs using repo scripts with larger targets:
   - `scripts/run_build_real_routing_dataset.py --paired-outcomes --include-reasoning-then-revise --subset-size 300`
   - `scripts/run_build_hard_gsm8k_routing_dataset.py --max-queries 300`
   - `scripts/run_build_math500_routing_dataset.py --subset-size 300`
3. Result: expansion runs were blocked by missing `OPENAI_API_KEY` for real model inference.

## Expansion result
- Original dataset size: **400** prompts.
- New expanded dataset size: **400** prompts.
- Net increase: **0**.

### Counts by regime
- `gsm8k_random_100`: 100
- `hard_gsm8k_100`: 100
- `hard_gsm8k_b2`: 100
- `math500_100`: 100

### Counts by label
- `reasoning_greedy`: 366
- `direct_plus_revise`: 29
- `reasoning_then_revise`: 5

### Split handling
- Train/validation/test splits were regenerated for the expanded artifact path.
- Split counts: train 272, validation 56, test 72.

## Action coverage completeness
- All included rows have complete outcomes and costs for all 3 supported actions.

## Exact blockers
- **Primary blocker:** cannot generate additional per-prompt action outcomes without fresh inference runs; build scripts require `OPENAI_API_KEY` and live model calls.
- **Where observed:**
  - `run_build_real_routing_dataset.py` exits with `[BLOCKED] OPENAI_API_KEY is not set`.
  - `run_build_hard_gsm8k_routing_dataset.py` and `run_build_math500_routing_dataset.py` return summary JSON with `run_status: BLOCKED`, blocker `OPENAI_API_KEY missing`.
- **Protocol blocker:** there are no committed >100-per-regime canonical artifacts with full 3-action outcomes to consume offline.

## Is dataset size now sufficient for meaningful learned-router training?
- **Not substantially improved in this run** (still 400 total). The expansion target (e.g., 300+ per regime) was not reached due the explicit inference-access blocker above.

## Files generated for this step
- `scripts/build_routing_ml_dataset_expanded.py`
- `data/routing_ml_dataset_expanded.csv`
- `data/routing_ml_dataset_expanded.jsonl`
- `data/routing_ml_splits_expanded.csv`
- `data/routing_ml_dataset_expanded_summary.json`
- `docs/ROUTING_ML_DATASET_EXPANSION_REPORT.md`
