# Routing ML Dataset Report

## Scope and repo-grounded setup
- Included canonical finalized regimes: gsm8k_random_100, hard_gsm8k_100, hard_gsm8k_b2, math500_100.
- Source artifacts: committed enriched routing CSVs under `data/real_*_routing_dataset_enriched.csv`.
- Supported per-prompt actions with committed outcomes/costs used for labels:
  - `reasoning_greedy`
  - `direct_plus_revise`
  - `reasoning_then_revise`

## Action-space status inferred from committed artifacts
- In canonical manuscript comparisons, practical routing is primarily binary (`reasoning_greedy` vs `direct_plus_revise`).
- In committed per-prompt enriched datasets, a third action outcome column exists: `reasoning_then_revise_correct`.
- Therefore this dataset is built as a **3-action** routing-label dataset, not forced to binary.
- Multi-action beyond these three (e.g., self-consistency variants, official BEST-Route action sets) is blocked for this dataset because per-prompt committed outcomes aligned to these four regimes are not uniformly available.

## Label definition
Primary policy used:
1. choose the **cheapest correct action** among available actions for that prompt.
2. if no available action is correct, choose the **cheapest action**.

Costs used:
- `reasoning_greedy`: from `reasoning_cost` (fallback 1.0)
- `direct_plus_revise`: from `revise_cost` (fallback 2.0)
- `reasoning_then_revise`: fixed 2.0 (repo run logic uses two calls)

Utility fallback was **not** needed.

## Dataset size and composition
- Total prompts: 400
- Prompts per regime:
  - gsm8k_random_100: 100
  - hard_gsm8k_100: 100
  - hard_gsm8k_b2: 100
  - math500_100: 100

- Label distribution:
  - reasoning_greedy: 366
  - direct_plus_revise: 29
  - reasoning_then_revise: 5

## Split policy
- Split strategy: per-prompt split assignment with stratification by `(regime, best_action_label)` where feasible.
- Target ratio: 70/15/15 (train/validation/test), with deterministic seed 42.
- No prompt appears in multiple splits (validated in builder checks).

- Split counts:
  - train: 272
  - test: 72
  - validation: 56

## Files generated
- `scripts/build_routing_ml_dataset.py`
- `data/routing_ml_dataset.csv`
- `data/routing_ml_dataset.jsonl`
- `data/routing_ml_splits.csv`
- `data/routing_ml_dataset_summary.json`

## Blockers / limitations
- Learned-router training baselines requiring `scikit-learn` were previously blocked in this environment; this dataset build itself does not require sklearn.
- Official BEST-Route multi-model action space is not runnable from committed artifacts, so it is excluded from label space.
