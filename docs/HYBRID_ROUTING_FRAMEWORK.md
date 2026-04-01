# Hybrid Routing Framework

This document describes the modular hybrid strategy-recommendation subsystem in
`src/routing_hybrid/`.

## Architecture

The framework splits responsibilities into replaceable layers:

1. **Dataset builder** (`dataset_builder.py`): converts prompt-level routing data
   into per-candidate rows `(prompt_id, action_name)`.
2. **Feature layer** (`features/`): grouped families that can be turned on/off
   by config.
3. **Heuristic layer** (`heuristics/`): rule registry for feature augmentation,
   candidate pruning/forbidding, and utility priors.
4. **Model layer** (`models/`, `calibration.py`): interchangeable predictors for
   per-candidate success scoring.
5. **Utility layer** (`utility.py`): configurable objective definitions.
6. **Optimizer layer** (`optimizers/`): action selection under budget.
7. **Eval/reporting** (`eval.py`, `reporting.py`): metrics, baselines, outputs.

## Data Flow

1. Build candidate rows:
   - `python3 scripts/build_hybrid_routing_dataset.py`
2. Run hybrid recommender:
   - `python3 scripts/run_hybrid_strategy_recommender.py --config ...`
3. Pipeline stages:
   - load candidate rows
   - apply feature families
   - apply heuristics (optional)
   - fit model on train split
   - optionally calibrate probabilities on validation split
   - compute per-candidate utility
   - run optimizer under budget on test split
   - write predictions, chosen actions, metrics, and ablations

## What Is Predicted

Current default prediction task is:

- **per-candidate success probability** `P(correct | prompt, action)`

This is used by utility functions and optimizers. The framework also includes
fields and interfaces for gain/utility-style predictions.

## Heuristics + ML Interaction

- ML provides `pred_p_success` / score.
- Heuristics can:
  - add heuristic features
  - mark candidates as dominated/forbidden
  - apply utility adjustments (`heur_utility_adjustment`)
- Utility combines both:
  - e.g. `p_success - lambda * cost + heuristic_adjustment - beta * uncertainty`

## Optimizers

- `per_prompt_argmax`: independent local argmax.
- `greedy_upgrade`: starts from cheapest actions, greedily upgrades by marginal
  utility per added cost.
- `mckp_exact`: exact DP over multiple-choice knapsack with cost discretization
  (`cost_scale`).
- `lambda_search`: budget-aware per-prompt argmax using dual-lambda search.

## Extension Guide

### Add a new model

1. Add file under `src/routing_hybrid/models/`.
2. Implement `fit`, `predict_proba`, and `feature_importance`.
3. Register in `models/registry.py`.

### Add a new heuristic rule

1. Add rule function in `heuristics/default_rules.py`.
2. Register in `heuristics/registry.py`.
3. Enable rule in config.

### Add a new optimizer

1. Add optimizer class in `optimizers/`.
2. Implement `solve(candidate_rows, budget)`.
3. Register in `optimizers/registry.py`.

## Extension Demo Components Included

The framework now ships with concrete extension examples:

- New model: `gradient_boosting`
- New feature family: `risk_features`
- New heuristic rule: `ambiguity_penalty`
- New utility formula: `p_correct_times_reward_minus_cost`
- New optimizer: `lambda_search`

Reference config:

- `configs/hybrid_router_extension_demo.yaml`

### Change utility formula

- Add or modify formula branch in `utility.py`.
- Switch config `utility.name` and parameters.

### Turn feature families on/off

- Edit config `features.families`.

## Grounded Data Sources Used

Current implementation is grounded on committed:

- `data/routing_ml_dataset.csv`

The dataset builder derives action candidates from available `action_*_correct`
and `action_*_cost` columns, so it gracefully handles action-set differences.

## Current Limitations

- Current committed candidate data is primarily 2-action
  (`reasoning_greedy`, `direct_plus_revise`), so optimization diversity is
  limited by available actions.
- Utility labels are based on available correctness/cost fields and not new API
  runs.
- Exact MCKP uses integer cost discretization (`cost_scale`).
