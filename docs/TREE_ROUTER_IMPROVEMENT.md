# Tree Router Improvement

This document describes the upgraded tree-based ensemble stack for hybrid
strategy recommendation.

## Targets

Supported training targets:

- `success_binary`: predict `P(correct | prompt, action)`
- `gain_binary`: predict whether action beats baseline action
- `utility_regression`: predict utility-like scalar target

Default is per-candidate scoring (`prompt_id`, `action_name`) rather than only
multiclass best-action labels.

Reference configs:

- `configs/tree_router_smoke.yaml`
- `configs/tree_router_gsm8k.yaml`
- `configs/tree_router_hard_gsm8k.yaml`
- `configs/tree_router_hard_gsm8k_b2.yaml`
- `configs/tree_router_math500.yaml`
- `configs/tree_router_pooled.yaml`
- `configs/tree_router_pooled_gain.yaml`
- `configs/tree_router_pooled_utility.yaml`

## Why Tree Ensembles

- **Decision Tree:** interpretable baseline, low overhead.
- **Bagging Tree:** variance reduction from bootstrap aggregation.
- **Random Forest:** stronger decorrelation via feature sub-sampling.
- **Gradient Boosting:** additive residual fitting for harder boundaries.
- **AdaBoost:** lightweight boosting around shallow trees/stumps.

All implementations use scikit-learn only (no heavy external dependency).

## Training Modes

- **Per-regime** training/eval
- **Pooled** multi-regime training/eval
- **Leave-one-regime-out** transfer style evaluation

Configured in YAML under `training.scenarios`.

## Calibration

Optional probability calibration:

- `none`
- `sigmoid` (Platt)
- `isotonic`

Calibration summary is written to `calibration_summary.csv`.

## Downstream Decision Evaluation

Model scores are evaluated through selectors:

- `per_prompt_argmax`
- `greedy_upgrade`
- `mckp_exact`

Reported against:

- cheapest baseline
- always-expensive baseline
- oracle upper bound
- existing learned-router summary if available

## Extension Path for New Datasets

Candidate rows are loaded from `candidate_rows.csv`, so adding datasets is
mainly:

1. append new regimes to candidate data
2. include regime names in config `data.regimes`
3. rerun pooled/per-regime configs

No tree-model code changes are required for new regimes.

