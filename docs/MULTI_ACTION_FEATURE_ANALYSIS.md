# Multi-action routing — feature analysis

## Scope

Tree-based models (decision tree, bagging over trees, random forest, histogram gradient boosting) expose `feature_importances_` when trained. The training script `scripts/run_multi_action_model_eval.py` records the top 30 features by importance in `outputs/multi_action_models/<dataset>_model_results.json` under each target’s `models.<name>.feature_importance_top`.

For a single decision tree, it also records a shallow walk of the root splits in `tree_top_splits` (feature name, threshold, depth).

## Current empirical status (this branch)

On the **live slices we ran** (15 GSM8K tail queries, 12 MATH500 queries with `gpt-4o-mini`), every oracle label collapsed to a **single class** (`reasoning_greedy`) because all four actions achieved identical correctness on every query (ties broken to the cheapest action). **No classifiers were fit**; the JSON documents `run_status: SKIPPED` and `skip_reason` instead of importances.

### Interpretability takeaway

Until the oracle assigns **at least two distinct winning actions** across the training slice, feature-importance analysis for “when the model chooses revise vs self-consistency” is **not yet grounded in data**. To unblock:

1. Increase `N` so disagreements appear, or
2. Filter to queries where `reasoning_greedy__correct != self_consistency_3__correct` (or similar) before training, or
3. Use a stronger / different model so action outcomes diverge.

## Features used (query + first-pass)

- **Query features** (`qf__*`): from `extract_query_features` — length, numeric counts, multi-step cues, equation-like patterns, etc.
- **First-pass features** (`fp__*`): from `extract_first_pass_features` on the **raw output of `reasoning_greedy`** for that query (the supervised signal is “best action” given question + cheap first reasoning pass).

## Binary revise routing vs multi-action

The legacy binary task (revise vs not) uses overlapping cheap signals; here the label space is **four actions**. Once labels are non-degenerate, expect trees to split first on **first-pass uncertainty / parse failure** (if those correlate with benefit from `direct_plus_revise` or `reasoning_then_revise`) and on **question complexity proxies** for when `self_consistency_3` wins.
