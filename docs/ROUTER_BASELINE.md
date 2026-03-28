# Router Baseline

## What the Router Baseline Predicts

The router baseline is a first, interpretable attempt at predicting *which
inference strategy to use* for a given query, using only cheap query-level
features that are available **before** any model call.

Two prediction tasks are supported:

| Task | Target column | Type |
|---|---|---|
| **Binary** | `direct_already_optimal` | 0 / 1 — can we just use `direct_greedy`? |
| **Multiclass** | `best_accuracy_strategy` | strategy name — which strategy was oracle-best? |

## Features Used (v1)

All 13 cheap query-only features from `extract_query_features()`:

| Feature | Description |
|---|---|
| `question_length_chars` | Raw character count |
| `question_length_tokens_approx` | Whitespace-token count |
| `num_numeric_mentions` | Number of numeric tokens |
| `num_sentences_approx` | Sentence count (split on `.!?`) |
| `has_multi_step_cue` | Multi-step keyword present |
| `has_equation_like_pattern` | Inline arithmetic expression |
| `has_percent_symbol` | `%` digit present |
| `has_fraction_pattern` | `a/b` fraction pattern |
| `has_currency_symbol` | `$`, `€`, etc. |
| `max_numeric_value_approx` | Largest number in question |
| `min_numeric_value_approx` | Smallest number in question |
| `numeric_range_approx` | `max − min` |
| `repeated_number_flag` | Same number appears twice |

First-pass output features are **not** used in v1 (they require a model call).

## Models Implemented

### 1. Majority Baseline (always available)
Always predicts the most frequent class seen during training.  Zero external
dependencies.  Sets the minimum bar that any learned model must beat.

### 2. Decision Tree (depth 3)
A shallow CART tree fitted with Gini impurity.  Interpretable via feature
importances.  Uses `scikit-learn` when installed; falls back to a pure-Python
implementation otherwise.

### 3. Logistic Regression (binary task only, sklearn required)
Standard binary logistic regression with feature standardisation.  Applied only
on the binary `direct_already_optimal` task.

## Running the Baseline

```bash
# Requires routing_dataset.csv with oracle labels
python3 scripts/run_router_baseline.py

# Custom paths
python3 scripts/run_router_baseline.py \
    --routing-csv outputs/routing_dataset/routing_dataset.csv \
    --output-dir  outputs/router_baseline
```

If the routing CSV does not exist, the script tries to rebuild it from oracle
outputs.  If oracle outputs are also missing, it stops with a clear blocker
message explaining the exact steps needed.

## Outputs

All outputs are written to `outputs/router_baseline/` by default:

| File | Contents |
|---|---|
| `summary.json` | Per-task results for all models (accuracy, class distribution, feature importances) |
| `binary_predictions.csv` | Per-query true/predicted labels for the binary task |
| `multiclass_predictions.csv` | Per-query true/predicted labels for the multiclass task |

### `summary.json` structure

```json
{
  "binary_task": [
    {
      "task": "binary",
      "model_name": "majority_baseline",
      "n_train": 12,
      "n_test": 3,
      "accuracy": 0.6667,
      "class_distribution": {"0": 8, "1": 7},
      "feature_importances": {},
      "note": ""
    },
    { "model_name": "sklearn_decision_tree_depth3", ... },
    { "model_name": "sklearn_logistic_regression", ... }
  ],
  "multiclass_task": [ ... ],
  "sklearn_available": true
}
```

## Why This Is Only a First Baseline

1. **Small oracle dataset** — the oracle evaluation currently covers ≤ 20
   queries.  With so few labelled examples any learned model will overfit or
   produce near-random results.  Treat accuracy numbers as rough estimates,
   not paper claims.

2. **No first-pass features** — first-pass output features (`first_pass_parse_success`,
   `first_pass_has_uncertainty_phrase`, …) are excluded in v1.  These are
   expected to be the most informative signals, but they require one cheap
   model call per query.

3. **No cross-validation** — the current implementation uses a simple 80/20
   holdout split (or train-on-all for tiny datasets).  Leave-one-out or k-fold
   CV is not yet implemented.

4. **No cost awareness** — the binary task only asks "is direct enough?", not
   "which strategy maximises accuracy per unit cost?".  A cost-aware label
   (`cheapest_correct_strategy`) is available in the routing dataset for future
   experiments.

5. **No calibration** — predicted probabilities are not calibrated; the output
   is a hard label, not a routing score.

## Limitations

- With < 20 oracle-labelled queries, decision-tree splits are likely spurious.
- The majority baseline can appear competitive if one strategy dominates — this
  is a dataset balance issue, not a model quality signal.
- Feature importances from the pure-Python tree are rough (split-count-based)
  and should not be interpreted as causal.
- Results will improve substantially once a larger oracle evaluation is run
  (100+ queries with real model outputs).

## Connection to the Routing Dataset

The router baseline reads directly from the routing dataset CSV produced by
`scripts/build_routing_dataset.py`.  The routing dataset is the bridge between
raw query features and oracle strategy labels — see
[`docs/ROUTING_DATASET.md`](ROUTING_DATASET.md) for details.
