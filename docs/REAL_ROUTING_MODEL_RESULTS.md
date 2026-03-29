# Real Routing Model Results (Initial GSM8K Attempt)

## Scope

This report covers the first learned-router attempt on the real GSM8K routing dataset produced by:

- `scripts/run_build_real_routing_dataset.py`

and evaluated by:

- `scripts/run_real_routing_model_eval.py`

## Models planned

- decision tree
- bagging over trees
- boosting over shallow trees

Target label:

- `revise_helpful`

Evaluation metrics:

- accuracy
- precision
- recall
- F1
- false positive rate

Routing simulation compares:

- learned model route
- always `reasoning_greedy`
- always `direct_plus_revise`
- heuristic calibrated-role proxy
- heuristic unified-error proxy

## Current status in this environment

If real dataset creation is blocked (e.g., missing `OPENAI_API_KEY`), learned-model evaluation is also blocked by design.

Blocked evidence should be labeled:

- `blocked`

Measured model metrics are only labeled:

- `measured_now`

once real dataset rows exist.

## Outputs when model eval succeeds

- `outputs/real_routing_model/model_metrics.csv`
- `outputs/real_routing_model/per_query_predictions.csv`
- `outputs/real_routing_model/routing_simulation.csv`
- `outputs/real_routing_model/feature_importance.csv`
