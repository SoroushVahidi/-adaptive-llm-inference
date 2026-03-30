# Real Learned Router Results (GSM8K, N=100)

**Evidence:** **measured_now** for CSV/JSON under `outputs/real_routing_model/`. **exploratory_only** for claims about learnability at scale.

## Dataset

- `data/real_gsm8k_routing_dataset.csv` — 100 rows, **2** with `revise_helpful=1`, **98** with `0`.
- Command: `python3 scripts/run_real_routing_model_eval.py` (paired-outcomes CSV → `outputs/real_routing_model/`)

## scikit-learn

Required for training; installed in the run environment. Added to `pyproject.toml` optional `dev` deps as `scikit-learn>=1.3`.

## Cross-validated models

Stratified K-fold with `n_splits = min(5, n_pos, n_neg)` → **2 folds** (only 2 positives).

| Model | Accuracy | Precision | Recall | F1 | FPR |
|-------|----------|-----------|--------|-----|-----|
| decision_tree | 0.95 | 0.0 | 0.0 | 0.0 | ~0.031 |
| bagging_trees | 0.96 | 0.0 | 0.0 | 0.0 | ~0.020 |
| boosting_shallow_trees | 0.96 | 0.0 | 0.0 | 0.0 | ~0.020 |

**Interpretation:** With **2 positives**, CV folds often contain **0 or 1** positive in validation; the model **defaults to predicting 0**, hence **zero recall** and **undefined F1** behavior (reported as 0). Accuracy is **majority-class** performance, not evidence of learning `revise_helpful`.

## Routing simulation (`outputs/real_routing_model/routing_simulation.csv`)

Learned routes achieve **0.90** accuracy (same as always-reasoning) vs **0.92** for always-revise and **0.92** for heuristic columns matching V6/V7 revise rates.

**Conclusion (honest):** **Not enough positive labels** to justify learned routing on this run; collect **more queries** or **stratified hard subsets** where revise_helpful is more frequent.

## Artifacts

- `model_metrics.csv`, `per_query_predictions.csv`, `routing_simulation.csv`, `feature_importance.csv`, `summary.json`
