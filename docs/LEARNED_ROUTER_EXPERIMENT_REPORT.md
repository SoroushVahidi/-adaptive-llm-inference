# Learned Router Experiment Report

Date: 2026-03-31  
Status: End-to-end implementation completed from committed repository artifacts (no new API runs).

## 1) Canonical setup used

- Regimes: `gsm8k_random_100`, `hard_gsm8k_100`, `hard_gsm8k_b2`, `math500_100`
- Action space used for learned routing labels:
  - `reasoning_greedy`
  - `direct_plus_revise`
- Label policy:
  - Preferred: cheapest correct action
  - Fallback: `reasoning_greedy` when both actions are incorrect
- Feature policy:
  - Routing-time numeric features only from enriched CSVs
  - Excluded action outcome/cost/label columns to avoid oracle leakage

## 2) What was built

### Scripts added

- `scripts/build_or_extend_routing_ml_dataset.py`
- `scripts/train_routing_model.py`
- `scripts/eval_routing_model.py`

### Dataset artifacts

- `data/routing_ml_dataset.csv`
- `data/routing_ml_dataset_metadata.json`

### Model artifacts

- `outputs/routing_ml_models/logistic_regression.pkl`
- `outputs/routing_ml_models/decision_tree.pkl`
- `outputs/routing_ml_models/random_forest.pkl`
- `outputs/routing_ml_models/model_validation_results.csv`
- `outputs/routing_ml_models/training_summary.json`

### Evaluation artifacts

- `outputs/routing_ml_eval/routing_model_comparison_test.csv`
- `outputs/routing_ml_eval/per_query_predictions_test.csv`
- `outputs/routing_ml_eval/evaluation_summary.json`

## 3) Dataset size and split

From `data/routing_ml_dataset_metadata.json`:

- Total rows: **400**
- Regime counts: 100 per canonical regime
- Label counts:
  - `reasoning_greedy`: 371
  - `direct_plus_revise`: 29
- Feature columns: 68
- Split counts:
  - train: 280
  - validation: 60
  - test: 60

Validation checks run:
- one row per `(regime, question_id)` pair
- no split leakage
- labels valid action names
- complete per-action correctness and cost for included actions

## 4) Models trained and selection

Trained models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

Validation selection rule:
- maximize validation routing accuracy
- tie-break by lower validation average cost

Validation summary (`outputs/routing_ml_models/model_validation_results.csv`):
- logistic_regression: val routing acc 0.8333, avg cost 1.6833
- decision_tree: val routing acc 0.8500, avg cost 1.1667
- random_forest: val routing acc 0.8500, avg cost 1.1000 (**selected**)

Best model:
- **random_forest**

## 5) Held-out test results

Best learned model (`learned_router_random_forest`) on test:
- classification accuracy: **0.9000**
- macro F1: **0.6727**
- confusion matrix (labels: reasoning_greedy, direct_plus_revise):
  - [[52, 2], [4, 2]]
- routing accuracy: **0.6667**
- average cost: **1.0667**
- revise rate: **0.0667**
- oracle gap (routing accuracy): **0.0667**

## 6) Baseline comparison on same held-out test split

From `outputs/routing_ml_eval/routing_model_comparison_test.csv`:

- always_cheap: routing acc 0.6333, cost 1.0000
- always_revise: routing acc 0.7167, cost 2.0000
- oracle: routing acc 0.7333, cost 1.1000
- adaptive_policy_v5: routing acc 0.7167, cost 1.5000
- confidence_threshold: routing acc 0.6833, cost 1.1000
- learned_router_random_forest: routing acc 0.6667, cost 1.0667

Interpretation:
- Learned router improves over always-cheap in routing accuracy (+0.0334) with small cost increase (+0.0667).
- Learned router is **below** adaptive v5 and always-revise on routing accuracy for this held-out split.
- Learned router is below confidence-threshold on routing accuracy but slightly lower cost.

## 7) Did this materially improve the paper?

Current answer: **not yet materially** as a main-claim replacement.

Why:
- The learned model is competitive in cost-efficiency vs always-cheap but does not surpass canonical adaptive v5 in held-out routing accuracy.
- The label distribution is highly imbalanced (371 vs 29), making robust action-learning difficult in this 2-action setup.

Practical value added:
- The project now has a reproducible, leakage-checked, split-based learned-routing pipeline with explicit artifacts and baseline comparisons.

## 8) Wulver/API usage and blockers

- Wulver used: **No**
- New API runs used: **No**
- Blockers encountered: **None blocking implementation**

Limitations (non-blocking):
- Action space is effectively 2-action with complete outcomes in canonical committed data.
- Existing prior learned-router baseline in `outputs/baselines/learned_router/` is CV-on-full-regime and not directly split-comparable.

## 9) Commands used

```bash
python3 scripts/build_or_extend_routing_ml_dataset.py
python3 scripts/train_routing_model.py
python3 scripts/eval_routing_model.py
```

## 10) Next missing piece (if stronger learned-router claim is desired)

- Add more positive revise-helpful examples and/or richer action outcomes under a controlled expansion plan (Wulver/API only if needed and explicitly logged), then re-run the same split-based pipeline for a fair comparison.
