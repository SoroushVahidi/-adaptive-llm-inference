# REVISE_HELPFUL_CLASSIFIER

## Bottleneck targeted
The current bottleneck is whether revise-worthiness is **learnable** from cheap, interpretable signals rather than hand-tuned thresholds.

## Label definition
Primary target:
- `revise_helpful = 1` if `reasoning_greedy` is wrong and `direct_plus_revise` is correct.
- `revise_helpful = 0` otherwise.

In this environment, oracle routing artifacts are unavailable, so the current run uses an **offline fallback** derived from `data/consistency_benchmark.json`:
- each candidate answer is treated as a simulated first-pass (`reasoning_greedy`) outcome,
- benchmark gold answer is treated as simulated `direct_plus_revise` outcome.

Evidence status for this dataset construction: `exploratory_only`.

## Feature families used
The classifier builder includes the existing engineered families:
- query-only (`extract_query_features`),
- target-quantity / wording-trap,
- constraint-aware,
- number-role coverage,
- calibrated role decisions,
- self-verification,
- selective-prediction confidence proxies,
- calibration/format confidence,
- step-verification,
- unified error/confidence scores.

## Models compared
Intended models (when `scikit-learn` is available):
1. single decision tree (shallow),
2. bagging over shallow trees,
3. AdaBoost over shallow trees.

Current environment status is recorded in `outputs/revise_helpful_classifier/summary.json`.

## Evaluation protocol
- small-data-safe CV (stratified `n_splits=min(5, min_class_count)`),
- metrics: accuracy, precision, recall, F1, false positive rate, confusion matrix,
- baseline heuristic comparison via offline proxies:
  - calibrated-role checker,
  - unified-error checker.

## Routing simulation protocol
Simulated policy:
- predict `1` -> choose `direct_plus_revise` (cost 2),
- predict `0` -> choose `reasoning_greedy` (cost 1).

Reported outputs:
- simulated accuracy,
- simulated average cost,
- revise trigger count/fraction,
- comparison vs reasoning/direct-plus-revise/heuristic routers.

## Current evidence labels
- Training/eval artifacts generated now: `measured_now`.
- Learned tree/bagging/boosting comparison: `blocked` if sklearn is missing.
- Claim that ML beats heuristics: `exploratory_only` until measured with trained models.
- Claim-ready status: `not claim_ready` in this environment snapshot.

## What remains unresolved
1. Need artifact-backed oracle routing labels for non-synthetic training.
2. Need an environment with sklearn to measure tree/bagging/boosting directly.
3. Need real routing validation (API-enabled) after offline model selection.
