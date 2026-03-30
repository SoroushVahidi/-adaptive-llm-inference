# MATH500 Real Routing Results (N=100)

**Evidence:** **measured_now** for committed `outputs/real_math500_routing/` and `data/real_math500_routing_dataset.csv`.

## Data

- Source: **HuggingFace** `HuggingFaceH4/MATH-500` (local `data/math500_uploaded_normalized.jsonl` was not required).
- **Answer matching:** `normalize_math_answer` on extracted `\boxed{}` / final-line answers (`answer_match_mode: math`).
- Reasoning prompt encourages `\boxed{}` or explicit final line.

## Headline metrics (`math500_run_summary.json`)

| Metric | Value |
|--------|-------|
| reasoning accuracy | **0.64** |
| direct_plus_revise accuracy | **0.64** |
| revise_helpful rate | **6%** (6 / 100) |

## Interpretation

**measured_now:** On this slice, the **second stage did not improve average accuracy** vs one-shot reasoning. Many MATH500 gold answers are **symbolic**; the revise path still emphasizes **numeric** extraction from `direct_plus_revise`, so `revise_helpful` stays low relative to total error mass.

## Policy eval (`outputs/real_math500_policy_eval/summary.json`)

- **V6 vs V7:** **tie** at **0.65** accuracy; V7 slightly higher cost (0.09 vs 0.03 revise rate).
- **adaptive_policy_v5** reaches **0.66** accuracy but **1.71** average cost (heavy revise).

## Learned router (`outputs/real_math500_routing_model/`)

- **6 positives** — CV F1 peaks around **0.4** (decision tree); bagging/boosting often collapse.
- **exploratory_only** for deployment; label count still small.

## Caveats

- Single model; 100 rows; symbolic mismatch between label definition and numeric-heavy revise pipeline.
