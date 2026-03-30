# Hard GSM8K Real Routing Results (N=100)

**Evidence:** **measured_now** for `outputs/real_hard_gsm8k_routing/`, `outputs/hard_regime_selection/`, and `data/real_hard_gsm8k_routing_dataset.csv`.

## Selection

- Pool: GSM8K **test** via HuggingFace (**1319** items available in cache after download).
- **Top 100** by composite hardness (see `docs/HARD_REGIME_ROUTING_STUDY.md`).
- Full ranked list + stats: `outputs/hard_regime_selection/`.

## Headline metrics (`hard_gsm8k_run_summary.json`)

| Metric | Value |
|--------|-------|
| reasoning accuracy | **0.79** |
| direct_plus_revise accuracy | **0.86** |
| revise_helpful rate | **12%** (12 / 100) |

**vs gsm8k_random100:** reasoning drops **0.90 → 0.79**; `revise_helpful` rises **2% → 12%**.

## Policy eval (`outputs/real_hard_gsm8k_policy_eval/summary.json`)

| Route | Accuracy | Avg cost |
|-------|----------|----------|
| reasoning_greedy | 0.79 | 1.0 |
| direct_plus_revise | 0.86 | 2.0 |
| adaptive_policy_v6 | 0.81 | 1.26 |
| adaptive_policy_v7 | 0.82 | 1.46 |

**measured_now:** V7 **+1 pp** over V6 with **+0.20** cost. Neither matches always-revise accuracy without higher cost.

## Learned router (`outputs/real_hard_gsm8k_routing_model/`)

- **12 positives** — best CV F1 **~0.69** (bagging).
- Routing simulation: **bagging_trees** **0.88** accuracy, **1.14** cost (vs 0.86 @ 2.0 for always-revise).

**exploratory_only:** Small N; CV variance is high.

## Caveats

- Hardness is **question-side only**, not model-verified difficulty.
- Same `direct_plus_revise` definition as GSM8K runs (direct + revise, not reasoning-first revise).
