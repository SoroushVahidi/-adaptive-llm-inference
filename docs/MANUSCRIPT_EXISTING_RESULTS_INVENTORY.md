# Manuscript Existing Results Inventory

**Purpose:** Read-only inventory of every result artifact currently present in `outputs/` that is
relevant to the binary real-routing paper track.  No experiments were run to produce this file.

**Reference docs:** `docs/PROJECT_CONTEXT.md`, `docs/MANUSCRIPT_RESULTS_READINESS.md`,
`docs/MANUSCRIPT_ASSET_MAP.md`, `docs/BASELINE_TRACKER.md`.

**Model used throughout:** `gpt-4o-mini` (OpenAI), matching-mode `numeric` for GSM8K, `math` for
MATH500/AIME.

---

## 1. Policy Evaluation Results (`outputs/real_*_policy_eval/`)

Each sub-directory contains `summary.json` (routes compared), `policy_comparison.csv`, and
`per_query_policy_decisions.csv`.  All have `run_status: COMPLETED`, `evidence_status: measured_now`,
`num_rows: 100`.

### 1.1 `outputs/real_policy_eval/` — GSM8K random-100

| Route | Accuracy | Avg cost | Revise rate |
|---|---|---|---|
| reasoning_greedy | 0.90 | 1.00 | 0.00 |
| direct_plus_revise | 0.92 | 2.00 | 1.00 |
| adaptive_policy_v5 | 0.92 | 1.29 | 0.29 |
| **adaptive_policy_v6** | **0.92** | **1.18** | 0.18 |
| adaptive_policy_v7 | 0.92 | 1.30 | 0.30 |

- `revise_helpful_prevalence`: 0.02 (2%)
- **Manuscript-usable:** ✅ Yes — complete 100-query slice, all routes present, measured_now.

### 1.2 `outputs/real_math500_policy_eval/` — MATH500 100

| Route | Accuracy | Avg cost | Revise rate |
|---|---|---|---|
| reasoning_greedy | 0.64 | 1.00 | 0.00 |
| direct_plus_revise | 0.64 | 2.00 | 1.00 |
| adaptive_policy_v5 | 0.66 | 1.71 | 0.71 |
| **adaptive_policy_v6** | **0.65** | **1.03** | 0.03 |
| adaptive_policy_v7 | 0.65 | 1.09 | 0.09 |

- `revise_helpful_prevalence`: 0.06 (6%)
- **Manuscript-usable:** ✅ Yes — complete 100-query slice, measured_now.

### 1.3 `outputs/real_hard_gsm8k_policy_eval/` — Hard GSM8K 100

| Route | Accuracy | Avg cost | Revise rate |
|---|---|---|---|
| reasoning_greedy | 0.79 | 1.00 | 0.00 |
| direct_plus_revise | 0.86 | 2.00 | 1.00 |
| adaptive_policy_v5 | 0.86 | 1.53 | 0.53 |
| adaptive_policy_v6 | 0.81 | 1.26 | 0.26 |
| **adaptive_policy_v7** | **0.82** | **1.46** | 0.46 |

- `revise_helpful_prevalence`: 0.12 (12%)
- **Manuscript-usable:** ✅ Yes — highest-signal regime (strongest revise benefit, clearest policy
  differentiation); 100 queries, measured_now.

### 1.4 `outputs/real_hard_gsm8k_b2_policy_eval/` — Hard GSM8K B2 100

| Route | Accuracy | Avg cost | Revise rate |
|---|---|---|---|
| reasoning_greedy | 0.83 | 1.00 | 0.00 |
| direct_plus_revise | 0.91 | 2.00 | 1.00 |
| **adaptive_policy_v5** | **0.91** | **1.41** | 0.41 |
| adaptive_policy_v6 | 0.89 | 1.27 | 0.27 |
| adaptive_policy_v7 | 0.89 | 1.40 | 0.40 |

- `revise_helpful_prevalence`: 0.09 (9%)
- **Manuscript-usable:** ✅ Yes — second hard-regime slice, complementary to 1.3 above; 100 queries,
  measured_now.

---

## 2. Cross-Regime Comparison (`outputs/cross_regime_comparison/`)

Files: `cross_regime_summary.json` (JSON array), `cross_regime_summary.csv`,
`final_cross_regime_summary.csv`.

### 2.1 `cross_regime_summary.json` / `cross_regime_summary.csv`

Three-regime rollup (gsm8k_easy100, math500_100, hard_gsm8k_100):

| Regime | Reasoning acc. | Revise acc. | Best policy acc. | Best policy cost | RTR acc. | Learned router viability |
|---|---|---|---|---|---|---|
| gsm8k_easy100 | 0.90 | 0.92 | 0.92 (v6) | 1.18 | 0.93 | marginal_sparse |
| math500_100 | 0.64 | 0.64 | 0.65 (v6) | 1.03 | 0.67 | marginal_learner_signal |
| hard_gsm8k_100 | 0.79 | 0.86 | 0.82 (v7) | 1.46 | 0.90 | yes_weak |

- **Manuscript-usable:** ✅ Yes (3 complete regimes).

### 2.2 `final_cross_regime_summary.csv`

Extended rollup covering four dataset keys plus AIME:

| Dataset | Reasoning acc. | Revise acc. | Oracle acc. | Best policy acc. | Best policy cost | RTR acc. |
|---|---|---|---|---|---|---|
| gsm8k_random100 | 0.90 | 0.92 | 0.92 | 0.92 (v6) | 1.18 | 0.93 |
| hard_gsm8k_100 | 0.79 | 0.86 | 0.91 | 0.82 (v7) | 1.46 | 0.90 |
| math500_100 | 0.64 | 0.64 | 0.70 | 0.65 (v6) | 1.03 | 0.67 |
| aime2024 | 0.13 | 0.07 | 0.13 | *(empty)* | *(empty)* | 0.17 |

- ⚠️ **AIME row is partial** — `best_policy_accuracy`, `best_policy_cost`, `best_policy_name`
  columns are empty (no policy eval was run for AIME).
- **Manuscript-usable:** ✅ Yes for three complete rows; ⚠️ AIME row requires either a dedicated
  policy-eval run or removal from the table.

---

## 3. Next-Stage Evaluation (`outputs/next_stage_eval/`)

Four sub-directories, each containing `budget_curve.csv`, `cascade_curve.csv`,
`oracle_revise_helpful_summary.json`, and `next_stage_merged.json`.

### 3.1 Oracle revise-helpful summaries (upper-bound accuracy)

| Dataset | Oracle accuracy | Avg cost | Revise rate | n |
|---|---|---|---|---|
| gsm8k_random100 | 0.92 | 1.02 | 0.02 | 100 |
| hard_gsm8k_100 | 0.91 | 1.12 | 0.12 | 100 |
| hard_gsm8k_b2 | 0.92 | 1.09 | 0.09 | 100 |
| math500_100 | 0.70 | 1.06 | 0.06 | 100 |

- **Manuscript-usable:** ✅ Yes — establishes ceiling for "revise-helpful" oracle routing per regime.

### 3.2 Budget curves

Each `budget_curve.csv` maps `target_avg_cost → achieved_avg_cost, revise_rate, accuracy`.
Example (hard_gsm8k_b2):

| target_avg_cost | accuracy |
|---|---|
| 1.0 | 0.83 |
| 1.1 | 0.92 |
| 1.2–1.5 | 0.92 |
| 2.0 | 0.91 |

Matching curves exist for gsm8k_random100, hard_gsm8k_100, and math500_100.

- **Manuscript-usable:** ✅ Yes — cost-accuracy trade-off curves for all four regimes.

### 3.3 Cascade curves

`cascade_curve.csv` present for each of the four dataset keys.

- **Manuscript-usable:** ✅ Yes.

---

## 4. Oracle Routing Eval (`outputs/oracle_routing_eval/`)

Four JSON files (`<dataset>_oracle_summary.json`) reporting oracle upper bounds.  Numbers match
`next_stage_eval` oracle summaries exactly — these are the canonical oracle reference files.

Files:
- `outputs/oracle_routing_eval/gsm8k_random100_oracle_summary.json`
- `outputs/oracle_routing_eval/hard_gsm8k_100_oracle_summary.json`
- `outputs/oracle_routing_eval/hard_gsm8k_b2_oracle_summary.json`
- `outputs/oracle_routing_eval/math500_100_oracle_summary.json`

- **Manuscript-usable:** ✅ Yes — four complete oracle upper-bound measurements.

---

## 5. Paper Tables and Figures (`outputs/paper_tables/`, `outputs/paper_figures/`)

**⛔ Neither directory exists in the current repository.**

`docs/MANUSCRIPT_ASSET_MAP.md` and `docs/MANUSCRIPT_RESULTS_READINESS.md` describe these as the
output of `scripts/generate_paper_tables.py` and `scripts/generate_paper_figures.py`, but those
scripts have **not been run** (or their output is gitignored / not committed).

**Blocked export keys listed in MANUSCRIPT_ASSET_MAP.md:**
- `simulated_sweep` — missing `outputs/simulated_sweep/budget_sweep_comparisons.csv`
- `oracle_subset` — missing `outputs/oracle_subset_eval/summary.json`

---

## 6. Supporting / Background Artifacts

### 6.1 Raw routing datasets (LLM call logs)

| Path | Dataset | Queries | Model | Evidence status |
|---|---|---|---|---|
| `outputs/real_routing_dataset/` | GSM8K random-100 | 100 | gpt-4o-mini | measured_now |
| `outputs/real_hard_gsm8k_routing/` | Hard GSM8K 100 | 100 | gpt-4o-mini | measured_now |
| `outputs/real_hard_gsm8k_routing_b2/` | Hard GSM8K B2 100 | 100 | gpt-4o-mini | measured_now |
| `outputs/real_math500_routing/` | MATH500 100 | 100 | gpt-4o-mini | measured_now |
| `outputs/real_aime2024_routing/` | AIME 2024 | 30 | gpt-4o-mini | measured_now |

Each directory contains `raw_responses.jsonl`, `per_query_outputs.csv`, `*_run_summary.json`,
`checkpoint.json`, `provider_metadata.json`.

### 6.2 Learned routing models

| Path | Dataset | n_positive | Best model | Best F1 |
|---|---|---|---|---|
| `outputs/real_routing_model/` | GSM8K random-100 | 2 | decision_tree | 0.00 |
| `outputs/real_hard_gsm8k_routing_model/` | Hard GSM8K 100 | 12 | bagging_trees | **0.69** |
| `outputs/real_hard_gsm8k_b2_routing_model/` | Hard GSM8K B2 100 | 9 | bagging_trees | **0.57** |
| `outputs/real_math500_routing_model/` | MATH500 100 | 6 | decision_tree | 0.40 |

Each directory contains `model_metrics.csv`, `per_query_predictions.csv`,
`routing_simulation.csv`, and (where present) `feature_importance.csv`.

- ⚠️ **GSM8K random-100 routing model: completely degenerate** (F1 = 0 across all models; only 2
  positives in 100 queries — too sparse to learn).
- ⚠️ **MATH500 routing model: weak** (F1 = 0.40 best; 6 positives only).
- ✅ **Hard GSM8K 100 routing model: usable as a learned-router result** (bagging F1 = 0.69; 12
  positives, boosting precision = 0.75).
- ✅ **Hard GSM8K B2 routing model: usable** (bagging F1 = 0.57; 9 positives, bagging precision =
  0.80).
- Top features (MATH500 model): `q_max_numeric_value_approx` (38.8%), `unified_confidence_score`
  (27.9%), `q_min_numeric_value_approx` (17.5%).

### 6.3 Reasoning-then-revise (RTR) addon summaries

| Path | Dataset | RTR accuracy | RTR helpful rate |
|---|---|---|---|
| `outputs/reasoning_then_revise/gsm8k_rtr_addon_summary.json` | GSM8K random-100 | 0.93 | 0.03 |
| `outputs/reasoning_then_revise/hard_gsm8k_b2_rtr_addon_summary.json` | Hard GSM8K B2 | 0.88 | 0.06 |
| `outputs/reasoning_then_revise/math500_rtr_addon_summary.json` | MATH500 100 | 0.67 | 0.03 |

- ⚠️ **`hard_gsm8k_rtr_addon_summary.json` is missing** from `outputs/reasoning_then_revise/`.
  RTR accuracy for Hard GSM8K 100 is reported in `final_cross_regime_summary.csv` (0.90) but the
  backing JSON file is absent.

### 6.4 Budget sweep

`outputs/budget_sweep/` contains four budget-curve CSVs:
- `gsm8k_random100_budget_curve.csv`
- `hard_gsm8k_100_budget_curve.csv`
- `hard_gsm8k_b2_budget_curve.csv`
- `math500_100_budget_curve.csv`

These appear to duplicate (or pre-date) the per-key curves in `next_stage_eval/`.

- **Manuscript-usable:** ✅ Yes as supporting data; prefer `next_stage_eval/*/budget_curve.csv` for
  paper exports as they are produced by the canonical next-stage postprocessor.

### 6.5 Baselines

`outputs/baselines/` contains three strategy JSON files:

| File | Dataset | n_queries | Notes |
|---|---|---|---|
| `gsm8k_baseline_summary.json` | GSM8K | 30 | reasoning_greedy=0.93, SC-3=0.93, RTR=0.97 |
| `hard_gsm8k_baseline_summary.json` | Hard GSM8K | 30 | reasoning_greedy=0.80, SC-3=0.90, RTR=0.87 |
| `math500_baseline_summary.json` | MATH500 | 15 | reasoning_greedy=0.60, SC-3=0.53 |

- ⚠️ **NOT manuscript-adequate** — n=30/30/15 is far below the 100-query slices used throughout.
  `docs/MANUSCRIPT_RESULTS_READINESS.md` explicitly calls these "partial."
- **Cannot be cited alongside 100-query policy-eval numbers without qualification.**

### 6.6 Hard-regime selection metadata

- `outputs/hard_regime_selection/` and `outputs/hard_regime_selection_b2/` document the
  z-score-blend hardness formula and selection parameters (pool_size=1319, subset_size=100, seed=42,
  rank_offset=100 for B2).
- **Manuscript-usable:** ✅ Yes — supports the methodology section for how the hard-regime subsets
  were constructed.

### 6.7 Adaptive policy v7 probe

- `outputs/adaptive_policy_v7/` — false-positive / false-negative probe over 7 fixture rows.
- ⚠️ n=7 snapshot only; **appendix/methods illustration only**, not for numerical claims.

---

## 7. Strongest Currently Available Results for the Main Paper Story

| Story element | Best available artifact | Key numbers |
|---|---|---|
| **Adaptive routing improves accuracy over greedy at lower cost than always-revise** | `real_hard_gsm8k_policy_eval/summary.json` + `real_hard_gsm8k_b2_policy_eval/summary.json` | Hard GSM8K: greedy=0.79→v7=0.82 at cost 1.46 vs always-revise cost 2.00; B2: greedy=0.83→v5=0.91 at cost 1.41 |
| **Oracle ceiling confirms headroom for routing** | `oracle_routing_eval/hard_gsm8k_100_oracle_summary.json` | Oracle acc=0.91, cost=1.12 — adaptive policy (0.82/1.46) has headroom vs oracle |
| **Cross-regime budget-accuracy curves** | `next_stage_eval/hard_gsm8k_b2/budget_curve.csv` | +9 pp accuracy (0.83→0.92) achieved at avg cost 1.10 for hard B2 |
| **Regime dependence of revise helpfulness** | `cross_regime_comparison/final_cross_regime_summary.csv` | revise_helpful_rate: gsm8k=2%, hard_gsm8k=12% — strong signal of regime effects |
| **Learned routing is viable in hard regimes** | `real_hard_gsm8k_routing_model/summary.json` | Bagging F1=0.69, precision=0.64, recall=0.75 on hard GSM8K |
| **RTR is a practical alternative** | `cross_regime_comparison/final_cross_regime_summary.csv` | RTR acc=0.90 on hard_gsm8k_100 vs best policy 0.82; RTR cost=2.0 (no savings vs adaptive) |

---

## 8. Critical Missing Artifacts That Block the Results Section

| Missing artifact | Why it blocks | What generates it |
|---|---|---|
| **`outputs/paper_tables/`** and **`outputs/paper_figures/`** | Canonical publication-ready tables and figures do not exist; must run `scripts/generate_paper_tables.py` and `scripts/generate_paper_figures.py` | Run generators (no API cost; reads local CSVs/JSONs) |
| **`outputs/simulated_sweep/budget_sweep_comparisons.csv`** | Blocks the simulated/synthetic MCKP section of paper tables and figures | Run simulated sweep config |
| **`outputs/oracle_subset_eval/summary.json`** | Blocks oracle-subset paper table; existing assessment (docs) says n=15 is too small for claims | Run `scripts/run_oracle_subset_eval.py` with larger `max_samples` |
| **Strong baselines at n=100** (currently n=30/30/15 in `outputs/baselines/`) | Cannot fairly compare adaptive policies (n=100) against baselines (n=30) in the same table | Run `scripts/run_strong_baselines.py` with matching slices |
| **`outputs/reasoning_then_revise/hard_gsm8k_rtr_addon_summary.json`** | RTR number for hard_gsm8k_100 appears in `final_cross_regime_summary.csv` but source JSON is absent | Re-run RTR addon for hard_gsm8k_100 |
| **AIME policy eval** | `final_cross_regime_summary.csv` AIME row has empty best_policy columns; either fill or drop row | Either run policy eval for AIME or drop AIME from cross-regime table |

---

## 9. Summary Table

| Artifact group | Path | n | Manuscript-usable? | Notes |
|---|---|---|---|---|
| Policy eval — GSM8K random-100 | `outputs/real_policy_eval/` | 100 | ✅ Yes | |
| Policy eval — MATH500 | `outputs/real_math500_policy_eval/` | 100 | ✅ Yes | |
| Policy eval — Hard GSM8K 100 | `outputs/real_hard_gsm8k_policy_eval/` | 100 | ✅ Yes | **Strongest regime** |
| Policy eval — Hard GSM8K B2 | `outputs/real_hard_gsm8k_b2_policy_eval/` | 100 | ✅ Yes | |
| Cross-regime summary | `outputs/cross_regime_comparison/` | 3–4 regimes | ✅ Mostly | AIME row partial |
| Oracle routing eval | `outputs/oracle_routing_eval/` | 100 each | ✅ Yes | |
| Next-stage budget/cascade | `outputs/next_stage_eval/` | 100 each | ✅ Yes | |
| Budget sweep CSVs | `outputs/budget_sweep/` | 100 each | ✅ Supporting | Prefer next_stage_eval |
| RTR addon summaries | `outputs/reasoning_then_revise/` | 100 each | ✅ 3 of 4 | hard_gsm8k_100 missing |
| Learned routing models | `outputs/real_*_routing_model/` | 100 | ⚠️ Hard regimes only | GSM8K/MATH500 too sparse |
| Baselines | `outputs/baselines/` | 15–30 | ⚠️ Not adequate | Mismatched n |
| Paper tables | `outputs/paper_tables/` | — | ⛔ Missing | Not generated yet |
| Paper figures | `outputs/paper_figures/` | — | ⛔ Missing | Not generated yet |
| Simulated sweep | `outputs/simulated_sweep/` | — | ⛔ Missing | Not run yet |
| Oracle subset eval | `outputs/oracle_subset_eval/` | — | ⛔ Missing | Not run yet |
| Hard-regime selection metadata | `outputs/hard_regime_selection*/` | — | ✅ Methods | |
| Raw routing datasets | `outputs/real_*_routing/` | 100/30 | ✅ Source data | Backing for all above |
