# Policy Catalog

All entries are grounded in repository artifacts (source code, configs, and output files).
Fields marked "not recoverable from current repository artifacts" indicate information that
cannot be reconstructed without re-running experiments or accessing missing files.

---

## 1. Always-RG (reasoning_greedy)

| Field | Value |
|---|---|
| Policy name | Always-RG |
| Internal key | `reasoning_greedy` |
| Policy type | Fixed |
| Decision rule | Always run the cheap reasoning-greedy action; never revise. |
| Input artifacts | Per-query routing dataset CSVs |
| Output files | Column `reasoning_correct`, `reasoning_answer` in per-query CSVs |
| Regimes evaluated | math500_100, hard_gsm8k_100, hard_gsm8k_b2, gsm8k_random_100 |
| Accuracy (math500_100) | 0.64 |
| Accuracy (hard_gsm8k_100) | 0.79 |
| Accuracy (hard_gsm8k_b2) | 0.83 |
| Accuracy (gsm8k_random_100) | 0.90 |
| Avg cost | 1.0 (by definition) |

**Source:** `outputs/real_*/policy_comparison.csv` rows with `route=reasoning_greedy`.

---

## 2. Always-DPR (direct_plus_revise)

| Field | Value |
|---|---|
| Policy name | Always-DPR |
| Internal key | `direct_plus_revise` |
| Policy type | Fixed |
| Decision rule | Always run the expensive direct+revise action for every query. |
| Input artifacts | Per-query routing dataset CSVs |
| Output files | Column `revise_correct`, `revise_answer` in per-query CSVs |
| Regimes evaluated | math500_100, hard_gsm8k_100, hard_gsm8k_b2, gsm8k_random_100 |
| Accuracy (math500_100) | 0.64 |
| Accuracy (hard_gsm8k_100) | 0.86 |
| Accuracy (hard_gsm8k_b2) | 0.91 |
| Accuracy (gsm8k_random_100) | 0.92 |
| Avg cost | 2.0 (by definition, cost model cheap=1 / expensive=2) |

**Source:** `outputs/real_*/policy_comparison.csv` rows with `route=direct_plus_revise`.

---

## 3. Oracle

| Field | Value |
|---|---|
| Policy name | Oracle |
| Internal key | `oracle` (computed from per-query data) |
| Policy type | Oracle upper bound |
| Decision rule | For each query, choose whichever action (RG or DPR) produces the correct answer. Uses ground-truth labels. Not deployable. |
| Input artifacts | `outputs/oracle_routing_eval/*_oracle_summary.json`; per-query `reasoning_correct` + `revise_correct` columns |
| Output files | `outputs/oracle_routing_eval/*.json` |
| Regimes evaluated | math500_100, hard_gsm8k_100, hard_gsm8k_b2, gsm8k_random_100 |
| Oracle accuracy (math500_100) | 0.70 |
| Oracle accuracy (hard_gsm8k_100) | 0.91 |
| Oracle accuracy (hard_gsm8k_b2) | 0.92 |
| Oracle accuracy (gsm8k_random_100) | 0.92 |
| Oracle avg cost | ~1.02–1.12 (cheap when possible) |

**Source:** `outputs/oracle_routing_eval/` JSON files.

---

## 4. Adaptive Policy v5

| Field | Value |
|---|---|
| Policy name | Adaptive Policy v5 |
| Internal key | `adaptive_policy_v5` |
| Policy type | Adaptive (rule-based with calibrated signal) |
| Decision rule | Uses a calibrated role/unified-error score to trigger revise. Combines constraint violation features, number-role features, target-quantity features, and a unified error signal. Escalates when unified error score > threshold or strong error signal detected. See `src/policies/adaptive_policy_v5.py`. |
| Input artifacts | `data/real_*_routing_dataset_enriched.csv` (pre-computed features) |
| Output files | `outputs/real_*/per_query_policy_decisions.csv` columns `policy_v5`, `correct_if_v5`, `cost_v5` |
| Config | `configs/adaptive_policy_v5_gsm8k.yaml` |
| Regimes evaluated | math500_100, hard_gsm8k_100, hard_gsm8k_b2, gsm8k_random_100 |
| Key hyperparameters | `unified_error_revise_threshold=0.34`, `unified_error_maybe_threshold=0.25`, `unified_high_confidence_threshold=0.70` |

**Source:** `src/policies/adaptive_policy_v5.py`, `configs/adaptive_policy_v5_gsm8k.yaml`.

---

## 5. Adaptive Policy v6

| Field | Value |
|---|---|
| Policy name | Adaptive Policy v6 |
| Internal key | `adaptive_policy_v6` |
| Policy type | Adaptive (decoupled explanation vs. answer-error signals) |
| Decision rule | Decouples explanation-incompleteness warnings from revise-worthiness. Revise is primarily driven by `answer_error_score` (constraints, parse failures). Explanation pressure adds to `explanation_warning_score` but only triggers revise when combined with low answer trust. Protects concise-correct traces from false-positive escalation. See `src/policies/adaptive_policy_v6.py`. |
| Input artifacts | `data/real_*_routing_dataset_enriched.csv` |
| Output files | `outputs/real_*/per_query_policy_decisions.csv` columns `policy_v6`, `correct_if_v6`, `cost_v6` |
| Config | `configs/adaptive_policy_v6_offline.yaml` |
| Regimes evaluated | math500_100, hard_gsm8k_100, hard_gsm8k_b2, gsm8k_random_100 |

**Source:** `src/policies/adaptive_policy_v6.py`, `configs/adaptive_policy_v6_offline.yaml`.

---

## 6. Adaptive Policy v7

| Field | Value |
|---|---|
| Policy name | Adaptive Policy v7 |
| Internal key | `adaptive_policy_v7` |
| Policy type | Adaptive (extended v6 with additional error triggers) |
| Decision rule | Extends v6 without restoring concise-correct false positives. Adds: (1) weekday question + numeric final mismatch trigger, (2) "how much more" + answer equals first dollar price in question, (3) last "= N" in body vs. final number mismatch, (4) low-confidence escalation for categorical/low-trust cases. See `src/policies/adaptive_policy_v7.py`. |
| Input artifacts | `data/real_*_routing_dataset_enriched.csv` |
| Output files | `outputs/real_*/per_query_policy_decisions.csv` columns `policy_v7`, `correct_if_v7`, `cost_v7` |
| Config | `configs/adaptive_policy_v7_offline.yaml` |
| Regimes evaluated | math500_100, hard_gsm8k_100, hard_gsm8k_b2, gsm8k_random_100 |

**Source:** `src/policies/adaptive_policy_v7.py`, `configs/adaptive_policy_v7_offline.yaml`.

---

## 7. Confidence Threshold Baseline

| Field | Value |
|---|---|
| Policy name | Confidence Threshold |
| Internal key | `confidence_threshold` |
| Policy type | Threshold-based baseline |
| Decision rule | Escalate (revise) when `unified_confidence_score < threshold`. Threshold is swept over a grid and the operating point is chosen per regime. See `scripts/run_confidence_threshold_baseline.py`. |
| Input artifacts | `data/real_*_routing_dataset_enriched.csv` (column `unified_confidence_score`) |
| Output files | `outputs/baselines/confidence_threshold/confidence_threshold_sweep.csv`, `confidence_threshold_summary.json` |
| Regimes evaluated | gsm8k_random_100, hard_gsm8k_100, hard_gsm8k_b2, math500_100 |
| Operating thresholds | gsm8k_random_100: 0.65; hard_gsm8k_100: 0.40; hard_gsm8k_b2: 0.40; math500_100: 0.25 |

**Source:** `scripts/run_confidence_threshold_baseline.py`, `outputs/baselines/confidence_threshold/`.

---

## 8. Learned Router – Logistic Regression

| Field | Value |
|---|---|
| Policy name | Learned Router (Logistic Regression) |
| Internal key | `learned_router_logistic_regression` |
| Policy type | Learned classifier |
| Decision rule | Binary classifier trained to predict `revise_helpful` from pre-computed features. Uses logistic regression with cross-validation. See `scripts/run_learned_router_baseline.py`. |
| Input artifacts | `data/real_*_routing_dataset_enriched.csv` (pre-computed feature columns) |
| Output files | `outputs/baselines/learned_router/learned_router_summary.csv` |
| Regimes evaluated | gsm8k_random_100, hard_gsm8k_100, hard_gsm8k_b2, math500_100 |
| Notes | Cross-validation folds vary by regime (2-fold for gsm8k due to low positive count). |

**Source:** `scripts/run_learned_router_baseline.py`, `outputs/baselines/learned_router/learned_router_summary.csv`.

---

## 9. Learned Router – Decision Tree

| Field | Value |
|---|---|
| Policy name | Learned Router (Decision Tree) |
| Internal key | `learned_router_decision_tree` |
| Policy type | Learned classifier |
| Decision rule | Binary decision tree classifier trained to predict `revise_helpful`. Degenerate on gsm8k_random_100 (all same class prediction). |
| Input artifacts | `data/real_*_routing_dataset_enriched.csv` |
| Output files | `outputs/baselines/learned_router/learned_router_summary.csv` |
| Regimes evaluated | gsm8k_random_100 (degenerate), hard_gsm8k_100, hard_gsm8k_b2, math500_100 |

**Source:** `scripts/run_learned_router_baseline.py`, `outputs/baselines/learned_router/learned_router_summary.csv`.

---

## 10. Reasoning-Then-Revise Add-on

| Field | Value |
|---|---|
| Policy name | Reasoning-Then-Revise |
| Internal key | `reasoning_then_revise` |
| Policy type | Fixed (sequential) |
| Decision rule | Run reasoning first, then always revise (sequential two-stage). Differs from DPR in that it conditions the revision on the reasoning output. |
| Input artifacts | `outputs/reasoning_then_revise/*_rtr_addon_summary.json` |
| Output files | `outputs/reasoning_then_revise/` JSON summaries |
| Regimes evaluated | math500_100, hard_gsm8k_100, hard_gsm8k_b2, gsm8k_random_100 (summary only) |
| Accuracy | gsm8k: 0.93, hard_gsm8k_100: 0.90, math500: 0.67, hard_gsm8k_b2: not recoverable from current CSV |

**Source:** `outputs/reasoning_then_revise/`, `outputs/cross_regime_comparison/final_cross_regime_summary.csv`.

---

## 11. Token-Budget Router (Length-Based, Compute-Only)

| Field | Value |
|---|---|
| Policy name | Token-Budget Router |
| Internal key | `token_budget_router` |
| Policy type | Threshold baseline (compute-only) |
| Decision rule | Escalate from RG to DPR when a cheap-route **length feature** is outside `[min_len_threshold, max_len_threshold]`; otherwise keep RG. No semantic error/constraint signals are used. |
| Feature modes | `raw` (cheap output length), `ratio_question_tokens` (cheap output length / input length), `zscore` (normalized cheap output length) |
| Input artifacts | `data/real_*_routing_dataset_enriched.csv` with cheap-route length fields |
| Tuning command | `python -m routing.token_budget_router.tune --config config/token_budget_router_default.yaml` |
| Evaluation command | `python -m routing.token_budget_router.eval --config config/token_budget_router_default.yaml` |
| Output files | `outputs/token_budget_router/*/policy_comparison.csv`, `outputs/token_budget_router/budget_curves/*_token_budget_curve.csv`, `outputs/token_budget_router/token_budget_router_summary.json` |
| Regimes evaluated | gsm8k_random_100, hard_gsm8k_100, hard_gsm8k_b2, math500_100 |

**Source:** `src/policies/token_budget_router.py`, `routing/token_budget_router/tune.py`, `routing/token_budget_router/eval.py`, `config/token_budget_router_default.yaml`.

---

## Summary Table

| Policy | Type | All 4 regimes? | Per-query data? | Oracle comparison? |
|---|---|---|---|---|
| Always-RG | Fixed | ✅ | ✅ | ✅ |
| Always-DPR | Fixed | ✅ | ✅ | ✅ |
| Oracle | Oracle upper bound | ✅ | ✅ (derived) | — |
| Adaptive v5 | Adaptive | ✅ | ✅ | ✅ |
| Adaptive v6 | Adaptive | ✅ | ✅ | ✅ |
| Adaptive v7 | Adaptive | ✅ | ✅ | ✅ |
| Confidence threshold | Threshold | ✅ | ❌ (sweep only) | ❌ |
| Learned Router (LR) | Learned | ✅ | ❌ (summary only) | ❌ |
| Learned Router (DT) | Learned | ✅ | ❌ (summary only) | ❌ |
| Reasoning-Then-Revise | Fixed sequential | ✅ (summaries) | ❌ | ❌ |
| Token-Budget Router | Threshold (compute-only) | ✅ | ✅ | ✅ |
