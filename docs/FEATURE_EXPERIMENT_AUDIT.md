# Feature Experiment Audit (Existing Repository Outputs)

## Trigger for this audit

Feature-experiment-like artifacts are already present in:
- `outputs/real_routing_model/`
- `outputs/real_hard_gsm8k_routing_model/`
- `outputs/real_hard_gsm8k_b2_routing_model/`
- `outputs/real_math500_routing_model/`

This audit checks these outputs for manuscript-safe interpretation and overclaim risk.

---

## 1) What is empirically present

1. Per-regime model metrics (accuracy/precision/recall/F1/FPR).
2. Class prevalence in each summary (`num_positive`).
3. Feature importance CSV for some regimes (not all).

Notably:
- `real_routing_model` (gsm8k random) has only 2 positives; all models show zero recall/F1.
- hard regimes have higher positive counts (9–12) with non-zero precision/recall.
- math500 has 6 positives and unstable behavior across models.

---

## 2) Overclaiming risks identified

## High risk

1. **Using accuracy as primary success metric** in imbalanced settings.
   - High accuracy with zero recall is not useful routing.
2. **Treating single-run feature importances as robust conclusions**.
   - Sparse positives and potential correlated features undermine stability.
3. **Cross-regime generalization claims** from mixed-prevalence, small-n settings.

## Moderate risk

1. Claiming model family superiority (tree vs bagging vs boosting) without CI/stability tests.
2. Inferring causal mechanism from split-based importance.

---

## 3) What can be safely claimed

1. Learned routing feasibility is strongly regime-dependent and prevalence-dependent.
2. Hard regimes provide more learnable signal than easy random GSM8K slices.
3. Feature analyses are currently **diagnostic** and **exploratory**, not definitive.

---

## 4) What should be avoided in manuscript

Avoid statements like:
- “Feature X is the key driver of revise-helpfulness in general.”
- “Our learned model reliably predicts revise-helpful across datasets.”
- “High accuracy demonstrates strong routing performance.”

---

## 5) Recommended reporting template for these outputs

For each regime, report:
1. positive prevalence,
2. precision/recall/F1 of positive class,
3. false-positive cost implications,
4. top features as **associated signals**,
5. caveat on sample size and stability.

---

## 6) Keep-or-appendix recommendation

Given current artifacts, the feature experiment should be:
- **Main text (brief):** one small paragraph/figure emphasizing regime dependence and exploratory nature.
- **Appendix (full):** detailed per-model metrics and full feature rankings.

This keeps the core contribution focused on interpretable policy design and headroom decomposition.
