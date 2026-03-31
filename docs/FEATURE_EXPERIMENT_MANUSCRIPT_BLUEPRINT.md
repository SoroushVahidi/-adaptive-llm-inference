# Feature Experiment Manuscript Blueprint

## Positioning assumption

Another agent may implement the feature experiment. This blueprint defines **how to interpret and write it** without overclaiming.

---

## 1) What the feature experiment should claim

It should claim **diagnostic and explanatory value**, not a replacement main contribution:

1. Certain lightweight, interpretable features can help distinguish:
   - revise-helpful cases,
   - safe cheap-correct cases,
   - and cases where both actions fail.
2. Feature patterns are **regime-dependent** and should be analyzed as such.
3. Feature evidence can clarify where V5→V6→V7 logic aligns (or misaligns) with observed outcomes.

---

## 2) What it should NOT claim

Do **not** claim:

1. Causal feature effects (“feature X causes routing success”).
2. Universal generalization across domains/models.
3. Stable superiority from one small-sample model run.
4. That feature importance alone validates policy correctness.

---

## 3) Outcomes that matter most

Prioritize reporting and interpretation for:

1. **`revise_helpful`** (positive class for escalation value).
2. **`safe_cheap`** (cheap-correct/no-revise utility region).
3. **`both_wrong`** (capability bottleneck region, routing-insensitive).
4. **`method_best_label`** (if multi-action labels exist): which strategy wins by query type.

If space is limited, prioritize in this order: `revise_helpful` > `safe_cheap` > `both_wrong` > `method_best_label`.

---

## 4) Result patterns that would strengthen contribution

1. **Consistent high-value feature families across hard regimes** for detecting revise-helpful.
2. **Clear separation** between answer-error-like signals and explanation-warning-like signals.
3. **Improved precision at controlled cost** (fewer unnecessary revises without losing too many revise-helpful hits).
4. **Regime-aware stability:** same qualitative ranking on both hard GSM8K splits.
5. **Taxonomy alignment:** top features map cleanly to revise-worthiness categories (target mismatch, constraint violations, body-final inconsistency).

---

## 5) Result patterns that weaken/complicate contribution

1. Feature importance dominated by superficial proxies (length-only effects) with weak semantic interpretation.
2. Highly unstable rankings across folds/regimes.
3. Gains only on one tiny slice with collapse elsewhere.
4. Precision-recall profile that mostly increases false positives (cost inflation) without meaningful recall gains.
5. Dependence on leakage-prone or non-deployable features.

---

## 6) How to connect feature experiment to V5→V6→V7 story

Use a **three-step narrative bridge**:

1. **V5 issue:** explanation and answer risk were partially entangled.
2. **V6 principle:** separate explanation-warning from answer-error.
3. **V7 patching:** add targeted answer-error triggers for missed patterns.

Then frame feature experiment as:

> “A post-hoc audit of whether data-driven feature salience agrees with this design progression and where additional headroom remains.”

This keeps policy design primary and feature modeling supportive.

---

## 7) How to present feature importance without causal overclaiming

Recommended phrasing:

- “Features with higher model importance are **associated** with revise-helpful labels in this dataset.”
- “Importance reflects model-specific split utility, not causal mechanism.”
- “We report stability across folds/regimes to reduce post-hoc narrative risk.”

Recommended safeguards to report:

1. Include class prevalence and base-rate context per regime.
2. Report precision/recall/F1, not accuracy alone.
3. Add ablation or permutation sanity checks where feasible.
4. Flag any apparent leakage pathways explicitly.

---

## 8) Suggested main-paper footprint

- Keep one compact figure/table in main text (top feature groups + outcome relevance).
- Move full feature lists, fold-by-fold rankings, and supplementary diagnostics to appendix.
- Tie main interpretation back to revise-worthiness taxonomy, not raw leaderboard metrics.
