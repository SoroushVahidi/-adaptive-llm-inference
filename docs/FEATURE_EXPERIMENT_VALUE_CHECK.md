# Feature Experiment Value Check

## Executive answer

A 15-feature experiment is **potentially useful** but only if framed as a **targeted diagnostic** tied to revise-worthiness categories and evaluated with prevalence-aware metrics. Otherwise it risks adding complexity without improving core contribution.

---

## 1) Will it materially improve contribution?

## Likely yes **if**

- it clarifies the answer-error vs explanation-warning separation,
- improves understanding of revise-worthiness by query type,
- and demonstrates stable, regime-aware signal relevance.

## Likely no **if**

- it is a generic feature-importance dump,
- results are unstable under small-sample class imbalance,
- or it does not change manuscript conclusions.

Bottom line: **medium upside, medium-high risk**.

---

## 2) Minimum useful version (MUV)

A minimal version worth including should have:

1. Two hard regimes (hard_gsm8k_100 + hard_gsm8k_b2) as primary testbed.
2. Outcome-centric reporting for `revise_helpful`, `safe_cheap`, `both_wrong`.
3. One compact table mapping top 5 features to taxonomy categories.
4. Stability check across folds/regimes (not just one ranking).
5. Conservative interpretation paragraph (association, not causation).

If this minimum is not met, move to appendix or defer.

---

## 3) Highest-value 5 features to prioritize (time-limited)

Prioritize features with semantic relevance to revise-worthiness:

1. **`target_quantity_mismatch`-style signal** (or nearest existing proxy).
2. **`constraint_word_conflict` / bound-violation family**.
3. **`unified_confidence_score`** (calibrated trust proxy).
4. **body–final inconsistency proxy** (e.g., tail-equals disagreement family where available).
5. **answer-type mismatch for category-sensitive questions** (e.g., weekday/numeric mismatch style).

If exact names differ by file, map nearest operational proxies and document mapping clearly.

---

## 4) Main risks

1. **Small sample / sparse positives** (`revise_helpful` often low).
2. **Overfitting to regime-specific quirks**.
3. **Feature redundancy / correlated proxies** making rankings unstable.
4. **Post-hoc storytelling** from one run.
5. **Leakage confusion** (using labels or downstream-only fields inadvertently).

---

## 5) Keep in main paper vs move to appendix: decision rules

## Keep in main paper if

1. Top feature groups are stable across hard regimes.
2. They align with taxonomy and V5→V6→V7 logic.
3. They improve interpretation of precision/recall tradeoffs (not just accuracy).
4. They sharpen manuscript claims without adding major caveats.

## Move to appendix if

1. Rankings are unstable or contradictory across regimes.
2. Gains are tiny/noisy and do not affect decisions.
3. Results depend on narrow artifacts with uncertain generalization.
4. Interpretation requires too many caveats relative to value.

---

## 6) Recommendation

Proceed with a **strictly scoped** feature experiment (diagnostic, not headline), and pre-commit interpretation rules before inspecting results to reduce post-hoc bias.
