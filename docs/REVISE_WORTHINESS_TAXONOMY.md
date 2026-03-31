# Revise-Worthiness Taxonomy (Manuscript-Oriented)

## Purpose

This taxonomy converts repository signal families into a compact, reviewer-friendly notion of **when escalation should happen** vs **when a warning should be logged but escalation avoided**.

The intent is to support an interpretable distinction between:
- **answer-error risk** (revise-worthy), and
- **explanation-quality oddities** (often warning-worthy only).

---

## Decision tiers

- **Revise**: strong evidence answer is wrong or fragile.
- **Soft warning**: explanation quality is suspicious but answer may still be correct.
- **No revise**: low expected value of escalation (or safe cheap-correct).

---

## Taxonomy categories

## A) Target quantity mismatch

- **Definition:** Final answer appears to target the wrong quantity (e.g., returns list price when asked “how much more”).
- **Action:** **Revise**.
- **Mapping to v5/v6/v7:**
  - v5: partly via unified/constraint proxies.
  - v6: intended via answer-error/consistency signals but can miss patterns.
  - v7: explicitly strengthened with template trigger (`need_more_answer_equals_list_price`).
- **Current support level:** **Moderate-to-strong** (targeted probes/logic evidence; limited large-sample stratified stats).

## B) Constraint violation / bounds / type violation

- **Definition:** Hard inconsistencies: parse/type mismatch, bound violations, modality mismatch, impossible sign/range.
- **Action:** **Revise**.
- **Mapping:**
  - v5: revise driver via unified error components.
  - v6: core of `answer_error_score`.
  - v7: preserved and extended.
- **Support:** **Strong** conceptually and implementation-wise.

## C) Body–final inconsistency

- **Definition:** Internal reasoning content disagrees with final answer (e.g., trailing equation value differs from final number).
- **Action:** **Revise** (or high-priority warning if parse ambiguous).
- **Mapping:**
  - v5: only indirectly.
  - v6: partial via consistency checks.
  - v7: explicit `tail_equals_disagrees_with_final` signal.
- **Support:** **Moderate** (clear mechanism + targeted examples; population calibration still limited).

## D) Copied-number failure (echo risk)

- **Definition:** Final answer suspiciously repeats salient question number/rate/list price without required transformation.
- **Action:** usually **Revise** for high-risk templates; otherwise **Soft warning**.
- **Mapping:**
  - v5/v6: partial via consistency/echo features.
  - v7: strengthened in specific “how much more” template.
- **Support:** **Moderate**.

## E) Generic explanation irregularity

- **Definition:** Short/incomplete trace, missing literals, weak transition cues, but no concrete answer-error evidence.
- **Action:** **Soft warning** (typically no immediate revise).
- **Mapping:**
  - v5: could over-trigger revise (false positives).
  - v6: moved to `explanation_warning_score` and decoupled from direct revise.
  - v7: preserved decoupling unless combined with low confidence/error signals.
- **Support:** **Strong** for conceptual need; **moderate** for quantitative calibration.

## F) Hard-unsolved regardless

- **Definition:** Both cheap and revise actions wrong (`both_wrong`), indicating capability bottleneck not routing mistake.
- **Action:** **No revise expectation from binary routing alone**; classify as model-capability headroom.
- **Mapping:**
  - all versions: outside policy selection power.
- **Support:** **Strong** (explicitly measurable in routing outcome breakdown tables).

## G) Safe cheap-correct

- **Definition:** Cheap reasoning already correct and stable (`both_correct` or high confidence with no answer-error).
- **Action:** **No revise**.
- **Mapping:**
  - v5: sometimes violated (over-escalation risk).
  - v6: better protected by answer-error-first logic.
  - v7: mixed; targeted FN fixes may increase some FP revise on easier slices.
- **Support:** **Strong** as decision objective; **moderate** for perfect implementation across regimes.

## H) Revise-helpful

- **Definition:** Cheap answer wrong, revise answer correct (`dpr_only_correct` / `revise_helpful=1`).
- **Action:** **Revise** (core positive class).
- **Mapping:**
  - all policies target this class.
  - v6/v7 differ in precision-recall tradeoff for capturing it.
- **Support:** **Strong** as measured outcome, but prevalence is regime-dependent and often sparse.

## I) Explanation-odd but answer-likely-correct

- **Definition:** Explanation warnings present, yet parsed final is coherent and no answer-error evidence is triggered.
- **Action:** **Soft warning** (not mandatory revise).
- **Mapping:**
  - v6’s central design point.
  - v7 should preserve this unless stronger answer-error evidence exists.
- **Support:** **Strong** in targeted false-positive fixture logic; broad prevalence estimates are **moderate**.

---

## Compact policy-to-taxonomy alignment

- **V5:** powerful but can conflate E/I with A/B/C, creating over-escalation in concise-correct cases.
- **V6:** explicit separation: A/B/C/D dominate revise; E/I become warning-first.
- **V7:** keeps V6 separation while adding targeted A/C/D-like triggers to reduce known false negatives.

---

## Manuscript usage recommendation

Present this taxonomy early in Methods as the conceptual scaffold, then map each policy version to which category boundaries it changed:
- V5: broad caution, weaker separation.
- V6: separation principle.
- V7: targeted recall repair while preserving separation intent.
