# Role Coverage + Unified Follow-up Summary

## Progression

1. **Raw role signals:** very high recall, very high FPR.
2. **Calibrated role signals:** lower recall but materially better FPR.
3. **Unified error signals:** broad recall boost again, but FPR increased versus calibrated role.

## Current takeaway

- Calibrated role remains more selective than raw/unified in current synthetic setting.
- Unified checker recovers recall on difficult categories (e.g., intermediate-as-final), but is not yet selective enough for claim-ready routing.

## Why v5 / unified over-triggered on concise correct traces

Role coverage marks many question literals as **required** in the reasoning text. **Unified** and **calibrated role** then treat **missing echoes** as strong error evidence. For **correct** answers with **short** CoT, that produces **high false revise rates** (documented in `docs/FALSE_POSITIVE_ANALYSIS.md`).

## How v6 changes interpretation of missing-number signals

**Adaptive policy v6** (`src/policies/adaptive_policy_v6.py`) feeds missing-number and `possible_intermediate_stop` into **`explanation_warning_score` only**. **Revise** is driven by **`answer_error_score`** (constraints + parsed-answer consistency checks) unless a **three-way combo** fires: high explanation pressure **and** low **`final_answer_confident`** **and** at least moderate answer error. See `docs/ADAPTIVE_POLICY_V6.md`.

## Evidence labels

- Synthetic progression: **measured_now**
- Real routing gains: **blocked**
- Final routing readiness: **exploratory_only**
