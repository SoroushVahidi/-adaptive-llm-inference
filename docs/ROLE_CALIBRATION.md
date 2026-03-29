# Role Calibration (Offline, Synthetic)

## Bottleneck targeted

The raw role-coverage checker had very high wrong-answer recall but over-triggered on correct answers, making it unsafe as a routing escalation signal. This pass calibrates role signals into tiered decisions (`no_escalation`, `maybe_escalate`, `strong_escalation_candidate`).

## What was added

- A calibration layer (`compute_calibrated_role_decision`) with:
  - `role_warning_score`
  - `role_strong_error_score`
  - signal strength labels (`strong_signal`, `medium_signal`, `weak_signal`)
  - final calibrated decision tier.
- Offline false-positive analysis and tradeoff reporting in `src/analysis/role_calibration.py`.

## Why raw role checking over-triggered

From synthetic benchmark false-positive analysis (`outputs/role_calibration/false_positive_analysis.csv`):

- Most frequent false-positive signal: `missing_required_number`.
- Frequent co-fires: `required_subtractive_number_missing`, `required_rate_number_missing`, and `possible_intermediate_stop_suspected`.
- Common likely causes:
  - implicit usage not recognized by simple text matching,
  - over-aggressive target/final-step inference,
  - over-interpretation of operator cues in multi-step phrasing.

## Calibration rules used

- Strong upweighting only for high-risk patterns:
  - missing strongly-required numbers,
  - missing subtractive numbers,
  - missing capacity/ceiling numbers,
  - intermediate-stop suspicion.
- Weaker/conditional handling for rate-missing signals (only when strong-missing is also present).
- Downweight when answer appears coherent and finalized with high role-coverage score.
- Escalation thresholding is conservative:
  - strong escalation: very high strong-error score,
  - maybe escalate: medium-strong combined evidence,
  - otherwise no escalation.

## Measured synthetic tradeoff

- Old checker: recall `0.725`, FPR `0.05`
- Raw role checker: recall `0.975`, FPR `0.85`
- Calibrated role checker: recall `0.85`, FPR `0.40`

Interpretation: calibration reduced FPR materially (`0.85 -> 0.40`) while preserving higher recall than old checker (`0.85 > 0.725`), but still not low enough for claim-ready routing use.

## Evidence status

- Synthetic calibration comparison: **measured_now**
- Real GSM8K/API routing impact: **blocked**
- Routing-readiness claim: **exploratory_only**
