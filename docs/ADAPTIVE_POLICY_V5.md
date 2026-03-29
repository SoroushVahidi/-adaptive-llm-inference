# Adaptive Policy V5 (Unified Error-Aware, Offline-Ready)

## What changed

Policy v5 now supports an optional unified-error mode that combines prior-work-inspired signal families, while keeping strategy families unchanged.

- Default baseline strategy behavior unchanged.
- Post-first-pass routing can use:
  - calibrated role-only logic, or
  - unified error + confidence logic.

## Unified routing rule (when enabled)

- high confidence + low error => `no_escalation`
- medium risk => `maybe_escalate`
- low confidence + high error => `strong_escalation_candidate` (revise)

No new reasoning strategy families are introduced.

## Evidence labels

- Dry/offline logic integration: **measured_now** (unit tests + synthetic benchmark)
- Real GSM8K/API routing run in this environment: **blocked**
- Claim that v5 now beats reasoning_greedy on real tasks: **exploratory_only**
