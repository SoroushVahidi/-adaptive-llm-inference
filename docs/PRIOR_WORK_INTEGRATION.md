# Prior-Work Integration: Error Detection + Routing Signals

This document maps major prior-work idea categories to lightweight offline implementations in this repository.

## 1) Self-verification signals

- **Source category:** self-checking / self-consistency literature, including findings that models are often weak verifiers.
- **Implemented:** `src/features/self_verification_features.py`.
- **Mapping:** stability and conflict proxies from a single reasoning trace (answer repetition, conflict markers, hedging).
- **Expected benefit:** better detection of internally unstable outputs.
- **Primary target:** recall and calibration quality.

## 2) Selective prediction / abstention signals

- **Source category:** selective prediction / abstention (ReCoVERR-style ideas).
- **Implemented:** `src/features/selective_prediction_features.py`.
- **Mapping:** confidence proxy from hedging, final-answer clarity, margin proxy, shallow-pass agreement hook.
- **Expected benefit:** better uncertainty-aware revise decisions.
- **Primary target:** FPR (by avoiding low-information escalations) and confidence ranking.

## 3) Confidence calibration features

- **Source category:** confidence calibration literature.
- **Implemented:** `src/features/calibration_features.py`.
- **Mapping:** answer-format confidence and confidence bins.
- **Expected benefit:** simple reliability prior without training.
- **Primary target:** FPR.

## 4) Step-wise verification (light)

- **Source category:** parser/verifier and structure-verification literature.
- **Implemented:** `src/features/step_verification_features.py`.
- **Mapping:** step count, equation-like pattern detection, missing transition logic, number reuse checks.
- **Expected benefit:** catch broken reasoning chains.
- **Primary target:** recall.

## 5) Unified aggregation

- **Source category:** success-prediction signal fusion.
- **Implemented:** `src/features/unified_error_signal.py` and benchmark integration.
- **Mapping:** interpretable weighted fusion of role, constraint, target, self-verification, selective, calibration, and step signals.
- **Expected benefit:** broader failure coverage.
- **Primary target:** both recall and FPR (tradeoff depends on thresholding).

## What is STILL unsolved after integration

1. Unified checker currently has high recall but too-high FPR on this synthetic benchmark.
2. Intermediate-as-final improves, but selective precision is still weak.
3. No real GSM8K/API routing evidence in this environment (blocked), so live utility remains exploratory.
4. Better threshold calibration / conditional gating is still required before claiming production-quality escalation behavior.

## Evidence status

- Offline consistency benchmark effects: **measured_now**
- Real adaptive routing impact: **blocked**
- "Best-of-known" practical readiness: **exploratory_only**
