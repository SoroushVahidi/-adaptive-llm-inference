# Number Role Features

## Purpose

Add lightweight, interpretable per-number role metadata and role-coverage checks so routing can detect likely intermediate/target mistakes.

## Implemented components

### 1) Number extraction
`extract_problem_numbers(question_text)` extracts:
- numeric literals,
- number words,
- multiplicative/fraction words.

Per extracted number it records surface text, normalized value, source type, sentence index, local context, token span (when available), unit hint, nearby verbs.

### 2) Role assignment
`assign_number_roles(question_text)` assigns:
- expected role (`add`, `subtract`, `multiply`, `divide`, `ratio`, `compare_difference`, `capacity_ceiling`, `unknown`),
- `required_for_final_answer`,
- `strongly_required_for_final_answer`,
- confidence,
- target relation hints.

### 3) Raw role coverage features
`compute_role_coverage_features(question_text, reasoning_text, parsed_answer)` computes:
- required/strong-required coverage counts,
- missing-by-role signals,
- intermediate-stop suspicion,
- role coverage score,
- triggered raw role signals.

### 4) Calibrated decision layer
`compute_calibrated_role_decision(...)` maps raw features to:
- `role_warning_score`,
- `role_strong_error_score`,
- signal strength labels,
- calibrated decision tier:
  - `no_escalation`
  - `maybe_escalate`
  - `strong_escalation_candidate`

Calibration intentionally downweights weak isolated misses and upweights strong structural misses (subtractive/capacity/strongly-required/intermediate-stop).

## Known limitations

- String matching can miss implicit number usage.
- No symbolic proof; this is heuristic scoring only.
- Calibrated signals are not claim-ready for real routing until API-based policy evaluation is unblocked.

## Evidence status

- Offline consistency-benchmark behavior: **measured_now**
- Real GSM8K routing impact: **blocked**
- Routing utility claim: **exploratory_only**
