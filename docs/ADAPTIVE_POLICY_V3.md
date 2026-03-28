## Adaptive policy v3

### Goal

Calibrate the revise trigger from v2 so that `direct_plus_revise` fires on some
queries, rather than all queries (v1) or zero queries (v2).

### What changed from v2

V2 used a conservative boolean `revise_recommended` rule over the violation
signals. On the GSM8K subset, that made the policy collapse to
`reasoning_greedy` on all 20 queries.

V3 keeps the same signal family but replaces the hard trigger with a weighted
revise score:

- `final_answer_missing_or_unclear`
- `parse_failure`
- `malformed_output`
- `uncertainty_phrase_present`
- `too_many_intermediate_numbers_without_clear_final`
- `contradiction_like_phrase_present`
- `target_mismatch_suspected`
- `unit_mismatch_suspected`
- `impossible_value_suspected`

Each signal contributes a small interpretable weight, and revise triggers when
the total score crosses a threshold.

### Threshold sweep

V3 evaluates a small sweep of settings on the same GSM8K subset:

- conservative
- medium
- aggressive

The chosen setting is selected for practical tradeoff, not just maximum
accuracy. Preference goes to settings that:

- beat or match `reasoning_greedy`
- use less cost than always revise
- trigger revise on a nontrivial but not extreme fraction of queries

### Why this matches the calibration goal

The calibration target is clear from earlier runs:

- v1: `direct_plus_revise` fired on `20 / 20`
- v2: `direct_plus_revise` fired on `0 / 20`

V3 therefore searches only over threshold/weight settings of the same v2
signals. It does not add strategies or redesign the policy structure.

### Evaluation outputs

`scripts/run_adaptive_policy_v3_eval.py` writes:

- `outputs/adaptive_policy_v3/summary.json`
- `outputs/adaptive_policy_v3/summary.csv`
- `outputs/adaptive_policy_v3/per_query_results.csv`
- `outputs/adaptive_policy_v3/threshold_sweep.csv`

### Intended status

This is still a rule-based baseline. The main question is whether weighted
signal calibration can finally produce a genuinely selective revise policy with
a better cost-quality tradeoff than both v1 and v2.
