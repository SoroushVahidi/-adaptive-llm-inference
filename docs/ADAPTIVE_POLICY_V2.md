## Adaptive policy v2

### Goal

Improve `adaptive_policy_v1` so that revise triggers selectively rather than on
nearly every query.

### What changed from v1

V1 treated long reasoning traces with many intermediate numbers as unstable.
On the GSM8K subset, that made `direct_plus_revise` fire on all 20 queries.

V2 replaces that broad trigger with narrower math-specific violation signals.

### Violation signals

V2 keeps the policy rule-based and lightweight. It uses string/regex signals
from the first reasoning output:

- `final_answer_missing_or_unclear`
- `parse_failure`
- `malformed_output`
- `uncertainty_phrase_present`
- `too_many_intermediate_numbers_without_clear_final`
- `contradiction_like_phrase_present`
- `target_mismatch_suspected`
- `unit_mismatch_suspected`
- `impossible_value_suspected`

These do not use any symbolic math library or learned classifier.

### Routing rules

1. If the question looks simple, use `direct_greedy`.
2. Otherwise, run `reasoning_greedy`.
3. Trigger `direct_plus_revise` only if one or more strong violation signals
   fire.
4. Optionally allow `reasoning_best_of_3` only for a narrow severe-instability
   case.
5. Optionally allow `strong_direct` only when configured and clearly justified.

### Why this matches the oracle findings

The GSM8K oracle analysis suggested:

- `reasoning_greedy` is the main cheap improvement over direct
- `direct_plus_revise` is the strongest corrective strategy
- extra sampling is only a marginal gain
- revise should therefore be reserved for cases where the reasoning trace looks
  specifically suspect

V2 tries to encode exactly that: reasoning by default, revise only for
recognizable violation patterns.

### Evaluation outputs

`scripts/run_adaptive_policy_v2_eval.py` writes:

- `outputs/adaptive_policy_v2/summary.json`
- `outputs/adaptive_policy_v2/summary.csv`
- `outputs/adaptive_policy_v2/per_query_results.csv`

### Intended status

This is still a rule-based baseline. The main question for v2 is not only raw
accuracy, but whether revise became selective and whether that improved the
cost-quality tradeoff relative to v1.
