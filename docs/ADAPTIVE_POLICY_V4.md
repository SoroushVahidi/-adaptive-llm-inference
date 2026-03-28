## Adaptive policy v4

### Goal

Improve revise triggering by focusing on question-answer consistency rather than
generic output instability.

### What changed from v3

V3 calibrated the v2 signal family with a weighted threshold, but its best
setting still behaved almost exactly like `reasoning_greedy`.

V4 changes the emphasis:

- keep the same simple strategy set
- keep the rule-based structure
- trigger `direct_plus_revise` mainly from question-answer consistency features

### Constraint-aware signal families

V4 adds lightweight heuristic checks for:

- `answer_type_mismatch_suspected`
- `target_quantity_mismatch_suspected`
- `unit_mismatch_suspected`
- `impossible_sign_suspected`
- `integer_expected_but_noninteger_suspected`
- `percent_or_ratio_mismatch_suspected`
- `answer_not_mentioned_in_final_statement_suspected`
- `constraint_word_conflict_suspected`
- simple bound heuristics when a very obvious total is available

These features are regex/string/rule based only. There is no symbolic solver or
external math library.

### Routing rule

1. If the question looks simple, allow `direct_greedy`.
2. Otherwise run `reasoning_greedy`.
3. Escalate to `direct_plus_revise` only when the weighted combination of
   constraint-aware signals and a small subset of stronger v2 signals crosses
   the revise threshold.

### Why this matches the task

The new focus is not generic instability. It is whether the candidate answer
looks inconsistent with what the question asked for:

- wrong quantity type
- wrong unit
- wrong sign
- wrong constraint word
- missing final restatement of the parsed answer

### Evaluation outputs

`scripts/run_adaptive_policy_v4_eval.py` writes:

- `outputs/adaptive_policy_v4/summary.json`
- `outputs/adaptive_policy_v4/summary.csv`
- `outputs/adaptive_policy_v4/per_query_results.csv`
- `outputs/adaptive_policy_v4/signal_firing_summary.csv`

### Intended status

V4 is still a simple rule-based baseline. The main question is whether
constraint-aware consistency checks make revise more usefully selective than v3.
