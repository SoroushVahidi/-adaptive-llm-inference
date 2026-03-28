## Adaptive policy v1

### Goal

Implement the first simple version of `pi(x, z)` as an interpretable rule-based
router over existing strategies.

### Policy rules

1. If the question looks simple, use `direct_greedy`.
2. Otherwise, use `reasoning_greedy`.
3. After the first reasoning pass, if the output looks unstable, escalate to
   `direct_plus_revise`.
4. Optionally, for very hard queries with unstable many-number reasoning output,
   allow `reasoning_best_of_3`.
5. Optionally, for high-complexity queries with severely unstable reasoning
   output, allow `strong_direct`.

### Question-side features z(x)

The current repository branch does not contain a standalone
`precompute_features.py`, so v1 computes the same style of lightweight features
online:

- `num_numeric_mentions`
- `question_length_words`
- `question_length_chars`
- `has_multi_step_cue`

These are used to form a simple notion of "simple question":

- low numeric mention count
- no multi-step cue
- short text

### First-pass output features

After `reasoning_greedy`, the policy inspects the first-pass output using the
same cheap parseability heuristics already used in the selective-escalation
path:

- parse failure
- malformed output
- uncertainty phrases
- too many numbers in the output

These are used to define whether the first reasoning pass looks unstable.

### Why this matches the oracle findings

The GSM8K oracle subset evaluation showed:

- `reasoning_greedy` is a strong low-cost improvement over `direct_greedy`
- `structured_sampling_3` is not attractive
- `reasoning_best_of_3` gives only small marginal gains for much higher cost
- `direct_plus_revise` is the strongest corrective multi-stage strategy

So v1 intentionally encodes the simplest plausible policy suggested by the
oracle evidence:

- use direct only for clearly easy questions
- default to one-pass reasoning on harder questions
- use revise as the main correction path
- keep more expensive fallbacks rare and optional

### Evaluation outputs

`scripts/run_adaptive_policy_eval.py` writes:

- `outputs/adaptive_policy_v1/summary.json`
- `outputs/adaptive_policy_v1/summary.csv`
- `outputs/adaptive_policy_v1/per_query_results.csv`

### Intended status

This is a v1 baseline, not a learned policy. Clarity and interpretability are
the main goals.
