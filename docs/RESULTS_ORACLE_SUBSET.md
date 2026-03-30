# Results — Oracle Subset Evaluation

**Date/time (UTC):** 2026-03-30T01:53:50Z
**Config:** `configs/oracle_subset_eval_gsm8k.yaml`
**Dataset:** GSM8K bundled test sample, 15 queries
**Query IDs:** gsm8k_test_0, gsm8k_test_1, gsm8k_test_2, gsm8k_test_3, gsm8k_test_4, gsm8k_test_5, gsm8k_test_6, gsm8k_test_7, gsm8k_test_8, gsm8k_test_9, gsm8k_test_10, gsm8k_test_11, gsm8k_test_12, gsm8k_test_13, gsm8k_test_14

## Strategy Accuracy

| Strategy | Accuracy | Correct/Total |
|----------|----------|---------------|
| `direct_greedy` | 0.4667 | 7/15 |
| `reasoning_best_of_3` | 0.6000 | 9/15 |
| `structured_sampling_3` | 0.6000 | 9/15 |
| `direct_plus_verify` | 0.4667 | 7/15 |
| `direct_plus_revise` | 0.6667 | 10/15 |
| `direct_plus_critique_plus_final` | 0.6667 | 10/15 |
| `first_pass_then_hint_guided_reason` | 0.6000 | 9/15 |

## Oracle Metrics

| Metric | Value |
|--------|-------|
| Oracle accuracy (≥1 strategy correct) | 0.7333 |
| Direct (`direct_greedy`) accuracy | 0.4667 |
| **Oracle-direct gap** | **+0.2667** |
| Queries where direct was already optimal | 46.7% |

## Strategy Contributions

### Fixes over `direct_greedy` (queries where direct_greedy was wrong and this strategy was correct)

| Strategy | Queries fixed |
|----------|---------------|
| `direct_plus_verify` | 3 |
| `direct_plus_revise` | 3 |
| `direct_plus_critique_plus_final` | 3 |
| `reasoning_best_of_3` | 2 |
| `structured_sampling_3` | 2 |
| `first_pass_then_hint_guided_reason` | 2 |

### Cheapest correct strategy count

| Strategy | Times cheapest correct |
|----------|------------------------|
| `direct_greedy` | 7 |
| `reasoning_best_of_3` | 0 |
| `structured_sampling_3` | 0 |
| `direct_plus_verify` | 3 |
| `direct_plus_revise` | 1 |
| `direct_plus_critique_plus_final` | 0 |
| `first_pass_then_hint_guided_reason` | 0 |

## Interpretation

- Oracle accuracy of 0.7333 vs direct accuracy of 0.4667 gives
  a gap of +0.2667, showing significant headroom
  for adaptive strategy selection.
- 46.7% of queries are already solved by `direct_greedy` at minimum cost,
  meaning these queries do not benefit from more expensive strategies.
- Strategies that most frequently fix `direct_greedy` failures are the best
  candidates for a selective-escalation or adaptive-routing policy.

## What this means for the paper

The oracle gap of +0.2667 over 15 queries quantifies the upper bound on accuracy
improvement that any oracle adaptive policy could achieve.  The cheapest-correct
distribution shows that low-cost strategies often suffice, while the "fixes" column
identifies which multi-stage or multi-sample strategies add the most value.
These numbers motivate the adaptive compute allocation approach proposed in the paper.
