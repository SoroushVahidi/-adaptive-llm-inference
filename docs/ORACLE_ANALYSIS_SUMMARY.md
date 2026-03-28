## Oracle subset evaluation summary

### Exact command

`python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml`

### Dataset and subset

- Dataset: `gsm8k`
- Split: `test`
- Subset source: `src/datasets/bundled/gsm8k_test_sample.json`
- Queries evaluated: `20`

### Strategies included

- `direct_greedy`
- `reasoning_greedy`
- `reasoning_best_of_3`
- `structured_sampling_3`
- `direct_plus_verify`
- `direct_plus_revise`
- `direct_plus_critique_plus_final`
- `first_pass_then_hint_guided_reason`
- `strong_direct`

### Key numeric metrics

#### Baselines

- `direct_greedy` accuracy: `0.50`
- `reasoning_greedy` accuracy: `0.65`
- `strong_direct` accuracy: `0.55`

#### Oracle

- Oracle accuracy: `0.75`
- Oracle minus direct gap: `+0.25`
- Oracle minus reasoning_greedy gap: `+0.10`
- Average oracle cost on successful queries: `1.07`

#### Per-strategy usefulness

| Strategy | Accuracy | Wins | Fixes direct failures | Fixes reasoning failures | Cheapest correct | Avg cost |
|---|---:|---:|---:|---:|---:|---:|
| `direct_greedy` | 0.50 | 10 | 0 | 0 | 10 | 1.0 |
| `reasoning_greedy` | 0.65 | 13 | 3 | 0 | 3 | 1.0 |
| `reasoning_best_of_3` | 0.70 | 14 | 4 | 1 | 0 | 3.0 |
| `structured_sampling_3` | 0.65 | 13 | 3 | 0 | 0 | 3.0 |
| `direct_plus_verify` | 0.45 | 9 | 3 | 2 | 1 | 2.0 |
| `direct_plus_revise` | 0.70 | 14 | 4 | 1 | 0 | 2.0 |
| `direct_plus_critique_plus_final` | 0.65 | 13 | 3 | 1 | 0 | 3.0 |
| `first_pass_then_hint_guided_reason` | 0.65 | 13 | 3 | 0 | 0 | 2.0 |
| `strong_direct` | 0.55 | 11 | 4 | 1 | 1 | 1.0 |

#### Global

- Fraction where `direct_greedy` is already oracle-optimal: `0.50`
- Fraction where `reasoning_greedy` is already oracle-optimal: `0.15`
- Queries where no strategy succeeded: `5 / 20` (`0.25`)

### Interpretation

#### Q1. Is one-pass reasoning stronger than direct on this subset?

Yes. `reasoning_greedy` improves from `0.50` to `0.65` at the same average cost (`1.0`), fixing `3` direct failures while remaining cheapest-correct on `3` queries. This matches the new MATH500 evidence that one-pass reasoning is the main useful intervention.

#### Q2. Is extra sampling adding value, or mostly cost?

Mostly cost, with limited marginal benefit. `reasoning_best_of_3` improves only from `0.65` to `0.70` over `reasoning_greedy`, but triples average cost from `1.0` to `3.0`. `structured_sampling_3` adds no gain over `reasoning_greedy` at all (`0.65` vs `0.65`) while still costing `3.0`. On this subset, the extra-sampling story is weak and `structured_sampling_3` looks especially wasteful.

#### Q3. Do multi-stage corrective strategies recover failures that simple reasoning does not?

Only partially.

- `direct_plus_revise` matches the top accuracy (`0.70`) at lower cost than best-of-3 (`2.0` vs `3.0`) and recovers `1` reasoning-greedy failure.
- `direct_plus_verify` is weak overall (`0.45`) but it produces the only unique success on the subset (`gsm8k_test_3`) and recovers `2` reasoning-greedy failures in total.
- `direct_plus_critique_plus_final` and `first_pass_then_hint_guided_reason` do not beat `reasoning_greedy`; they mostly duplicate existing successes with higher cost.

So the only multi-stage strategy with strong practical value here is `direct_plus_revise`, while `direct_plus_verify` may still deserve a narrow “special-case rescue” role because of its unique win.

#### Q4. Which strategies appear redundant given the current evidence?

Most redundant or weak relative to the current lens:

- `structured_sampling_3`: no gain over `reasoning_greedy`, much higher cost
- `first_pass_then_hint_guided_reason`: same accuracy as `reasoning_greedy`, higher cost
- `direct_plus_critique_plus_final`: same accuracy as `reasoning_greedy`, higher cost
- `strong_direct`: only `0.55`, worse than `reasoning_greedy`
- `direct_plus_verify`: weak overall, though not fully redundant because of one unique success

Potentially redundant between the two top performers:

- `reasoning_best_of_3` and `direct_plus_revise` both reach `0.70`
- `direct_plus_revise` is cheaper (`2.0` vs `3.0`), so it currently dominates `reasoning_best_of_3` on cost-efficiency

#### Q5. Based on this run plus the MATH500 diagnostic, what 5–8 strategies should remain in the shortlist?

Recommended shortlist:

1. `direct_greedy`
2. `reasoning_greedy`
3. `direct_plus_revise`
4. `reasoning_best_of_3`
5. `direct_plus_verify`
6. `strong_direct`

Optional keepers if a broader prompt-family comparison is still needed:

7. `direct_plus_critique_plus_final`
8. `first_pass_then_hint_guided_reason`

### Shortlist recommendation

#### Keep

- `direct_greedy` — essential low-cost baseline
- `reasoning_greedy` — strongest low-cost intervention across GSM8K oracle and prior MATH500 diagnostic
- `direct_plus_revise` — best corrective multi-stage candidate so far
- `reasoning_best_of_3` — keep as the main “extra sampling” comparator, even though gains are small
- `direct_plus_verify` — keep narrowly because it had the only unique success on this subset
- `strong_direct` — useful as a simple stronger-model reference

#### Drop or deprioritize

- `structured_sampling_3` — weak on MATH500 and no better than reasoning_greedy here
- `first_pass_then_hint_guided_reason` — duplicates reasoning_greedy at higher cost
- `direct_plus_critique_plus_final` — no clear gain beyond cheaper alternatives

### Output files

- `outputs/oracle_subset_eval/summary.json`
- `outputs/oracle_subset_eval/summary.csv`
- `outputs/oracle_subset_eval/per_query_matrix.csv`
- `outputs/oracle_subset_eval/oracle_assignments.csv`
- `outputs/oracle_subset_eval/pairwise_win_matrix.csv`
