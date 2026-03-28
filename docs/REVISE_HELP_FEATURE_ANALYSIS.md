# Revise-Help Feature Analysis

> **Purpose:** Use the newly added target-quantity / wording-trap features to
> measure whether they genuinely separate `revise_helps` queries from
> `direct_already_enough` queries on the GSM8K test set.  This analysis answers
> the key routing question: *does the new feature family have enough signal to
> justify one more hand-crafted router attempt, or should the project move
> directly to learned routing?*

---

## 1. Input Files

| File | Required | Notes |
|------|----------|-------|
| `outputs/oracle_subset_eval/oracle_assignments.csv` | Optional | Primary group-assignment source: `direct_greedy_correct`, `direct_already_optimal`, `cheapest_correct_strategy` |
| `outputs/oracle_subset_eval/per_query_matrix.csv` | Optional | Strategy-level `correct` flags per query; used to identify `direct_plus_revise` outcomes and `unique_other_strategy` cases |
| `outputs/revise_case_analysis/case_table.csv` | Optional | Manual category overrides; any row with a valid group label overrides the computed assignment |
| `src/datasets/bundled/gsm8k_test_sample.json` | Fallback | 20-query bundled sample; used for question text and as a synthetic corpus when no oracle data is available |

All inputs are optional.  When oracle data is missing the analysis still runs
on the bundled GSM8K sample (all queries assigned to `reasoning_enough`) so
that output files are always produced and the pipeline can be tested offline.

---

## 2. Group Definitions

Five mutually exclusive groups are assigned per query:

| Group | Definition |
|-------|-----------|
| `direct_already_enough` | `direct_greedy_correct = 1`; no extra compute needed |
| `revise_helps` | `direct_greedy_correct = 0`; `direct_plus_revise correct = 1` |
| `unique_other_strategy_case` | direct wrong; revise not helpful; a different strategy (e.g., `best_of_n`) was correct |
| `revise_not_enough` | no strategy was correct (oracle failed) |
| `reasoning_enough` | catch-all: direct wrong, revise unknown, oracle says solvable by some means |

The primary comparison of interest is **`revise_helps` vs `direct_already_enough`** because these two groups define the routing decision: does this query need a revise pass or not?

---

## 3. Feature-Rate Comparison Table

The table below is computed from the 20-query bundled GSM8K sample (all
assigned to `reasoning_enough` because no oracle data is available).  Replace
this with real oracle data by running the oracle evaluation and re-running the
script.

> **Note:** With real oracle data, the values in the `revise_helps` and
> `direct_already_enough` columns will be non-zero and meaningful.

| Feature | Source | revise_helps | direct_already_enough | difference |
|---------|--------|--------------|-----------------------|-----------|
| `asks_remaining_or_left` | TQ | — | — | — |
| `asks_total` | TQ | — | — | — |
| `asks_difference` | TQ | — | — | — |
| `asks_rate_or_unit` | TQ | — | — | — |
| `asks_money` | TQ | — | — | — |
| `asks_time` | TQ | — | — | — |
| `has_subtraction_trap_verb` | TQ | — | — | — |
| `has_addition_trap_structure` | TQ | — | — | — |
| `has_multi_operation_hint` | TQ | — | — | — |
| `likely_intermediate_quantity_ask` | TQ | — | — | — |
| `potential_answer_echo_risk` | TQ | — | — | — |
| `has_multi_step_cue` | Base | — | — | — |
| `has_currency_symbol` | Base | — | — | — |
| `has_percent_symbol` | Base | — | — | — |
| `has_fraction_pattern` | Base | — | — | — |
| `has_equation_like_pattern` | Base | — | — | — |
| `repeated_number_flag` | Base | — | — | — |

*TQ = target-quantity features (new); Base = existing query features*

Rates for the 20-query bundled sample (all in `reasoning_enough`):

| Feature | reasoning_enough (n=20) |
|---------|------------------------|
| `asks_remaining_or_left` | 0.10 |
| `asks_total` | 0.25 |
| `asks_difference` | 0.30 |
| `asks_rate_or_unit` | 0.25 |
| `asks_money` | 0.35 |
| `asks_time` | 0.50 |
| `has_subtraction_trap_verb` | 0.20 |
| `has_addition_trap_structure` | 0.35 |
| `has_multi_operation_hint` | 0.40 |
| `likely_intermediate_quantity_ask` | 0.25 |
| `potential_answer_echo_risk` | 0.25 |

The fire rates confirm that the new features are **non-trivially activating**
on the bundled sample (ranging from 10 %–50 %), meaning they are not degenerate
constant-zero features.

---

## 4. Top Separating Features

With real oracle data, the `feature_differences.csv` output ranks features by
`|revise_helps_rate − direct_already_enough_rate|` descending.  The
`example_cases.json` then shows which features actually fired for the top
revise-help queries.

Expected strong separating features (based on the feature design and GSM8K
problem structure):
1. `asks_remaining_or_left` — remainder problems are the canonical revise-help case
2. `has_subtraction_trap_verb` — spent/gave/sold verbs cause greedy to stop early
3. `likely_intermediate_quantity_ask` — multi-sentence, multi-number problems without an anchor
4. `has_multi_operation_hint` — chained operations are the main source of greedy failure
5. `asks_rate_or_unit` — rate×time problems often get unit confusion

---

## 5. Example Queries Where Revise Helps

These are drawn from the bundled GSM8K sample; they illustrate the feature
patterns expected to separate `revise_helps` from `direct_already_enough` once
real oracle data is available.

### Example 1 — "remaining" problem
**Question:**
> *"At the beginning of the day there were 74 apples in a basket. During the day 35
> apples were sold and 11 apples were rotten and had to be thrown away. How many
> apples were left in the basket at the end of the day?"*

**Expected features:**
- `asks_remaining_or_left = 1`  ("left")
- `has_subtraction_trap_verb = 1`  ("sold", "had to be thrown away")
- `has_multi_operation_hint = 1`  ("sold" + "thrown away" = two operation verbs)

**Why revise helps:** Direct greedy may return 74 − 35 = 39 (forgetting to
also subtract the 11 rotten apples).  Revision re-reads the question and
catches the two-step subtraction.

### Example 2 — "total" / rate problem
**Question:**
> *"Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of
> babysitting. How much did she earn?"*

**Expected features:**
- `asks_money = 1`  ("$12", "earn")
- `asks_rate_or_unit = 1`  ("an hour")
- `asks_time = 1`  ("50 minutes")

**Why revise helps:** Direct greedy may return $12 (the hourly rate) instead of
converting 50 minutes to hours and multiplying.  Revision explicitly checks
whether the unit conversion was performed.

### Example 3 — multi-step chain
**Question:**
> *"Betty is saving money for a new wallet which costs $100. Betty has only half of
> the money she needs. Her parents decided to give her $15 for that purpose, and her
> grandparents twice as much as her parents. How much more money does Betty need to
> buy the wallet?"*

**Expected features:**
- `asks_money = 1`  ("$100", "$15")
- `asks_difference = 1`  ("how much more")
- `has_multi_operation_hint = 1`  (multiple arithmetic verbs)
- `potential_answer_echo_risk = 1`  (many numbers in a short final question)

**Why revise helps:** Direct greedy may mis-track the "half of what she needs"
step and report the wrong remainder.  Revision forces a step-by-step
re-computation.

---

## 6. Conclusion: Do the New Features Justify One More Hand-Crafted Router?

**With real oracle data:** Run the analysis script and inspect the
`feature_differences.csv`.  The threshold for recommendation is:

- `max |difference| ≥ 0.20` **AND** ≥ 3 of the top-5 features are from the
  target-quantity family → **one more hand-crafted router attempt is justified**.
- `max |difference| < 0.10` → **move to learned routing now**.

**Without oracle data (current state):** The analysis confirms that the new
target-quantity features are non-degenerate (fire rates 10 %–50 % on GSM8K)
and are semantically motivated by the documented revise-help failure patterns.
The `insufficient_data` recommendation is the correct output until oracle
evaluation is run.

The feature design is grounded in two observations from the GSM8K error
taxonomy:
1. **Subtraction-final problems** (`asks_remaining_or_left`,
   `has_subtraction_trap_verb`) are the most common source of direct-greedy
   failure.
2. **Multi-step chains with rate or unit language** (`asks_rate_or_unit`,
   `has_multi_operation_hint`) are the second most common.

If real oracle data shows `revise_helps_rate` ≥ 0.3 on these features and
`direct_already_enough_rate` ≤ 0.1, the features provide enough signal for a
precision-focused hand-crafted router.  Otherwise, the correct path is to
train a lightweight classifier on the full combined feature vector.

---

## 7. Running the Analysis

```bash
# Run analysis (offline, with bundled GSM8K fallback):
python3 scripts/run_revise_help_feature_analysis.py

# Run with real oracle data (after oracle evaluation):
# Outputs from run_oracle_subset_eval.py must be present in:
#   outputs/oracle_subset_eval/oracle_assignments.csv
#   outputs/oracle_subset_eval/per_query_matrix.csv
python3 scripts/run_revise_help_feature_analysis.py
```

---

## 8. Output Files

| File | Description |
|------|-------------|
| `outputs/revise_help_feature_analysis/group_feature_rates.csv` | Per-group boolean feature rates and numeric feature means |
| `outputs/revise_help_feature_analysis/feature_differences.csv` | Per-feature (revise_helps − direct_already_enough), sorted by \|difference\| |
| `outputs/revise_help_feature_analysis/query_feature_table.csv` | Per-query feature values + group label |
| `outputs/revise_help_feature_analysis/example_cases.json` | Top example queries from `revise_helps`, with fired feature lists |

---

*Module:* `src/analysis/revise_help_feature_analysis.py`  
*Script:* `scripts/run_revise_help_feature_analysis.py`  
*Tests:* `tests/test_revise_help_feature_analysis.py`  
*Related:* `docs/TARGET_QUANTITY_FEATURES.md`, `docs/FEATURE_GAP_ANALYSIS.md`
