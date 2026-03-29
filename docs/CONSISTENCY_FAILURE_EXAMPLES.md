# Consistency Failure Examples (Grounded Extraction Attempt)

## Status

**Blocked by missing input artifacts.**

I attempted to extract real GSM8K cases from the exact files requested:

- `outputs/oracle_subset_eval/per_query_matrix.csv`
- `outputs/oracle_subset_eval/oracle_assignments.csv`
- `outputs/adaptive_policy_v1/per_query_results.csv`
- `outputs/adaptive_policy_v2/per_query_results.csv`
- `outputs/adaptive_policy_v3/per_query_results.csv`
- `outputs/adaptive_policy_v4/per_query_results.csv`

None of these files are present in the current repository checkout, and no equivalent CSVs were found elsewhere under `/workspace`.

Because of that, I cannot produce a **grounded** list of 5–10 real query/prediction examples without inventing data.

---

## What was checked

### 1) Required paths
All required files above are missing in this checkout.

### 2) Broader file search
No `per_query_matrix.csv`, `oracle_assignments.csv`, or adaptive-policy `per_query_results.csv` files were found anywhere in `/workspace`.

---

## Extracted examples

**Examples found: 0 (blocked by missing outputs).**

No real cases can be verified for:

- `reasoning_greedy` wrong + superficially plausible answer,
- then fixed by `direct_plus_revise` or another strategy.

---

## What is needed to complete this analysis

Please provide (or commit/copy into this workspace) the six output files listed above.  
Once available, I can immediately produce:

1. a 5–10 case grounded table with `question_id`, question text, gold answer, wrong `reasoning_greedy` answer, fixing strategy answer,
2. failure categories (target mismatch, intermediate-as-final, rate-vs-total, total-vs-remaining, floor/ceiling, hidden constraints, ratio constraints),
3. counts and most common failure type,
4. top 2–3 strongest examples.

---

## Notes

The bundled GSM8K sample (`src/datasets/bundled/gsm8k_test_sample.json`) is available, but without the requested oracle/adaptive prediction CSVs, it does not provide the model-output evidence needed for this task.
