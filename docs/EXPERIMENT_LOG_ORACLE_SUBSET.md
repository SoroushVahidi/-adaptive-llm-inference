# Experiment Log — Oracle Subset Evaluation

## Status: COMPLETED

**Date/time (UTC):** 2026-03-30T01:53:50Z

## Command

```bash
python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml
```

## Configuration

- **Config file:** `configs/oracle_subset_eval_gsm8k.yaml`
- **Dataset:** GSM8K bundled test sample (split: test, max_samples: 15)
- **Model:** `gpt-4o-mini`
- **Output directory:** `outputs/oracle_subset_eval`

## Strategies evaluated

- `direct_greedy`
- `reasoning_best_of_3`
- `structured_sampling_3`
- `direct_plus_verify`
- `direct_plus_revise`
- `direct_plus_critique_plus_final`
- `first_pass_then_hint_guided_reason`

## Query subset (15 queries)

`gsm8k_test_0`, `gsm8k_test_1`, `gsm8k_test_2`, `gsm8k_test_3`, `gsm8k_test_4`, `gsm8k_test_5`, `gsm8k_test_6`, `gsm8k_test_7`, `gsm8k_test_8`, `gsm8k_test_9`, `gsm8k_test_10`, `gsm8k_test_11`, `gsm8k_test_12`, `gsm8k_test_13`, `gsm8k_test_14`

## Output files

| File | Path |
|------|------|
| `per_query_matrix_csv` | `outputs/oracle_subset_eval/per_query_matrix.csv` |
| `summary_json` | `outputs/oracle_subset_eval/summary.json` |
| `summary_csv` | `outputs/oracle_subset_eval/summary.csv` |
| `oracle_assignments_csv` | `outputs/oracle_subset_eval/oracle_assignments.csv` |
| `pairwise_win_matrix_csv` | `outputs/oracle_subset_eval/pairwise_win_matrix.csv` |

## Runtime notes

Run completed successfully.  See `docs/RESULTS_ORACLE_SUBSET.md` for metrics.
