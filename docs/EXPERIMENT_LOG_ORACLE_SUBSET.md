# Experiment Log — Oracle Subset Evaluation

## Status: BLOCKED

**Date/time (UTC):** 2026-03-28T03:13:47Z

## Command

```bash
python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml
```

## Configuration

- **Config file:** `configs/oracle_subset_eval_gsm8k.yaml`
- **Dataset:** GSM8K (bundled test sample, max 15 queries)
- **Strategies intended:** `direct_greedy`, `reasoning_best_of_3`, `structured_sampling_3`, `direct_plus_verify`, `direct_plus_revise`, `direct_plus_critique_plus_final`, `first_pass_then_hint_guided_reason`
- **Output directory:** `outputs/oracle_subset_eval`

## Blocker

| Field | Value |
|-------|-------|
| Blocker type | `openai_api_key` |
| Detail | Missing OPENAI_API_KEY |

## Why the run could not complete

The live OpenAI API key (`OPENAI_API_KEY`) is not available in this
execution environment.  Without it the `OpenAILLMModel` initialiser raises a
`ValueError` before any queries are processed.

The script stops here and does **not** invent any numeric results.
All outputs are marked `"run_status": "BLOCKED"`.

## Where outputs were written

- `outputs/oracle_subset_eval/summary.json` — BLOCKED sentinel
