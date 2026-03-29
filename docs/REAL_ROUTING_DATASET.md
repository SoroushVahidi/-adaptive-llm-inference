# Real GSM8K Routing Dataset (First Build)

## What this dataset is

`data/real_gsm8k_routing_dataset.csv` is intended to be the first **real query-level routing table** for GSM8K.
Each row corresponds to one real GSM8K question and records:

- question text + gold answer,
- `reasoning_greedy` predicted answer,
- `direct_plus_revise` predicted answer,
- correctness for each strategy,
- `revise_helpful` label,
- engineered feature columns (query, target-quantity, constraint, number-role, calibration, self/step verification, unified error/confidence).

Label definition:

- `revise_helpful = 1` iff `reasoning_correct == 0` and `revise_correct == 1`; else `0`.

## Why this is better than synthetic consistency benchmark rows

The synthetic benchmark is still useful for controlled diagnostics, but it is not the preferred source for learned routing training. This dataset is tied to **real GSM8K query-level strategy executions** with explicit per-query outcomes, making it better aligned with the routing training target.

## Current run status

**Completed run (100 queries):** see `docs/REAL_GSM8K_ROUTING_STUDY.md` and `outputs/real_routing_dataset/gsm8k_subset_run_summary.json` (**measured_now**).

Run command:

```bash
python3 scripts/run_build_real_routing_dataset.py --subset-size 100
```

Use `--gsm8k-data-file path/to/normalized.jsonl` when a local normalized file exists; otherwise the builder uses **HuggingFace** `openai/gsm8k` for subsets larger than the bundled 20-row JSON.

In this environment, the run is blocked when `OPENAI_API_KEY` is missing or model access fails. In blocked mode, we write:

- `outputs/real_routing_dataset/gsm8k_subset_run_summary.json`

and stop without fabricating per-query outputs.

## Evidence status discipline

- real strategy outputs and labels: `measured_now` (only when inference succeeds)
- blocked environment checks: `blocked`
- synthetic fallback conclusions: `exploratory_only`
- paper-level claims from this dataset build alone: not `claim_ready` yet
