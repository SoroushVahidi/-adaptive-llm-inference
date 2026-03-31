# GPQA-Diamond Live Status

Last checked: 2026-03-31 (live snapshot after build completion)

## 1) Build progress / completion

- `outputs/real_gpqa_routing_dataset/checkpoint.json` currently reports:
  - `next_index: 198`
  - `total: 198`
- Interpreting progress: **198 / 198 queries completed**.
- Build run summary now exists and reports `run_status: "COMPLETED"`.

## 2) Core artifacts

### Enriched dataset CSV

- `data/real_gpqa_diamond_routing_dataset_enriched.csv`: **exists**
- Snapshot shape: **198 rows x 82 columns**
- Required columns present and non-empty on all current rows:
  - `reasoning_correct`: present, non-empty = 198
  - `revise_correct`: present, non-empty = 198
  - `revise_helpful`: present, non-empty = 198
  - `unified_confidence_score`: present, non-empty = 198

### Output directory artifacts

`outputs/real_gpqa_routing_dataset/` currently contains:

- `checkpoint.json`
- `gpqa_per_query_outputs.csv`
- `raw_responses.jsonl`
- `provider_metadata.json`

Run summary JSON (e.g., `gpqa_diamond_run_summary.json`): **not present yet** (expected until run finishes).
Run summary JSON: **present**

- `outputs/real_gpqa_routing_dataset/gpqa_diamond_run_summary.json`
- Key fields:
  - `run_status: "COMPLETED"`
  - `num_queries_requested: 198`
  - `num_queries_ok: 198`
  - `num_queries_error: 0`
  - `reasoning_accuracy: 0.4696969696969697`
  - `revise_accuracy: 0.41919191919191917`
  - `revise_helpful_rate: 0.08080808080808081`

## 3) Data quality / completion

- `gpqa_per_query_outputs.csv` snapshot:
- `gpqa_per_query_outputs.csv` snapshot:
  - total rows: 198
  - status counts: `ok=198`, `error=0`
- `raw_responses.jsonl` line count: 198
- Current run appears clean (no error rows).
- Manuscript-grade status: **build artifacts complete** for GPQA-Diamond paired outcomes + features.

## 4) Policy-eval readiness

- `outputs/real_gpqa_policy_eval/`: **not created yet**.
- Ready to run policy eval now? **Yes**.
- Exact reason: enriched CSV and run summary are now complete for all 198 queries.
- Planned command (once build completes):

```bash
python3 scripts/run_real_policy_eval.py \
  --dataset-csv data/real_gpqa_diamond_routing_dataset_enriched.csv \
  --output-dir outputs/real_gpqa_policy_eval \
  --conf-target-cost 1.2
```

## 5) Final grounded summary

- **GPQA status:** `COMPLETE` + `EVAL READY`
- **Completed rows / total rows:** `198 / 198`
- **Manuscript-grade GPQA evidence exists yet?:** `Yes` for build artifacts (paired outcomes + features). Policy-eval outputs are still pending.
- **Exact next command to run:**

```bash
python3 scripts/run_real_policy_eval.py \
  --dataset-csv data/real_gpqa_diamond_routing_dataset_enriched.csv \
  --output-dir outputs/real_gpqa_policy_eval \
  --conf-target-cost 1.2
```
