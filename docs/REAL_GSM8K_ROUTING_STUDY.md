# Real GSM8K Routing Study (First End-to-End Run)

**Evidence status:** **measured_now** for numbers in this document when they match committed artifacts under `outputs/real_routing_dataset/`, `outputs/real_policy_eval/`, and `data/real_gsm8k_routing_dataset.csv`. **exploratory_only** for generalization beyond this single 100-query draw.

---

## Pre-run checks

| Check | Result (this workspace) |
|-------|-------------------------|
| `OPENAI_API_KEY` | Present |
| `GEMINI_API_KEY` | Present (unused; OpenAI path used) |
| `scikit-learn` | Installed for learned-router eval (`pip install` in env; also listed under `[project.optional-dependencies]` dev) |
| Normalized GSM8K file | Default `data/gsm8k_uploaded_normalized.jsonl` **not** present → loader used **HuggingFace** `openai/gsm8k` test split |
| Output dirs | Writable under `outputs/` and `data/` |

## Provider and model

- **Provider:** OpenAI (existing `OpenAILLMModel`)
- **Model:** `gpt-4o-mini`
- **Calls per query:** 3 (1× `reasoning_greedy` + 2× `direct_plus_revise` pipeline) → **300** chat completions for N=100

## Dataset build

- **Command:** `python3 scripts/run_build_real_routing_dataset.py --subset-size 100 --gsm8k-data-file /nonexistent.jsonl`  
  (forces HF path when local file missing; bundled JSON has only 20 rows so subset 100 triggers HF.)
- **Rows completed:** **100 / 100** (`run_status: COMPLETED`)
- **Data source:** `huggingface_openai_gsm8k_test` (annotated in `outputs/real_routing_dataset/gsm8k_subset_run_summary.json`)

## Outputs

| Artifact | Path |
|----------|------|
| Routing dataset (features + labels) | `data/real_gsm8k_routing_dataset.csv` |
| Run summary | `outputs/real_routing_dataset/gsm8k_subset_run_summary.json` |
| Per-query table (truncated reasoning in CSV) | `outputs/real_routing_dataset/gsm8k_per_query_outputs.csv` |
| Full rows + raw | `outputs/real_routing_dataset/raw_responses.jsonl` |
| Provider metadata | `outputs/real_routing_dataset/provider_metadata.json` |

**Checkpointing:** Each query appends to `raw_responses.jsonl` and rewrites CSVs; `outputs/real_routing_dataset/checkpoint.json` tracks `next_index`.

## Labels

- `reasoning_correct` / `revise_correct`: numeric match vs gold (`Decimal` normalization).
- `revise_helpful` = 1 iff `reasoning_correct == 0` and `revise_correct == 1`.

## Headline metrics (N=100)

From `gsm8k_subset_run_summary.json`:

- **reasoning_greedy accuracy:** 0.90  
- **direct_plus_revise accuracy:** 0.92  
- **revise_helpful count:** **2** (rate **0.02**)

## Policy evaluation (same 100 queries)

Script: `python3 scripts/run_real_policy_eval.py`

| Route | Accuracy | Avg cost (proxy) | Revise rate |
|-------|----------|------------------|-------------|
| reasoning_greedy | 0.90 | 1.0 | 0.0 |
| direct_plus_revise | 0.92 | 2.0 | 1.0 |
| adaptive_policy_v5 | 0.92 | 1.29 | 0.29 |
| adaptive_policy_v6 | 0.92 | 1.18 | 0.18 |
| adaptive_policy_v7 | 0.92 | 1.30 | 0.30 |

**V7 vs V6 accuracy:** **tie** (0.92 both); V7 uses **more** revise on this slice.

## Strongest cheap baseline (measured)

**reasoning_greedy** at **0.90** accuracy, **1.0** cost proxy.

## Strongest corrective baseline (measured)

**direct_plus_revise** at **0.92** accuracy, **2.0** cost proxy.

## Limitations

- Single model; single 100-query slice; **2** positive `revise_helpful` labels → learned router **cannot** estimate recall meaningfully (see `docs/REAL_ROUTING_MODEL_RESULTS.md`).
- `direct_plus_revise` in this repo uses **direct+revise** (not reasoning-first revise); paired comparison is still a valid **routing** probe but not identical to “revise after reasoning_greedy.”
