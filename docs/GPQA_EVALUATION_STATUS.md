# GPQA-Diamond: Manuscript-Style Routing Evaluation

**Last updated:** 2026-03-31

This document is the single place for **dataset access**, **routing dataset build**, and **policy evaluation** status for GPQA-Diamond in this repository.

---

## 1. Dataset access (Hugging Face)

**Status:** **Solved** — use the official API with the **config name**:

```python
from datasets import load_dataset
load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
```

A single-argument `load_dataset("Idavidrein/gpqa")` fails with `ValueError: Config name is missing` (not a gating error). See `docs/GPQA_TOKEN_AND_ACCESS_DIAGNOSIS.md`.

Loader code: `src/datasets/gpqa.py` (`OFFICIAL_DATASET`, `OFFICIAL_CONFIG`, `load_gpqa_diamond`, `load_gpqa_diamond_mc`). Fallback Hub: `hendrydong/gpqa_diamond_mc` if official load fails.

---

## 2. Enriched routing dataset build (real paired outcomes + features)

**Purpose:** Same artifact family as GSM8K/MATH500: `reasoning_raw`, `direct_plus_revise` outputs, `reasoning_correct`, `revise_correct`, `revise_helpful`, engineered features including **`unified_confidence_score`**.

**Entry point:** `scripts/run_build_real_routing_dataset.py` with **`--paired-outcomes`** and **`--dataset gpqa_diamond`**.

**Prerequisites:**

- **`OPENAI_API_KEY`** set (GPT-4o-mini or configured `--model`).
- **`HF_TOKEN`** (or Hugging Face CLI login) if your account must authenticate to `Idavidrein/gpqa`.

**Commands:**

```bash
export OPENAI_API_KEY=sk-...
# Optional but recommended for gated Hub access:
export HF_TOKEN=hf_...

python scripts/run_build_real_routing_dataset.py \
  --paired-outcomes \
  --dataset gpqa_diamond \
  --subset-size 198 \
  --output-dir outputs/real_gpqa_routing_dataset \
  --output-dataset-csv data/real_gpqa_diamond_routing_dataset_enriched.csv
```

**Default paths (if you omit overrides):**

| Artifact | Path |
|----------|------|
| Enriched routing CSV | `data/real_gpqa_diamond_routing_dataset_enriched.csv` |
| Run summary JSON | `outputs/real_gpqa_routing_dataset/gpqa_diamond_run_summary.json` |
| Per-query CSV | `outputs/real_gpqa_routing_dataset/gpqa_per_query_outputs.csv` |
| Raw JSONL | `outputs/real_gpqa_routing_dataset/raw_responses.jsonl` |

**Implementation:** `src/data/build_real_routing_dataset.py` — `dataset="gpqa_diamond"` uses **MCQ** prompts (`src/evaluation/strong_baselines_eval.prompt_for_query`) and **`run_direct_plus_revise(..., answer_mode="multiple_choice")`**.

**Mirror-only build (no official Hub):**

```bash
python scripts/run_build_real_routing_dataset.py \
  --paired-outcomes \
  --dataset gpqa_diamond \
  --gpqa-prefer-mirror \
  --subset-size 198 \
  --output-dataset-csv data/real_gpqa_diamond_routing_dataset_enriched.csv
```

---

## 3. Policy evaluation (manuscript-style)

**Entry point:** `scripts/run_real_policy_eval.py`

Evaluates **reasoning_greedy**, **direct_plus_revise**, **adaptive_policy v5–v7**, **confidence_threshold** (if `unified_confidence_score` is present), and **oracle** (revise only when `revise_helpful==1`).

```bash
python scripts/run_real_policy_eval.py \
  --dataset-csv data/real_gpqa_diamond_routing_dataset_enriched.csv \
  --output-dir outputs/real_gpqa_policy_eval \
  --conf-target-cost 1.2
```

**Outputs:**

| File | Description |
|------|-------------|
| `policy_comparison.csv` | All routes: accuracy, avg_cost, revise_rate |
| `per_query_policy_decisions.csv` | Per-query policy choices and correctness |
| `summary.json` | Run summary + comparison |
| `confidence_threshold_sweep.csv` | Present when enriched CSV includes `unified_confidence_score` |

**Code:** `src/evaluation/real_policy_eval.py` (extended for oracle + confidence on enriched rows).

---

## 4. Confidence-threshold baseline registry

`src/baselines/confidence_threshold_router.REGIME_FILES` includes:

```text
"gpqa_diamond_198": "data/real_gpqa_diamond_routing_dataset_enriched.csv"
```

`scripts/run_confidence_baseline.py` **skips** this regime until the CSV exists (warning logged).

---

## 5. Current checkout status (this environment)

| Stage | Status | Evidence |
|-------|--------|------------|
| HF official GPQA load pattern | OK | `docs/GPQA_TOKEN_AND_ACCESS_DIAGNOSIS.md` |
| Routing build (API calls) | **Blocked — invalid API key** | Full paired run **was executed** (198/198 rows processed); **every** completion returned **HTTP 401** `invalid_api_key` (see below). No valid `reasoning_raw` / revise outputs were produced. |
| `data/real_gpqa_diamond_routing_dataset_enriched.csv` | **Not written** | Build ends with `run_status: "PARTIAL"` when all queries error; enriched CSV is only materialized on success path. |
| Policy eval outputs under `outputs/real_gpqa_policy_eval/` | **Not run** | Requires enriched CSV. |

### 5a. Measured failure (2026-03-31)

**Command (from repo root):**

```bash
python3 scripts/run_build_real_routing_dataset.py \
  --paired-outcomes \
  --dataset gpqa_diamond \
  --subset-size 198 \
  --output-dir outputs/real_gpqa_routing_dataset \
  --output-dataset-csv data/real_gpqa_diamond_routing_dataset_enriched.csv \
  --model gpt-4o-mini \
  --timeout 120
```

**Summary JSON:** `outputs/real_gpqa_routing_dataset/gpqa_diamond_run_summary.json` — `run_status: "PARTIAL"`, `num_queries_ok: 0`, `num_queries_error: 198`, `evidence_status: "measured_now"`.

**Root cause:** OpenAI rejected the configured key with **401** and message `invalid_api_key` / “Incorrect API key provided”. In this checkout, **`.env` contains a placeholder** `OPENAI_API_KEY=sk-...` (not a live secret). `python-dotenv` uses `override=False`, so a **real** key can still be supplied via `export OPENAI_API_KEY=...` in the shell before running.

**Artifacts from the failed run (for debugging, not for policy metrics):**

- `outputs/real_gpqa_routing_dataset/gpqa_per_query_outputs.csv` — `status=error` on every row
- `outputs/real_gpqa_routing_dataset/raw_responses.jsonl` — error traces
- `outputs/real_gpqa_routing_dataset/checkpoint.json` — may show `next_index: 198` after a full pass; **delete checkpoint + stale CSV/JSONL** before retrying with a valid key so rows are not skipped as “done”

**Distinction:**

- **Dataset access (HF):** solved — loading `Idavidrein/gpqa` + `gpqa_diamond` works.
- **Routing artifact generation (OpenAI):** **blocked** until a **valid** `OPENAI_API_KEY` is used (replace placeholder in `.env` or export in the shell).

**Success criterion:** After you run §2 with a **valid** API key, §3 produces **real** `measured_now` summaries. Until then, the repository records **exact commands**, **measured error codes**, and **no fabricated GPQA metrics**.

---

## 6. Relation to `run_small_pass.py`

`python scripts/run_small_pass.py` runs **AIME** + **four-regime confidence** tables only. It does **not** run GPQA; see `gpqa_status: NOT_RUN_IN_SMALL_PASS` in `outputs/small_pass/small_pass_run_summary.json` when using that orchestrator.
