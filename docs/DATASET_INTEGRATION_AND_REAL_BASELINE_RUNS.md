# Dataset integration and real baseline runs

This document records **what was accessed**, **what failed**, **loaders added**, **real-model runs completed**, **action disagreement**, and **remaining blockers** for the strong-baselines / routing experiments.

## 1. Datasets successfully accessed

| Dataset | Source used | Normalized cache written |
|--------|-------------|---------------------------|
| **AIME 2024** | `HuggingFaceH4/aime_2024` (split `train`, 30 rows) | `data/aime_2024_normalized.jsonl` (30 lines, `{"question","answer"}`) |
| **GPQA Diamond (MCQ)** | `hendrydong/gpqa_diamond_mc` on HuggingFace (split `test`, 198 rows) | `data/gpqa_diamond_normalized.jsonl` (198 lines: `question`, `choices`, `answer` letter) |
| **Hard GSM8K** | Existing loader (longest questions from GSM8K test) | (optional) use `gsm8k_data_file` in config |
| **MATH500** | `HuggingFaceH4/MATH-500` | (optional) local JSON via `math500_data_file` |

Loader modules:

- `src/datasets/aime2024.py` — `load_aime2024(...)`; optional `data_file` for JSON/JSONL; writes normalized JSONL when loading from HF.
- `src/datasets/gpqa.py` — `load_gpqa_diamond(...)`; parses `(A)…(D)` blocks from HF `problem` text; gold letter from `\\boxed{X}` in `solution`.

Multiple-choice parsing for model outputs: `src/utils/mcq_answer.py` (`extract_mcq_letter`, etc.).

## 2. Datasets / sources that failed or were bypassed

### 2.1 Official GPQA GitHub ZIP (idavidrein/gpqa)

- **Source attempted:** `https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip` (same file in cloned repo).
- **Exact error when extracting:** `skipping: dataset/gpqa_diamond.csv unable to get password` (InfoZIP / unzip).
- **Root cause:** The ZIP entries use **encryption** (`flag_bits` includes the encryption bit). Standard unzip without a password cannot read CSVs.
- **Cause category:** *Dataset packaging / access* — password-protected archive; not a HuggingFace token issue.
- **Fix:** Obtain the **ZIP password** from the GPQA authors / paper instructions, then unzip locally and point `gpqa_data_file` at the extracted CSV, **or** continue using the public HF mirror `hendrydong/gpqa_diamond_mc` (used here).

### 2.2 Raw CSV path (404)

- **URL:** `https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset/gpqa_diamond.csv`
- **Exact error:** HTTP 404.
- **Fix:** Use `dataset.zip` or HuggingFace mirrors, not that raw path.

### 2.3 HuggingFace gated `Idavidrein/gpqa`

- **Exact error:** `DatasetNotFoundError(... gated dataset ... ask for access)`.
- **Fix:** Accept gate on the Hub and `huggingface-cli login`, or use `hendrydong/gpqa_diamond_mc` (ungated).

## 3. Pipeline changes for new data

- **`Query`** (`src/datasets/gsm8k.py`): optional `choices` for MCQ; `answer` is numeric string or **single letter** for MCQ.
- **`strong_baselines_eval`**: `task_type` in `{"numeric","math","mcq"}`; MCQ prompts append labeled options; grading uses letter extraction + normalization.
- **`self_consistency`**: `majority_vote_self_consistency(..., use_mcq=True)` votes on letters.
- **Disagreement analysis**: `compute_disagreement_analysis` + `outputs/baselines/{dataset}_disagreement_analysis.json`.
- **`scripts/run_strong_baselines.py`**: datasets `aime2024`, `gpqa_diamond`; **`OpenAILLMModel`** (fixed import); config `evaluate.*` to skip expensive router sweeps; `confidence_router_thresholds` to shorten sweeps.

## 4. Baselines run successfully (real model)

**Config:** `configs/strong_baselines_real_smoke.yaml`  
**Model:** `gpt-4o-mini` via `OPENAI_API_KEY` (verified present in this environment).  
**Scope:** `max_samples: 2` per dataset; **compute ladder only** (`evaluate.confidence_router/output_router/best_route_style: false`) so the run completes in reasonable time (~6 minutes wall clock for 4 datasets × 2 queries × 5 actions = 40 forward passes plus self-consistency multi-samples).

**Outputs updated under `outputs/baselines/`:**

- `{hard_gsm8k,math500,aime2024,gpqa_diamond}_compute_ladder.json`
- Same `*_disagreement_analysis.json`
- `final_baseline_summary.csv`, `dataset_rollup.csv`, `run_log.json`

**Not run in this smoke (intentional):** confidence router CSV sweeps, output-aware router JSON, BEST-route-style JSON — enable in config when you accept **multi-hour** API cost for larger `max_samples`.

**Full-scale config template:** `configs/strong_baselines_real.yaml` (still has routers enabled; expect long runtime and possible timeouts without higher `timeout_seconds` and batching).

## 5. Accuracy / cost summary (this smoke, 2 queries each)

From `outputs/baselines/dataset_rollup.csv` (static ladder only; adaptive columns `n/a` because routers were skipped):

| Dataset | Best static method | Accuracy | Avg cost (proxy) |
|---------|-------------------|----------|------------------|
| hard_gsm8k | direct_plus_revise | 1.0 | 2.0 |
| math500 | reasoning_greedy | 0.0 | 1.0 |
| aime2024 | reasoning_greedy | 0.0 | 1.0 |
| gpqa_diamond | self_consistency_3 | 0.5 | 3.0 |

*These numbers are **not** stable at N=2; they only prove end-to-end integration and API execution.*

## 6. Action disagreement summary (same smoke)

| Dataset | % queries all actions same correctness | % ≥2 actions differ | Multi-action classifier trainable (`≥2` best-accuracy labels) |
|---------|----------------------------------------|---------------------|----------------------------------------------------------------|
| hard_gsm8k | 0% | 100% | **No** (only `direct_plus_revise` as cheapest correct on both) |
| math500 | 100% | 0% | **No** (all wrong → single `none_correct` bucket) |
| aime2024 | 100% | 0% | **No** (all wrong) |
| gpqa_diamond | 50% | 50% | **Yes** (`reasoning_then_revise` vs `none_correct`) |

Pairwise rates: see each `*_disagreement_analysis.json`.

## 7. Do AIME / GPQA increase label diversity vs GSM8K-only?

- **At this microscopic N:** AIME (2 Q) shows **no** diversity in best-action labels (all incorrect). GPQA shows **two** best-accuracy classes including `none_correct`, so the **degeneracy of a single “always X” label can break** when the model sometimes gets a hard MCQ right with a stronger action.
- **Hypothesis for larger N:** GPQA and AIME should increase **pairwise disagreement** and **multi-class best-action labels** relative to easy GSM8K slices, but this must be re-measured with `max_samples` in the hundreds and full router evaluation.

## 8. Bottleneck status

- **Reduced:** New datasets are **integrated and runnable**; real API **ladder** runs work; **disagreement metrics** are automated.
- **Still open:** (1) **Scale** — need larger `max_samples` and overnight jobs. (2) **Router cost** — full confidence sweeps multiply calls; use `confidence_router_thresholds` and `evaluate` flags. (3) **Official GPQA files** — encrypted ZIP unless password obtained. (4) **Rate limits / timeouts** — long runs may need retries (not yet in `OpenAILLMModel`).

## Explicit Blockers Encountered

| Step | Dataset / component | Exact error | Cause | Next step to fix |
|------|---------------------|------------|-------|------------------|
| GPQA official ZIP | idavidrein/gpqa `dataset.zip` | `unable to get password` on unzip | Password-protected / encrypted ZIP | Get author password or keep using `hendrydong/gpqa_diamond_mc` |
| Raw GitHub CSV | gpqa_diamond.csv path | HTTP 404 | File not at that raw URL | Use zip or HF mirror |
| Gated HF | `Idavidrein/gpqa` | `DatasetNotFoundError` … gated | No Hub access token / not accepted | `huggingface-cli login` + accept terms, or use mirror |
| Full real run (earlier attempt) | all | Process exceeded time budget / no output flush | Too many API calls (default 19×2 confidence sweeps × datasets) | Disable routers or shrink thresholds / `max_samples` (as in `strong_baselines_real_smoke.yaml`) |
