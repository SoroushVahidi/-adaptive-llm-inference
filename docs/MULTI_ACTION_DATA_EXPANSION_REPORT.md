# Multi-action data expansion — diagnosis report

Generated from live runs on branch `cursor/multi-action-routing-pipeline-dc5d` (model: **gpt-4o-mini**, unless noted). CSV and JSON artifacts live under `data/` and `outputs/` (both are **gitignored**).

---

## 1. Datasets that ran successfully

| Dataset | Builder flag | N | Output CSV | Oracle summary | Disagreement JSON |
|--------|--------------|---|------------|----------------|-------------------|
| Hard GSM8K (tail) | `--dataset gsm8k_hard --subset-size 100` | 100 | `data/multi_action_routing_hard_gsm8k_large.csv` | `outputs/multi_action_oracle/hard_gsm8k_large_oracle_summary.json` | `outputs/multi_action_oracle/hard_gsm8k_large_disagreement_analysis.json` |
| MATH500 | `--dataset math500 --subset-size 100` | 100 | `data/multi_action_routing_math500_large.csv` | `outputs/multi_action_oracle/math500_large_oracle_summary.json` | `outputs/multi_action_oracle/math500_large_disagreement_analysis.json` |
| AIME 2024 | `--dataset aime2024 --subset-size 30` | 30 (full HF train) | `data/multi_action_routing_aime2024.csv` | `outputs/multi_action_oracle/aime2024_oracle_summary.json` | `outputs/multi_action_oracle/aime2024_disagreement_analysis.json` |
| GPQA Diamond (public mirror) | `--dataset gpqa --subset-size 100` | 100 | `data/multi_action_routing_gpqa.csv` | `outputs/multi_action_oracle/gpqa_oracle_summary.json` | `outputs/multi_action_oracle/gpqa_disagreement_analysis.json` |

Normalized sidecars:

- `data/aime_2024_normalized.jsonl` — from **HuggingFaceH4/aime_2024** (`train` split; Hub has no `test`).
- `data/gpqa_diamond_normalized.jsonl` — from **aradhye/gpqa_diamond** (`train`).

---

## 2. Blocked or alternate-source attempts (explicit)

### 2.1 GPQA — gated official Hub id

**DATASET BLOCKED (official id):** GPQA (author upload)  
**Source attempted:** `datasets.load_dataset("Idavidrein/gpqa", ...)` (probe during development)  
**Error:** `DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub. You must be authenticated to access it.`  
**Cause:** HuggingFace **auth required** (gated dataset).  
**Fix:** `huggingface-cli login` / set `HF_TOKEN`, accept the dataset terms, or use a public mirror.

**Resolution used:** Loaded **`aradhye/gpqa_diamond`** (fallback: `nichenshun/gpqa_diamond`) — public, 198 rows, schema `problem` + `answer` (letter).

### 2.2 GitHub `idavidrein/gpqa`

Not cloned in this run: the **public HF mirror** satisfied the task without scraping GitHub. To use the canonical repo offline: clone [https://github.com/idavidrein/gpqa](https://github.com/idavidrein/gpqa), convert CSV/JSON to `{question, options, correct_option}`, and pass `--dataset gpqa` with a local loader extension (not required once HF mirror worked).

### 2.3 AIME 2024 Hub splits

**HuggingFaceH4/aime_2024** and **Maxwell-Jia/AIME_2024** both expose only split **`train`** (30 problems). Calling `split="test"` raises `ValueError: Unknown split "test". Should be one of ['train'].`  
**Cause:** schema / split naming on the Hub, not missing data.  
**Fix:** use `split="train"` (as the builder does).

### 2.4 Environment warnings

Hugging Face printed: *unauthenticated requests* — **Cause:** no `HF_TOKEN`. **Fix:** set `HF_TOKEN` for higher rate limits (optional; downloads succeeded).

---

## 3. Label distributions (best_accuracy_action)

From each `*_oracle_summary.json` → `action_win_counts_best_accuracy`:

| Dataset | reasoning_greedy | direct_plus_revise | reasoning_then_revise | self_consistency_3 |
|---------|-----------------|----------------------|------------------------|--------------------|
| hard_gsm8k_large (100) | 98 | 1 | 1 | 0 |
| math500_large (100) | 95 | 5 | 0 | 0 |
| aime2024 (30) | 30 | 0 | 0 | 0 |
| gpqa (100) | 80 | 9 | 6 | 5 |

**Utility labels** (`best_utility_action_lambda_*`): match accuracy winners on these runs except where λ>0 breaks ties (see summary JSON `tie_counts_utility_by_lambda`).

---

## 4. Disagreement statistics

From `*_disagreement_analysis.json` → `disagreement` (pairwise keys are **lexicographic** `action_a|action_b` with fixed action order):

| Dataset | % all four same correctness | % ≥2 actions differ (any pairwise correctness flip) |
|---------|----------------------------|------------------------------------------------------|
| hard_gsm8k_large | 98% | **2.0%** |
| math500_large | 93% | **7.0%** |
| aime2024 | 100% (30/30) | **0%** |
| gpqa | 52% | **48%** |

**Highest action disagreement:** **GPQA** (48% of queries have at least one pairwise correctness difference among the four actions).

---

## 5. Is label degeneracy resolved?

- **Partially yes:** **GPQA** has **four distinct** `best_accuracy_action` winners (80 / 9 / 6 / 5). **MATH500** has **two** classes (95 / 5). **Hard GSM8K** has **three** classes (98 / 1 / 1).
- **Still degenerate:** **AIME 2024** on this model — all actions tied on correctness every time at the 0/1 level; oracle always tie-breaks to **`reasoning_greedy`**. `self_consistency_3` is **strictly worse** on 1/30 queries (accuracy 0.033 vs 0.066) but that does not change the *binary* correctness vector when all are wrong.

---

## 6. Which action wins most often?

- **hard_gsm8k_large:** `reasoning_greedy` (98/100).
- **math500_large:** `reasoning_greedy` (95/100); `direct_plus_revise` on 5.
- **aime2024:** `reasoning_greedy` (30/30 tie-break).
- **gpqa:** `reasoning_greedy` (80/100); largest non-greedy share on this batch.

---

## 7. Classifier training (`run_multi_action_model_eval.py`)

| CSV | Result |
|-----|--------|
| `math500_large` | **Trained** — 2 classes for `best_accuracy_action`; holdout split used (`train_test_split`). |
| `hard_gsm8k_large` | **Trained** — 3 classes; stratify **disabled** for rare classes (`train_test_stratified: false` in JSON). |
| `gpqa` | **Trained** — **4 classes** for `best_accuracy_action`; full multi-action label diversity. |
| `aime2024` | **SKIPPED** — single class `reasoning_greedy` for all targets; same `skip_reason` as before. |

Outputs:

- `outputs/multi_action_models/math500_large_model_results.json`
- `outputs/multi_action_models/hard_gsm8k_large_model_results.json`
- `outputs/multi_action_models/gpqa_model_results.json`
- `outputs/multi_action_models/aime2024_model_results.json` (skipped classifiers)

**Note:** Logistic regression may emit `ConvergenceWarning` on these small feature sets; trees/forests are the primary interpretable models.

---

## 8. Success criteria check

| Criterion | Met? |
|-----------|------|
| Labels ≥2 classes | **Yes** (math500, gsm8k, gpqa) |
| Classifier not skipped | **Yes** for gsm8k/math500/gpqa; **No** for aime2024 |
| Action disagreement >10–20% | **Yes** for GPQA (48%); **No** for AIME (0%); GSM8K 2%; MATH500 7% |
| Meaningful action diversity | **Strongest on GPQA** |

---

## 9. Code changes relevant to this expansion

- `scripts/run_build_multi_action_dataset.py` — `aime2024` / `gpqa` modes, default large CSV names, writes `*_disagreement_analysis.json`.
- `src/evaluation/multi_action_routing.py` — `compute_disagreement_analysis`, `write_disagreement_analysis`, fixed pairwise key ordering.
- `src/datasets/aime2024.py`, `src/datasets/gpqa_diamond.py` — loaders + normalized JSONL writers.
- `src/evaluation/oracle_subset_eval.py` — `answer_mode` + `gold_normalizer` (AIME uses `normalize_math_answer` on gold).
- `src/evaluation/strategy_expansion_eval.py` — MC prompts + `extract_mc_answer` path for GPQA.
- `src/utils/answer_extraction.py` — `extract_mc_answer`.
- `scripts/run_multi_action_model_eval.py` — skip stratify when any class count &lt; 2; log `class_counts_full_dataset`.

---

## 10. How to reproduce

```bash
export OPENAI_API_KEY=...
pip install -e ".[dev]"

python3 scripts/run_build_multi_action_dataset.py --dataset gsm8k_hard --subset-size 100
python3 scripts/run_build_multi_action_dataset.py --dataset math500 --subset-size 100
python3 scripts/run_build_multi_action_dataset.py --dataset aime2024 --subset-size 30 \
  --write-normalized-jsonl data/aime_2024_normalized.jsonl
python3 scripts/run_build_multi_action_dataset.py --dataset gpqa --subset-size 100 \
  --write-normalized-jsonl data/gpqa_diamond_normalized.jsonl

python3 scripts/run_multi_action_model_eval.py --csv data/multi_action_routing_gpqa.csv --dataset-name gpqa
```

For gated GPQA: authenticate to Hugging Face first, or keep using `aradhye/gpqa_diamond`.
