# AIME-2024 and GPQA: Small Pass Notes (Updated)

**Date:** 2026-03-31 (revised)

---

## Summary

| Item | Status | Notes |
|------|--------|--------|
| AIME-2024 policy evaluation | ✅ Completed | `outputs/small_pass/` |
| Confidence baseline (four main regimes) | ✅ Completed | `outputs/baselines/confidence_threshold/` |
| GPQA-Diamond **dataset access** (HF) | ✅ **Solved** | Use `load_dataset("Idavidrein/gpqa", "gpqa_diamond", ...)`. See `docs/GPQA_TOKEN_AND_ACCESS_DIAGNOSIS.md`. |
| GPQA-Diamond **enriched routing CSV + policy eval** | 🔴 **Blocked on OpenAI auth (measured)** | Hub/load is fine. A full **198-query** paired build was run; **all** LLM calls returned **HTTP 401** `invalid_api_key` because the configured key was invalid (e.g. placeholder `sk-...` in `.env`). See **`docs/GPQA_EVALUATION_STATUS.md` §5a** — replace with a valid key and clear stale checkpoints before retry. |
| Full GPQA (non-Diamond configs) | Optional / out of scope | Main manuscript uses Diamond-scale slice; other configs exist on the Hub but are not wired into this pipeline. |

---

## GPQA: What changed vs the old “blocked” story

**Previously:** Docs treated GPQA as unreachable because `load_dataset("Idavidrein/gpqa")` was called **without** a config name (which raises `ValueError`), or Hub access was unclear.

**Now:**

1. **Official data path** is `Idavidrein/gpqa` + config **`gpqa_diamond`** + split **`train`** (198 rows). Implemented in `src/datasets/gpqa.py` via `load_gpqa_diamond(..., prefer_official=True)`.

2. **Normalized JSONL only** (`data/gpqa_diamond_normalized.jsonl`) is insufficient for routing evaluation: it has no `reasoning_raw` or labels. That was correctly described as “dataset only.”

3. **Manuscript-grade routing rows** are produced by the **same** paired pipeline as GSM8K: `src/data/build_real_routing_dataset.py` with `dataset="gpqa_diamond"` (MCQ prompts + `direct_plus_revise`). Outputs **`data/real_gpqa_diamond_routing_dataset_enriched.csv`** when using the recommended CLI in `docs/GPQA_EVALUATION_STATUS.md`.

4. **Remaining blocker** is **not** Hugging Face access in principle — it is **successful OpenAI completions** during the paid LLM build. A trial run completed all indices but **0/198** queries succeeded (401 invalid key); see `outputs/real_gpqa_routing_dataset/gpqa_diamond_run_summary.json` and **`docs/GPQA_EVALUATION_STATUS.md` §5a**. Policy metrics are intentionally **not** committed until a valid key produces real rows.

---

## Small-pass orchestrator (`run_small_pass.py`)

GPQA is **not** executed inside `python scripts/run_small_pass.py` (that script targets AIME + main-regime confidence). The summary JSON records:

- `"gpqa_status": "NOT_RUN_IN_SMALL_PASS"`
- `"gpqa_note"` → pointer to `docs/GPQA_EVALUATION_STATUS.md`

---

## No fake results

The repository does **not** ship fabricated GPQA routing accuracies. After you run the build + eval commands in `docs/GPQA_EVALUATION_STATUS.md`, you may commit the resulting CSVs/JSON under the usual `data/` and `outputs/` allowlist in `.gitignore`.
