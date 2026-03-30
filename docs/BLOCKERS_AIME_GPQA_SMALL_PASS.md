# Blockers: AIME-2024 and GPQA Small Experiment Pass

**Date:** 2026-03-30  
**Pass:** Small manuscript-strengthening experiment pass

---

## Summary

| Item | Status | Blocker? |
|------|--------|----------|
| AIME-2024 policy evaluation | ✅ COMPLETED | None |
| AIME-2024 confidence baseline | ✅ COMPLETED | None |
| Confidence baseline (main regimes) | ✅ COMPLETED | None |
| GPQA-Diamond policy evaluation | ❌ BLOCKED | Missing routing features |
| GPQA-Diamond confidence baseline | ❌ BLOCKED | Missing routing features |
| Full GPQA (non-Diamond) | ❌ BLOCKED | Hub gating + missing routing features |

---

## Blocker 1: GPQA-Diamond — Missing Routing Features

### What was attempted

Evaluate the routing policies (v5/v6/v7) and the confidence-threshold baseline
on the GPQA-Diamond subset.

### What failed

The committed file `data/gpqa_diamond_normalized.jsonl` contains only:
- `question`: question text
- `choices`: list of 4 answer choices (A–D)
- `answer`: correct answer label ("A" in all 198 rows — normalized to always-A)
- `id`: HuggingFace row ID

It does **not** contain:
- `reasoning_raw`: first-pass model response (needed for all routing policies)
- `reasoning_correct`: correctness of first-pass (needed for evaluation)
- `revise_correct`: correctness after revision (needed for evaluation)
- `revise_helpful`: whether revision helped (needed for oracle)
- `unified_confidence_score`: confidence signal (needed for confidence baseline)
- Any of the 80+ routing features computed by `src/features/`

### Root cause

The GPQA Diamond data was normalized from a public HuggingFace source
(`hendrydong/gpqa_diamond_mc`, 198 rows) but was never processed through the
routing dataset pipeline (`src/data/build_real_routing_dataset.py`), which
calls GPT-4o-mini to generate model responses and computes all features.

### What URL/path was tried

- `data/gpqa_diamond_normalized.jsonl` — exists, 198 rows, but only 4 columns
- No API calls were made in this pass (as per pass constraints)

### What a manual step would unblock this

```bash
# Requires OPENAI_API_KEY in environment
# Estimated: ~198 × 2 API calls (reasoning + revision) at gpt-4o-mini rate
python scripts/run_build_real_routing_dataset.py \
  --dataset gpqa \
  --subset-size 100 \
  --output-dataset-csv data/real_gpqa_routing_dataset.csv \
  --output-dir outputs/real_gpqa_routing

# After that, run the policy eval:
python scripts/run_real_policy_eval.py \
  --dataset-csv data/real_gpqa_routing_dataset.csv \
  --output-dir outputs/gpqa_policy_eval
```

**Estimated cost:** ~200 API calls × 2 passes ≈ 400 gpt-4o-mini completions.
At current pricing (~$0.0001/1K tokens, ~500 tokens each), this is well under $1.

**Time estimate:** ~10–15 minutes with standard API rate limits.

---

## Blocker 2: Full GPQA (Non-Diamond) — Hub Gating

### What failed

The official GPQA dataset on HuggingFace Hub (`Idavidrein/gpqa`) is gated.
As documented in `docs/GPQA_ACCESS_CHECK.md`, access was not granted to this
account at the time of the pass.

### What was tried

- `load_dataset("Idavidrein/gpqa")` → `DatasetNotFoundError` (gated)
- GitHub raw CSV URL → HTTP 404
- Fallback: `hendrydong/gpqa_diamond_mc` → ✅ accessible (198 rows, used for normalization)

### What would unblock this

1. Request access on https://huggingface.co/datasets/Idavidrein/gpqa
2. Once granted, `load_dataset("Idavidrein/gpqa", trust_remote_code=True)` with
   `HF_TOKEN` in environment
3. Then run the routing dataset pipeline as above

---

## What Was Completed Despite Blockers

Despite the GPQA blockers, the following was completed in this pass:

1. **AIME-2024 policy evaluation** — all routing policies evaluated on the 30
   committed AIME queries. Results: routing does not help (revise_helpful=0 for
   all queries). See `docs/SMALL_EXPERIMENT_PASS_AIME_GPQA.md`.

2. **Confidence-threshold baseline** — evaluated on all four main manuscript
   regimes. Strong competitive baseline: matches or exceeds best adaptive policy
   accuracy at similar cost. See `docs/CONFIDENCE_ROUTER_BASELINE.md`.

3. **Combined comparison table** — `outputs/paper_tables_small_pass/small_pass_combined_comparison.csv`
   shows all strategies side-by-side including confidence baseline.

4. **Blocker infrastructure** — this document and the evaluation code correctly
   report GPQA status as BLOCKED rather than producing fake results.

---

## No Fake Results

As required by the pass honesty rule, no GPQA routing evaluation results were
fabricated. The `outputs/small_pass/small_pass_run_summary.json` correctly
records `"gpqa_status": "BLOCKED"` with a clear explanation.
