# Strong baselines report

This document summarizes the **strong baseline suite** added for EAAI-style comparisons: self-consistency variants, a fixed compute ladder, confidence-based routing, output-aware routing, and a simplified BEST-Route-style router.

## 1. What was implemented

| Component | Location | Description |
|-----------|----------|-------------|
| Self-consistency (N=3, N=5) | `src/baselines/self_consistency.py` | Normalized majority vote; `self_consistency_ambiguous` when the winning vote is empty; `self_consistency_tie` when counts tie (deterministic lexicographic break + `logging.info`) |
| BaselineResult fields | `src/baselines/base.py` | `self_consistency_ambiguous`, `self_consistency_tie` |
| Metrics | `src.evaluation.metrics.compute_accuracy` | Adds aggregate ambiguous/tie counts and rates when any result sets those flags |
| `reasoning_greedy`, `reasoning_then_revise` | `src/evaluation/strategy_expansion_eval.py` | Reused by ladder and routers; registered in expanded eval |
| Hard GSM8K | `src/datasets/hard_gsm8k.py` | Test split sorted by question length (longest first), optional cap |
| Strong baselines driver | `src/evaluation/strong_baselines_eval.py` | Compute ladder, confidence-threshold curve, output-aware router, BEST-Route-style simplification |
| CLI | `scripts/run_strong_baselines.py` | End-to-end run + CSV/JSON under `outputs/baselines/` |
| Experiment runner | `scripts/run_experiment.py` | Baselines `self_consistency_3`, `self_consistency_5` |

**Cost proxy:** number of `generate` / `generate_n` calls (same as existing strategy expansion). **Extra compute rate** (in ladder and summaries): `avg_cost_proxy − 1` (margin over single-call `reasoning_greedy`).

## 2. What ran successfully

| Run | Config | Model | Data |
|-----|--------|-------|------|
| Offline smoke | `configs/strong_baselines_dummy.yaml` | Dummy | Bundled `gsm8k_test_sample.json` + `math500_tiny.json` |
| HF-backed smoke | `configs/strong_baselines_hf_smoke.yaml` | Dummy | GSM8K test, Hard GSM8K (longest 40), MATH500 (40) via HuggingFace |

Artifacts (overwritten by the latest run):

- `outputs/baselines/{gsm8k,hard_gsm8k,math500}_compute_ladder.json`
- `outputs/baselines/{dataset}_confidence_router.csv`
- `outputs/baselines/{dataset}_output_router.json`
- `outputs/baselines/{dataset}_best_route_style.json`
- `outputs/baselines/final_baseline_summary.csv`
- `outputs/baselines/dataset_rollup.csv`
- `outputs/baselines/run_log.json`

## 3. What failed and why

### 3.1 MATH500 + constraint features (fixed)

**Symptom:** `decimal.InvalidOperation: ConversionSyntax` inside `_question_profile` when evaluating routers on MATH500 questions containing LaTeX-style numeric fragments (e.g. `\frac`) that the question-number regex matched incorrectly.

**Cause:** Parsing failure — `Decimal(...)` on non-numeric tokens.

**Fix:** Skip tokens that do not convert to `Decimal` in `src/features/constraint_violation_features.py`.

### 3.2 Official BEST-Route

**Status:** Not reproduced. `external/best_route/.repo` is absent; `src/baselines/external/best_route_wrapper.py` remains a stub.

**What we still run:** `best_route_style` in `strong_baselines_eval.py` — a **simplified** threshold policy on `difficulty_proxy + (1 − confidence_proxy)` over actions `reasoning_greedy`, `self_consistency_3`, `reasoning_then_revise`.

### 3.3 AIME / GPQA

**Status:** No loaders or configs exist in this repository for AIME or GPQA. **Missing:** dataset integration (and, for live eval, API keys / HF access as applicable).

### 3.4 Real LLM runs

**Blocker if used:** `OPENAI_API_KEY` must be set for `model.type: openai` in config. The smoke configs use **dummy** only.

## 4. Accuracy vs cost (latest HF smoke, dummy model, 40 queries each)

The dummy model is for **pipeline verification**, not meaningful accuracy. Numbers below illustrate **file layout and relative costs**, not LLM quality.

From `outputs/baselines/dataset_rollup.csv` (best accuracy among listed methods):

| Dataset | Best static (method, acc, cost) | Best adaptive (method, acc, cost) |
|---------|--------------------------------|-----------------------------------|
| gsm8k | self_consistency_3, 0.075, 3.0 | conf_router_direct_plus_revise @ τ=0.5, 0.075, 1.5 |
| hard_gsm8k | self_consistency_3, 0.025, 3.0 | conf_router_direct_plus_revise @ τ=0.2, 0.05, 1.0 |
| math500 | self_consistency_3, 0.025, 3.0 | conf_router_direct_plus_revise @ τ=0.15, 0.025, 1.0 |

Full per-method rows: `outputs/baselines/final_baseline_summary.csv`. Confidence sweep: per-dataset `*_confidence_router.csv`.

## 5. Strongest baseline per dataset (this smoke)

By **accuracy** on the rollup above: **self_consistency_3** (static) on all three slices; best **adaptive** point in this run was a **confidence router** operating point, not uniformly better than static on accuracy.

## 6. Does adaptive routing beat static?

**On this dummy smoke:** Sometimes an adaptive point **matches** the best static accuracy at **lower** average cost (e.g. GSM8K: 0.075 at cost 1.5 vs 3.0). This does **not** generalize until repeated with a real LLM and larger `max_samples`.

**How to verify with a real model:** `configs/strong_baselines_openai.yaml` (create) with `model.type: openai`, unset bundled `*_data_file` keys to use HF, and run `python3 scripts/run_strong_baselines.py --config ...`.

## 7. Self-consistency vs revise

From `gsm8k_compute_ladder.json` (40-query dummy HF slice): **self_consistency_3** (0.075) scored higher than **direct_plus_revise** (0.0) and **reasoning_then_revise** (0.025). Again, this reflects the **dummy** stochastic “answers,” not a claim about GPT-class models.

## 8. Practical deployment takeaways

1. **Self-consistency** needs **tie** and **ambiguous** telemetry — now logged in results and aggregates.
2. **MATH-style text** in questions broke naive numeric profiling — **robust parsing** (skip bad tokens) matters for routing features.
3. **Cost–accuracy curves** for routers are CSV-friendly: sweep thresholds in `evaluate_confidence_router_curve`.
4. **BEST-Route** should stay clearly labeled as **simplified** until official code is integrated.

## Commands

```bash
pip install -e ".[dev]"
# Offline bundled data
python3 scripts/run_strong_baselines.py --config configs/strong_baselines_dummy.yaml
# HF slices (requires network first time)
python3 scripts/run_strong_baselines.py --config configs/strong_baselines_hf_smoke.yaml
pytest
ruff check src/ tests/ scripts/
```
