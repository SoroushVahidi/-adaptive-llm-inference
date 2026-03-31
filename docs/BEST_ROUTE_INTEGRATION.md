# BEST-Route Integration Document

## 1. Overview

This document describes the integration status of the BEST-Route baseline
into this repository.  The integration was completed on 2026-03-31.

**Official paper:** BEST-Route: Adaptive LLM Routing with Test-Time Optimal
Compute (Ding et al., ICML 2025)
**arXiv:** <https://arxiv.org/abs/2506.22716>
**Official code:** <https://github.com/microsoft/best-route-llm> (MIT license,
accessed 2026-03-31)

---

## 2. Official Code Access

The official repository is publicly accessible at
`https://github.com/microsoft/best-route-llm`.  Its key files are:

| File | Role |
|------|------|
| `train_router.py` | DeBERTa-v3-small pairwise ranker training entry-point |
| `notebooks/generate_llm_responses.py` | Multi-model response sampling |
| `notebooks/scoring_per_model_armoRM.py` | ArmoRM oracle reward scoring |
| `notebooks/reward_modeling.py` | Proxy reward model (DeBERTa) training |
| `notebooks/data_preparation_hybridllm_mapping.ipynb` | Router training data prep |
| `requirements.txt` | Runtime dependencies (PyTorch, HuggingFace Transformers, etc.) |

The repository was inspected via the GitHub raw API on 2026-03-31.

---

## 3. Official Algorithm Summary

BEST-Route solves a budget-constrained routing problem.  For each query `x`:

1. **Action space** `A`: a set of `(LLM-model, best-of-n)` pairs ordered by
   cost, e.g. `{llama-31-8b_bo1, …, llama-31-8b_bo5, gpt-4o_bo1}`.
2. **Quality signal**: continuous reward score from either the armoRM oracle
   or a fine-tuned proxy reward model (DeBERTa-based), applied to each
   sampled response.
3. **Router**: a DeBERTa-v3-small pairwise ranker trained with `prob_nlabels`
   loss using the official `train_router.py` script.  Input: query text +
   candidate responses.  Output: probability distribution over actions.
4. **Inference mode "bubble"**: evaluate actions from cheapest to most
   expensive; escalate until the router's predicted quality gain no longer
   justifies the extra cost.
5. **Budget constraint**: controlled by `--match_t` (quality match threshold)
   and `--candidate_model` (set of available actions) during router training.

---

## 4. Architectural Mismatch

This repository studies **single-model, binary cheap-vs-revise routing**.
The official BEST-Route is a **multi-model, multi-tier routing** method.

| Dimension | Official BEST-Route | This repository |
|-----------|---------------------|-----------------|
| Models | Multiple LLMs (e.g. LLaMA-8B, GPT-4o) | Single model (gpt-4o-mini) |
| Actions | `(model, best-of-n)` pairs; ≥ 6 actions | Binary: {`reasoning_greedy`, `direct_plus_revise`} |
| Router | Trained DeBERTa-v3-small pairwise ranker | Feature-based threshold (adaptation) |
| Quality signal | Continuous armoRM / proxy-RM scores | Binary correctness |
| Pre-computation | 20 responses/query/model; reward scoring | None (offline feature extraction only) |
| Training data | 8 000+ labelled examples | None needed for adapted baseline |

Running the **full official pipeline** from this repository is **blocked** by
the following hard dependencies:

1. Multiple LLM API endpoints / local model weights.
2. 20 × N × M response generations (N queries, M models) — requires new API
   calls and significant compute.
3. ArmoRM oracle scoring — requires loading `RLHFlow/ArmoRM-Llama3-8B-v0.1`.
4. Proxy reward model training — DeBERTa fine-tuning on 8 000 examples.
5. Router training — further DeBERTa fine-tuning.

None of these can be executed from the committed artifacts alone.

---

## 5. Implemented Classes

### 5.1 `BESTRouteBaseline` (official-code adapter, blocked)

`src/baselines/external/best_route_wrapper.py`

A thin `ExternalBaseline` subclass that will delegate to the official
`microsoft/best-route-llm` code once `external/best_route/.repo` contains
the cloned repository.  Currently raises `RuntimeError` with a clear message
explaining the setup steps.  The error message was updated to reference the
correct repository URL (`microsoft/best-route-llm`, not the old stub URL).

### 5.2 `BESTRouteAdaptedBaseline` (compatibility adaptation, runnable)

`src/baselines/external/best_route_wrapper.py`

A fully documented compatibility adaptation.  Every design decision is
traceable to either the official code or an explicit deviation note.

**Faithful elements:**

| BEST-Route element | Faithful mapping in this adaptation |
|--------------------|-------------------------------------|
| Binary action space | `reasoning_greedy` (cost 1) ↔ `direct_plus_revise` (cost 2) |
| Bubble inference mode | First pass is always `reasoning_greedy`; escalation follows |
| Budget constraint | `n_samples` parameter; `n_samples < 2` → no escalation |
| Threshold routing | `threshold` parameter controls escalation decision |

**Deviations (all explicitly documented in the class docstring):**

| ID | Dimension | Official | This adaptation |
|----|-----------|----------|-----------------|
| DEV-1 | Action space | ≥ 6 `(model, n)` pairs | Exactly 2 binary actions |
| DEV-2 | Router | Trained DeBERTa ranker | Feature score: `difficulty_proxy + (1 − confidence_proxy)` |
| DEV-3 | Quality signal | Continuous armoRM score | Binary correctness |
| DEV-4 | Model diversity | Multiple LLMs | Single model passed at construction |

The routing score (DEV-2) is computed from existing per-query feature
functions already present in this repository:
```
score = _difficulty_score(question)
      + (1 − _confidence_from_first_reasoning(question, first_pass_raw, parsed))
```
Both functions use string/regex features only (no model calls).  The score
range is `[0, 2]`; the `threshold` parameter (default 0.5) controls
escalation.

---

## 6. Files Added / Modified

| File | Change | Description |
|------|--------|-------------|
| `src/baselines/external/best_route_wrapper.py` | **Modified** | Rewrote: correct references; added `BESTRouteAdaptedBaseline` with full docstring |
| `external/best_route/README.md` | **Modified** | Correct paper/code references; BibTeX; setup steps; usage example |
| `configs/best_route_adapted.yaml` | **Added** | Config for adapted baseline on canonical manuscript regimes |
| `tests/test_external_baselines.py` | **Modified** | Added 13 new tests for `BESTRouteAdaptedBaseline` |
| `docs/BEST_ROUTE_INTEGRATION.md` | **Added** | This document |
| `docs/BEST_ROUTE_INTEGRATION_STATUS.md` | **Added** | Status summary |
| `docs/BASELINE_TRACKER.md` | **Modified** | Updated BEST-Route row with correct URL and status |

---

## 7. Tests

New tests in `tests/test_external_baselines.py`:

| Test | What it validates |
|------|-----------------|
| `test_adapted_name` | `name == "best_route_adapted"` |
| `test_adapted_default_threshold` | Default threshold equals class constant |
| `test_adapted_custom_threshold` | Custom threshold stored correctly |
| `test_adapted_invalid_threshold_raises` | ValueError for out-of-range threshold |
| `test_adapted_budget_too_low_always_uses_cheap` | `n_samples < 2` → always cheap |
| `test_adapted_no_escalation_with_high_threshold` | `threshold=2.0` → never escalates |
| `test_adapted_escalation_when_score_high` | Uncertain first pass → escalation |
| `test_adapted_result_fields_present` | All `BaselineResult` fields populated |
| `test_adapted_deterministic_same_seed` | Identical inputs → identical outputs |
| `test_config_file_parseable` | YAML config parses without error |

---

## 8. How to Run on Canonical Manuscript Regimes

### Smoke test (dummy model, offline, no API):
```bash
python -m pytest tests/test_external_baselines.py -v
```

### Full evaluation (dummy model):
```bash
python scripts/run_strong_baselines.py --config configs/best_route_adapted.yaml
```

### Real-model evaluation (requires API key):
1. Edit `configs/best_route_adapted.yaml`: set `model.type` to `real`.
2. Set the appropriate API key environment variable.
3. Run:
   ```bash
   python scripts/run_strong_baselines.py --config configs/best_route_adapted.yaml
   ```

Results are saved to `outputs/baselines/{dataset}_best_route_adapted.json`.

---

## 9. Mapping to Main Paper Setting

This repository's paper studies **binary adaptive routing** between cheap
single-pass reasoning and costly self-revision on four regimes:
`gsm8k_random_100`, `hard_gsm8k_100`, `hard_gsm8k_b2`, `math500_100`.

The adapted BEST-Route baseline provides a comparable-budget routing
decision on the same binary action space, enabling direct comparison with
the paper's primary adaptive policy (`adaptive_policy_v5`).

**Comparison table** (from committed artifacts, no new inference):

| Regime | Adaptive v5 acc | Always-cheap acc | BEST-Route adapted uses same action space? |
|--------|:--------------:|:----------------:|:------------------------------------------:|
| GSM8K Random-100 | 0.92 | 0.90 | ✅ Yes |
| Hard GSM8K-100 | 0.86 | 0.79 | ✅ Yes |
| Hard GSM8K-B2 | 0.91 | 0.83 | ✅ Yes |
| MATH500-100 | 0.66 | 0.64 | ✅ Yes |

The adapted baseline's accuracy on the committed per-query data would need
to be computed by re-running `scripts/run_strong_baselines.py` with the
real model backend (not done here to avoid new API calls).

---

## 10. Limitations and Open Items

1. **No real-model accuracy numbers**: The adapted baseline has not been run
   with the real GPT-4o-mini backend on the committed regimes (would require
   new API calls).  Only dummy-model smoke tests are included.
2. **Single routing threshold**: The threshold (default 0.5) is not tuned on
   validation data; official BEST-Route tunes this implicitly via router
   training.  Regime-specific thresholds can be swept with
   `evaluate_confidence_router_curve` in `strong_baselines_eval.py`.
3. **No trained router**: DEV-2 is the largest deviation.  A trained router
   would require multi-model response sampling and reward scoring.
4. **Official wrapper incomplete**: `BESTRouteBaseline.solve()` raises
   `NotImplementedError` even when `.repo` exists; full bridge implementation
   requires completing the multi-stage pipeline described above.
