# Baseline Implementation Audit

**Date:** 2026-04-01  
**Evidence basis:** Source code inspection of `src/baselines/`, `src/policies/`, `scripts/`, and `docs/BASELINE_TRACKER.md`

---

## Summary Table

| Baseline Name | Status | Canonical Run Path / File | Used in Main Paper | Blocker if Not Runnable |
|---------------|--------|--------------------------|-------------------|------------------------|
| `reasoning_greedy` (always-cheap) | **fully implemented** | `src/evaluation/real_policy_eval.py` (offline, reads enriched CSV) | ✅ Yes — main baseline | None |
| `direct_plus_revise` (always-revise) | **fully implemented** | `src/evaluation/real_policy_eval.py` (offline, reads enriched CSV) | ✅ Yes — main baseline | None |
| `adaptive_policy_v5` | **fully implemented** | `src/policies/adaptive_policy_v5.py`; run via `scripts/run_real_policy_eval.py` | ✅ Yes — headline adaptive | None (offline) |
| `adaptive_policy_v6` | **fully implemented** | `src/policies/adaptive_policy_v6.py`; run via `scripts/run_real_policy_eval.py` | ✅ Yes — main table | None (offline) |
| `adaptive_policy_v7` | **fully implemented** | `src/policies/adaptive_policy_v7.py`; run via `scripts/run_real_policy_eval.py` | ✅ Yes — main table | None (offline) |
| `oracle` | **fully implemented** | `src/evaluation/real_policy_eval.py` (offline) | ✅ Yes — upper bound | None |
| `confidence_threshold_router` | **fully implemented** | `src/baselines/confidence_threshold_router.py`; run via `scripts/run_confidence_threshold_baseline.py` | ⚠️ Appendix / supporting | None (offline, uses `unified_confidence_score` from enriched CSV) |
| `learned_router` (logistic + decision tree) | **fully implemented** | `src/baselines/learned_router_baseline.py`; run via `scripts/run_learned_router_baseline.py` | ⚠️ Appendix only | Requires `scikit-learn`; offline |
| `best_of_n` | **fully implemented** | `src/baselines/best_of_n.py` | ⚠️ Strategy eval only | Requires LLM API for live runs; strategy eval uses pre-computed data |
| `self_consistency` | **fully implemented** | `src/baselines/self_consistency.py` | ⚠️ Strategy eval only | Requires LLM API for live runs |
| `greedy` (direct answer) | **fully implemented** | `src/baselines/greedy.py` | ✅ Yes | None |
| `BESTRouteAdaptedBaseline` | **partial — adapted, not official** | `src/baselines/external/best_route_wrapper.py` | ❌ No (appendix note only) | Official pipeline requires multi-model responses and DeBERTa reward-model; `BESTRouteAdaptedBaseline` runs offline with binary-setting adaptation but deviates from official protocol |
| `TALEBaseline` | **stub only** | `src/baselines/external/tale_wrapper.py` | ❌ No | Official TALE repo not cloned into `external/tale/.repo`; would require `OPENAI_API_KEY` for prompting |
| Vanilla CoT | **absent** (mentioned in BASELINE_TRACKER.md as planned) | — | ❌ No | No implementation; would require LLM API |
| Snell et al. (TTC scaling) | **absent** | — | ❌ No | No official code found; no reimplementation |
| MCKP allocation baseline | **absent** (`configs/simulated_mckp.yaml` exists; no live eval) | `scripts/run_simulated_allocation.py` | ❌ No (simulated sweep only) | Simulated-only; no enriched-CSV adaptation |
| SelfBudgeter | **absent** | — | ❌ No | No code; no official release found |
| DEER | **absent** | — | ❌ No | No code; no official release found |
| Rewarding Progress / PRM | **absent** | — | ❌ No | No code; requires process reward model |

---

## Detailed Notes

### Fully Runnable (Offline — No API Key Needed)

The following baselines can be re-run immediately on the committed enriched CSVs:

1. **reasoning_greedy / direct_plus_revise / oracle** — columns `reasoning_correct`, `revise_correct`, `revise_helpful` already in CSV; `real_policy_eval.py` reads directly.
2. **adaptive_policy_v5/v6/v7** — policy logic reads `question` and `reasoning_raw` from CSV; all feature extraction is deterministic and offline.
3. **confidence_threshold_router** — reads `unified_confidence_score` from enriched CSV; sweeps threshold values.
4. **learned_router** — trains sklearn logistic regression / decision tree on enriched CSV features; uses 5-fold CV.

```bash
# Run all offline baselines on one regime:
python3 scripts/run_real_policy_eval.py \
  --dataset-csv data/real_gsm8k_routing_dataset_enriched.csv \
  --output-dir outputs/real_policy_eval

python3 scripts/run_confidence_threshold_baseline.py

python3 scripts/run_learned_router_baseline.py
```

### Partially Implemented

**BESTRouteAdaptedBaseline** (`src/baselines/external/best_route_wrapper.py`)  
- A documented compatibility adaptation exists and is runnable in the binary routing setting.  
- Deviates from the official BEST-Route pipeline (which uses multi-model responses, DeBERTa-v3 reward model, and bubble-up evaluation).  
- The official repo (`https://github.com/microsoft/best-route-llm`) is not cloned.  
- Blocker for official integration: requires pre-generating multiple model responses + trained reward model.

### Stubs Only

**TALEBaseline** (`src/baselines/external/tale_wrapper.py`)  
- Class skeleton only; `solve()` is not implemented.  
- Official repo not cloned into `external/tale/.repo`.  
- To activate: `git clone https://github.com/ChenWu98/TALE external/tale/.repo` and implement wrapper.

### Absent / Planned

Vanilla CoT, Snell et al., SelfBudgeter, DEER, Rewarding Progress — no code present. Flagged in `docs/BASELINE_TRACKER.md` as `📋 Planned`.

---

## Conclusion for Q1 Submission

**Fully implemented and runnable offline:** reasoning_greedy, direct_plus_revise, oracle, adaptive_policy_v5/v6/v7, confidence_threshold_router, learned_router, best_of_n (with API), self_consistency (with API), greedy.

**Gap:** The major competitive baselines from recent literature (TALE, Snell et al., SelfBudgeter, DEER) are absent. These are the strongest potential reviewers' requests. BESTRoute has a runnable adaptation but not the official pipeline.

**Recommendation:** Priority for submission strengthening should be implementing TALE (official code public) and a functional BESTRoute comparison using the adapted single-model version.
