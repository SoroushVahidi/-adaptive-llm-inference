# BEST-Route Integration Status

**Date:** 2026-03-31
**Official code access:** ✅ Yes — `https://github.com/microsoft/best-route-llm` (MIT)
**Integration status:** ⚠️ Partial — Compatibility adaptation complete; official pipeline blocked

---

## Summary

| Item | Status |
|------|--------|
| Official code accessible | ✅ Yes |
| Official code cloned into repo | ❌ No (multi-stage pipeline required) |
| Official wrapper (`BESTRouteBaseline`) | ⚠️ Blocked — raises RuntimeError without `.repo` |
| Compatibility adaptation (`BESTRouteAdaptedBaseline`) | ✅ Complete and runnable |
| Config file | ✅ `configs/best_route_adapted.yaml` |
| Tests | ✅ 14 tests; all pass |
| Integration documentation | ✅ `docs/BEST_ROUTE_INTEGRATION.md` |
| Dummy-model smoke test | ✅ Passes offline |
| Real-model evaluation | ❌ Not run (would require new API calls) |

---

## Files Changed

| File | Type | Summary |
|------|------|---------|
| `src/baselines/external/best_route_wrapper.py` | Modified | Correct references; `BESTRouteAdaptedBaseline` added |
| `external/best_route/README.md` | Modified | Correct paper/code info; BibTeX; setup steps |
| `configs/best_route_adapted.yaml` | Added | Config for adapted baseline |
| `tests/test_external_baselines.py` | Modified | +13 tests for adapted baseline; +1 config test |
| `docs/BEST_ROUTE_INTEGRATION.md` | Added | Full integration document |
| `docs/BEST_ROUTE_INTEGRATION_STATUS.md` | Added | This file |
| `docs/BASELINE_TRACKER.md` | Modified | Corrected GitHub URL and updated status |

---

## Run Commands

### Tests (offline, no API required)
```bash
python -m pytest tests/test_external_baselines.py -v
```

### Adapted baseline evaluation (dummy model, offline)
```bash
python scripts/run_strong_baselines.py --config configs/best_route_adapted.yaml
```

### Adapted baseline evaluation (real model, API required)
```bash
# Edit configs/best_route_adapted.yaml: set model.type to "real"
python scripts/run_strong_baselines.py --config configs/best_route_adapted.yaml
```

---

## Official Code Blockers

Full integration of `BESTRouteBaseline` (the official-code wrapper) is
blocked by the following hard dependencies:

| Blocker | Reason | Effort estimate |
|---------|--------|-----------------|
| Multiple LLM backends | Official routes between LLaMA-8B and GPT-4o | High (API keys + inference cost) |
| 20 response samples/query/model | Required for reward scoring | High (API cost) |
| ArmoRM oracle scoring | Requires 8B reward model locally | Medium (model download) |
| Proxy reward model training | DeBERTa fine-tuning on 8 000 examples | High (GPU hours) |
| Router training | DeBERTa-v3-small fine-tuning | High (GPU hours) |

None of these can be completed from the committed artifacts alone.

---

## Deviations from Official Algorithm

See `docs/BEST_ROUTE_INTEGRATION.md §5.2` for full deviation notes.

| ID | Summary | Impact |
|----|---------|--------|
| DEV-1 | Binary action space (2 vs ≥ 6 actions) | Limits routing granularity |
| DEV-2 | Feature-based router (vs trained DeBERTa) | Routing quality lower than official |
| DEV-3 | Binary quality signal (vs continuous armoRM) | Less informative reward signal |
| DEV-4 | Single model (vs multi-LLM) | No model-diversity benefit |

---

## Confidence Level

- **Faithfulness**: High — all deviations are explicitly documented and
  traceable to the official code/paper.
- **Correctness of adapted algorithm**: High — bubble inference and budget
  constraint are faithfully implemented; routing signal is a well-motivated
  proxy.
- **Numerical accuracy vs official BEST-Route**: Not verifiable without
  running the full official pipeline.

---

## Remaining Work

1. **Real-model evaluation**: Run `BESTRouteAdaptedBaseline` on the four
   canonical manuscript regimes with the real backend to produce reportable
   numbers.
2. **Threshold tuning**: Sweep `threshold` on a held-out validation split to
   find the optimal operating point per regime.
3. **Official wrapper bridge**: Once `.repo` is cloned and the pipeline is
   completed, implement `BESTRouteBaseline.solve()` to delegate to the
   official inference code.
4. **Multi-action extension**: If a multi-action setting is ever added to
   this repo, extend `BESTRouteAdaptedBaseline` to use more than 2 actions.
