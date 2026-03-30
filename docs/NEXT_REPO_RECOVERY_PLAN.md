# Next repository recovery plan (manuscript-ready branch)

**Scope:** Repository-grounded snapshot at `/workspace` only. No assumptions about other git branches or undeclared assets.

**Audit note:** `docs/FULL_PROJECT_STATE_AUDIT.md` lists some paths as absent; this plan reflects **current tree inspection** (e.g. `docs/PAPER_ARTIFACT_GENERATION_STATUS.md`, `docs/EXPERIMENT_ARTIFACT_STATUS.md`, and `src/paper_artifacts/` **are present**; generated `outputs/paper_tables/` and `outputs/paper_figures/` appear after running the export scripts but are **not git-whitelisted** in `.gitignore`).

---

## A. What is already usable now

### Binary real-routing pipeline (end-to-end in code + checked-in data)

| Stage | Scripts | Primary inputs | Primary outputs (repo) |
|-------|---------|----------------|-------------------------|
| GSM8K 100 (paired) | `scripts/run_build_real_routing_dataset.py` | HF GSM8K or `--gsm8k-data-file`; `OPENAI_API_KEY` | `outputs/real_routing_dataset/*`, `data/real_gsm8k_routing_dataset.csv` (+ `_enriched.csv` if enrichment run) |
| Hard GSM8K | `scripts/run_build_hard_gsm8k_routing_dataset.py` | `outputs/hard_regime_selection/hard_gsm8k_selection.csv` | `outputs/real_hard_gsm8k_routing/*`, `data/real_hard_gsm8k_routing_dataset*.csv` |
| Hard GSM8K B2 | Same pattern with B2 selection | `outputs/hard_regime_selection_b2/hard_gsm8k_selection.csv` | `outputs/real_hard_gsm8k_routing_b2/*`, `data/real_hard_gsm8k_b2_routing_dataset*.csv` |
| MATH500 | `scripts/run_build_math500_routing_dataset.py` | MATH500 source per script | `outputs/real_math500_routing/*`, `data/real_math500_routing_dataset*.csv` |
| AIME 2024 | `scripts/run_build_aime_routing_dataset.py` | HF AIME 2024 | `outputs/real_aime2024_routing/*`, `data/real_aime2024_routing_dataset.csv` |
| Learned binary router | `scripts/run_real_routing_model_eval.py` | Enriched routing CSVs | `outputs/real_*_routing_model/summary.json`, `routing_simulation.csv`, metrics |
| Adaptive / fixed-route policy comparison | `scripts/run_real_policy_eval.py` (default GSM8K CSV; override `--dataset-csv`) | Routing CSV | `outputs/real_*_policy_eval/summary.json`, `policy_comparison.csv`, per-query decisions |
| Cross-regime rollups | `scripts/run_cross_regime_comparison.py`, `scripts/run_final_cross_regime_summary.py` | Policy + dataset artifacts | `outputs/cross_regime_comparison/*.csv` |
| Next-stage / oracle routing summaries | `scripts/run_next_stage_postprocess.py` (per dataset key) | Routing CSVs + configs | `outputs/oracle_routing_eval/*_oracle_summary.json`, `outputs/next_stage_eval/<key>/*` |

**Documentation:** `docs/REAL_ROUTING_DATASET.md`, `docs/ROUTING_DATASET.md`, `docs/HARD_GSM8K_ROUTING_RESULTS.md`, `docs/MATH500_ROUTING_RESULTS.md`, `docs/REAL_GSM8K_ROUTING_STUDY.md`, `docs/HARD_REGIME_ROUTING_STUDY.md`, `docs/REAL_ROUTING_MODEL_RESULTS.md`, `docs/MANUSCRIPT_RESULTS_READINESS.md`.

### Label strength (from `docs/FULL_PROJECT_STATE_AUDIT.md`, CSV-backed)

- **Strong for binary routing narrative:** hard GSM8K (100) — **17%** reasoning/revise disagreement; MATH500 (100) — **12%**.
- **Weak / control:** GSM8K random 100 — **2%** disagreement; AIME (30) — sparse revise signal in summaries.

### Paper export **code** (present; outputs are generated on demand)

- `scripts/generate_paper_tables.py`, `scripts/generate_paper_figures.py`
- `src/paper_artifacts/paths.py`, `exports_tables.py`, `exports_figures.py`
- `docs/PAPER_ARTIFACT_GENERATION_STATUS.md`, `docs/EXPERIMENT_ARTIFACT_STATUS.md` describe inputs and exporter keys.

**Partial export verified on this snapshot (non-`--strict`):** most exporters succeed from existing `outputs/`; **blockers** observed: missing `outputs/simulated_sweep/` (simulated tables/figures) and `outputs/oracle_subset_eval/summary.json` (oracle subset table).

---

## B. What is missing but essential

| Gap | Status in snapshot | Why it matters |
|-----|-------------------|----------------|
| `outputs/paper_tables/`, `outputs/paper_figures/` | Absent until export scripts run; not whitelisted in `.gitignore` | Manuscript tables/figures; reproducible “camera-ready” CSV/PNG |
| Full **`--strict`** paper pass | Fails until simulated sweep + oracle subset artifacts exist | Single command to assert no missing inputs |
| `data/multi_action_routing_*.csv` | **No files** | Required input for `scripts/run_multi_action_model_eval.py` |
| `outputs/multi_action_oracle/` | **No directory** | Written by `scripts/run_build_multi_action_dataset.py` (oracle summaries, disagreement JSON) |
| `outputs/multi_action_models/` | **No directory** | Written by `run_multi_action_model_eval.py` |
| `outputs/simulated_sweep/*` | **Missing** | Blocks simulated sweep exporters |
| `outputs/oracle_subset_eval/summary.json` | **Missing** | Blocks oracle subset table exporter |

---

## C. Recommended path: **PATH A** (stay on this branch; paper export first from current binary-routing outputs)

**Justification (shortest credible path to manuscript-ready results):**

1. **Infrastructure is already in-repo:** `src/paper_artifacts/` and `generate_paper_*.py` are not absent; rebuilding from scratch is unnecessary. The gap is **running** exporters and optionally filling **two** upstream artifact holes for `--strict`.
2. **Strongest grounded evidence is binary hard GSM8K + MATH500** (non-degenerate disagreement). Multi-action CSVs and `outputs/multi_action_models/` are **fully missing**; generating them is a **second API-heavy track** that does not unblock the core cost–accuracy story if the paper leads with binary routing and cross-regime policy tables.
3. **PATH B** (multi-action CSVs first) delays manuscript artifacts and may still yield **degenerate or thin** multi-action labels on math slices (per `docs/MULTI_ACTION_ROUTING_RESULTS.md` and related reports)—worth doing **after** the binary export loop is green.
4. **PATH C** (merge another branch first) is **only** justified if you need **pre-generated artifacts** without re-running API builds. This snapshot does not contain those artifacts; the remote name `origin/cursor/multi-action-routing-pipeline-dc5d` exists in `git remote` listings and may be worth **inspecting** for CSVs or outputs before merging—**do not assume** it matches this tree without a diff review.

---

## D. Exact next 10 steps (files / scripts)

1. **Regenerate partial paper tables (already mostly works):**  
   `python3 scripts/generate_paper_tables.py`  
   Inspect `outputs/paper_tables/export_manifest.json` for blockers.

2. **Regenerate partial paper figures:**  
   `python3 scripts/generate_paper_figures.py`  
   Inspect `outputs/paper_figures/export_manifest.json`.

3. **Fill simulated sweep gap (unblocks simulated exporters + strict):**  
   `python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml`  
   → `outputs/simulated_sweep/budget_sweep_comparisons.csv`, `noise_sensitivity_comparisons.csv`, etc.

4. **Fill oracle subset gap (unblocks oracle subset exporter + strict):**  
   `python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml`  
   → `outputs/oracle_subset_eval/summary.json` (requires `OPENAI_API_KEY`).

5. **Strict paper pass:**  
   `python3 scripts/generate_paper_tables.py --strict`  
   `python3 scripts/generate_paper_figures.py --strict`

6. **Align narrative with `docs/MANUSCRIPT_RESULTS_READINESS.md`:** cite `outputs/paper_tables/real_routing/real_policy_eval_comparison_long.csv`, `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv`, and regime-specific policy summaries under `outputs/real_*_policy_eval/summary.json`.

7. **Optional subset export during iteration:**  
   `python3 scripts/generate_paper_tables.py --only real_policy_comparison final_cross_regime cross_regime oracle_routing next_stage_budget_curves baselines`

8. **If multi-action is required for the manuscript:** run dataset build (API-heavy):  
   `python3 scripts/run_build_multi_action_dataset.py --help`  
   then e.g. hard GSM8K / MATH500 / GPQA per `docs/MULTI_ACTION_DATA_EXPANSION_REPORT.md` → `data/multi_action_routing_<slug>.csv` + `outputs/multi_action_oracle/`.

9. **Train / simulate multi-action routers:**  
   `python3 scripts/run_multi_action_model_eval.py --csv data/multi_action_routing_<slug>.csv --dataset-name <slug>`  
   → `outputs/multi_action_models/<slug>_model_results.json` (note: **no** dedicated row in current `generate_paper_tables.py` keys—results feed prose or a future exporter).

10. **Keep `docs/EXPERIMENT_ARTIFACT_STATUS.md` honest:** after runs, update “last verification” and artifact lists to match **this** tree (or add a dated note pointing to manifest JSON paths).

---

## Multi-action support: A / B / C (this snapshot)

| Category | Evidence |
|----------|----------|
| **A — Usable existing support** | `src/evaluation/multi_action_routing.py` (oracle labels, `best_accuracy_action`, CSV assembly, disagreement helpers); `scripts/run_build_multi_action_dataset.py`; `scripts/run_multi_action_model_eval.py`; `tests/test_multi_action_routing.py`. Binary **disagreement** is already in enriched real routing CSVs (`reasoning_correct` vs `revise_correct`). |
| **B — Partial support** | Docs (`docs/MULTI_ACTION_*.md`) reference CSV/JSON paths that are **not** in `data/` or `outputs/` here; paper exporters have **no** multi-action table key yet. |
| **C — Fully missing (artifacts)** | `data/multi_action_routing_*.csv`, `outputs/multi_action_oracle/`, `outputs/multi_action_models/`. |

---

## Paper export: rebuild vs recover

- **Do not rebuild from scratch:** `scripts/generate_paper_tables.py`, `scripts/generate_paper_figures.py`, and `src/paper_artifacts/` are present and functional.
- **Recover:** run the two missing upstream generators (simulated sweep, oracle subset) if full `--strict` is required; otherwise partial export already yields most real-routing and cross-regime artifacts.
