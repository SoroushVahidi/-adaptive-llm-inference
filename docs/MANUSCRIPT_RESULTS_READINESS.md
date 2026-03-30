# Manuscript results readiness

This document separates **exportability** (can `generate_paper_tables.py` / `generate_paper_figures.py` run with `--strict`?) from **scientific sufficiency** for an EAAI-style manuscript grounded in this repository.

Grounding: `docs/PROJECT_CONTEXT.md` (research goal, MCKP framing, baseline families, “empirical superiority” axis), `docs/NEXT_STAGE_EXPERIMENT_RESULTS.md`, `docs/STRONG_BASELINES_REPORT.md`, and the current contents of `outputs/` and `outputs/paper_*`.

---

## A. What is already manuscript-ready

These pieces are **coherent, documented, and large enough** to support *carefully scoped* claims, provided the text matches the evidence labels already used in repo docs (**measured_now** vs **exploratory_only** / small-N).

| Topic | Why it can be “ready” | Primary artifacts |
|-------|------------------------|-------------------|
| **Real adaptive policies vs fixed routes (cost–accuracy)** | Multiple `outputs/real_*_policy_eval/summary.json` files with `comparison` rows; exported long table + per-regime scatter figures. | `outputs/paper_tables/real_routing/real_policy_eval_comparison_long.csv`, `outputs/paper_figures/real_routing/*.png`, upstream policy summaries |
| **Cross-regime / final cross-regime narrative** | `final_cross_regime_summary.csv` aggregates reasoning, revise, oracle, best policy, RTR where enriched CSVs exist. | `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv`, `data/real_*_routing_dataset*_enriched.csv` |
| **Next-stage: oracle revise, budget curves, cascade** | Repo marks these as **measured_now**; exporters cover four dataset keys in merged budget CSV + PNGs. | `outputs/next_stage_eval/*/budget_curve.csv`, `cascade_curve.csv`, `outputs/oracle_routing_eval/*_oracle_summary.json`, paper exports under `next_stage/` |
| **Synthetic MCKP vs equal allocation** | Single diagnostic sweep (fixed config) produces stable CSVs and three standard plots; appropriate for **illustrating** allocation behavior on synthetic instances, not for claiming real-LLM optimality. | `configs/simulated_sweep.yaml` → `outputs/simulated_sweep/*`, `outputs/paper_tables/simulated_sweep/*`, `outputs/paper_figures/simulated_sweep/*` |

---

## B. What is provisional

Useful for drafting and structure, but **easy to misread** as “final” if not qualified in the manuscript.

| Topic | Provisional aspect | Evidence in repo |
|-------|--------------------|------------------|
| **Oracle subset strategy comparison** | **n = 15** queries (`outputs/oracle_subset_eval/summary.json`); fine for methods illustration, **not** for precise strategy ranking or significance. | `total_queries`: 15; bundled sample in `configs/oracle_subset_eval_gsm8k.yaml` |
| **Strong baselines ladder in exported tables** | Exported rollups read `*_baseline_summary.json`. Current committed-style numbers in workspace show **n_queries 30** (gsm8k/hard_gsm8k) and **15** (math500) with `docs/NEXT_STAGE_EXPERIMENT_RESULTS.md` explicitly calling baselines **partial** and MATH500 ladder truncated after a kill. | `outputs/baselines/*_baseline_summary.json`, docs §2 |
| **Learned router quality** | `outputs/real_routing_model/summary.json` documents **small positive count** and **0 F1** for listed models — honest but **not** a success story for “learned routing beats heuristics.” | `num_positive`, `models[].f1` |
| **AIME row in final cross-regime CSV** | Row exists (`aime2024`) but **best_policy_* columns are empty** in exported CSV (no matching policy eval block in `run_final_cross_regime_summary.py` for AIME). | `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv` |
| **Simulated sweep as “full empirical evaluation”** | One synthetic instance family, one seed in config; no multi-seed aggregation in the **paper export** path unless you add narrative pointing to `run_simulated_multi_seed.py` outputs. | `configs/simulated_sweep.yaml` |

---

## C. What still needs stronger evidence

Aligned with `PROJECT_CONTEXT.md` emphasis on **empirical comparisons** and **routing + budgeting**, these gaps matter if the paper asserts broad superiority or production readiness.

| Gap | Why it weakens claims | What would strengthen |
|-----|------------------------|------------------------|
| **Oracle subset sample size** | Cannot support tight confidence intervals or definitive strategy ordering. | Rerun `run_oracle_subset_eval.py` with larger `max_samples` (config), same model, document cost |
| **Strong baselines on full paper slices** | “Strong modern baselines” (self-consistency, routers, ladder) need **n** aligned with main real experiments (e.g. 100) on gsm8k / hard_gsm8k / math500. | `scripts/run_strong_baselines.py` + `configs/strong_baselines_real.yaml`, increase limits per `docs/NEXT_STAGE_EXPERIMENT_RESULTS.md` |
| **External / TALE / BEST-Route baselines** | `BASELINE_TRACKER` families exist; current **paper export** does not pull TALE/BEST-Route artifacts. | Runs under `src/baselines/external/` + explicit tables if claims require them |
| **Multi-seed or sensitivity for synthetic claims** | Single-instance sweep risks “one draw” criticism if sold as robust phenomenon. | `scripts/run_simulated_multi_seed.py` + summarize in text or extend exporters |
| **GPQA / other datasets** | Not in current paper export manifests; optional per project scope. | `run_recent_baselines_experiment.py` or strong baselines config if in scope |

---

## D. Minimum experiment plan before freezing “main” results

Prioritized for **highest leverage** vs effort (API costs apply where noted).

| Priority | Run | Why | Expected outputs | API / network | Touches main table/figure? |
|----------|-----|-----|------------------|---------------|----------------------------|
| 1 | `python3 scripts/run_strong_baselines.py --config configs/strong_baselines_real.yaml` (adjust `max_samples` / datasets to match main 100-query slices) | Addresses **partial** baseline evidence in `NEXT_STAGE_EXPERIMENT_RESULTS.md`; needed if the paper claims competitive baselines. | `outputs/baselines/*_baseline_summary.json`, `final_baseline_summary.csv`, ladder/router files | **Yes** (OpenAI + HF as configured) | **Yes** — `paper_tables/baselines/*` |
| 2 | `python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml` with increased `dataset.max_samples` | Oracle subset table is scientifically thin at n=15. | `outputs/oracle_subset_eval/summary.{json,csv}`, matrices | **Yes** | **Yes** — `paper_tables/oracle_subset/*` |
| 3 | (Optional) `python3 scripts/run_simulated_multi_seed.py --config configs/simulated_multi_seed.yaml` then point narrative or extend export | Robustness of MCKP vs equal under instance draw. | `outputs/simulated_multi_seed/*` | No | Strengthens **synthetic** subsection; may need exporter extension for a single “paper” table |
| 4 | Wire AIME (or drop row): either add policy eval + extend `run_final_cross_regime_summary.py` or exclude AIME from “main” table | Avoids a **partial row** in `final_cross_regime_summary.csv`. | Updated `final_cross_regime_summary.csv` | If policy eval needed: **Yes** | **Yes** — cross-regime table |
| 5 | Re-run / expand `run_next_stage_postprocess.py` only if routing CSVs change | Keeps budget/cascade curves aligned with refreshed routing data. | `outputs/next_stage_eval/*`, `outputs/oracle_routing_eval/*` | No if CSVs local | **Yes** — next-stage exports |

After any upstream change:

```bash
python3 scripts/generate_paper_tables.py --strict
python3 scripts/generate_paper_figures.py --strict
```

---

## Claim → evidence mapping (high level)

| Likely manuscript claim (from repo framing) | Supporting export(s) | Upstream artifact(s) | Strength now |
|---------------------------------------------|----------------------|----------------------|--------------|
| Batch allocation can be cast as MCKP; equal vs optimized allocation differs under budget/noise (synthetic) | Simulated sweep tables + 3 figures | `outputs/simulated_sweep/*.csv` | **Strong** for single-config diagnostic; **partial** for generality |
| Adaptive policies trade off cost vs accuracy on real routing slices | Real policy long table + scatter figures | `outputs/real_*_policy_eval/summary.json` | **Strong** for **those** 100-query regimes |
| Revise-helpful oracle and budget/cascade behavior | Oracle routing merged CSV; next-stage merged budget CSV; budget/cascade PNGs | `oracle_routing_eval`, `next_stage_eval` | **Strong** per repo **measured_now** labels |
| Strategy landscape on a small GSM8K oracle set | Oracle subset exports | `outputs/oracle_subset_eval/*` | **Weak** for statistical claims; **OK** for qualitative / methods |
| Competitive self-consistency / router baselines | Baselines strategy CSVs | `outputs/baselines/*_baseline_summary.json` | **Partial** until n and real config match paper narrative |
| Learned router is a strong predictor | *(Not a dedicated paper export)* | `outputs/real_*_routing_model/summary.json` | **Weak** — metrics in repo show limited signal |

---

## Readiness labels by experiment family

| Family | Label | Notes |
|--------|--------|------|
| Simulated sweep (current config) | **USABLE BUT SHOULD BE STRENGTHENED** | Good for theory/diagnostic section; add multi-seed or refrain from overclaiming |
| Oracle subset (n=15) | **NOT YET ADEQUATE FOR CLAIMS** that need precise strategy comparisons or population-level rates | OK for pilot / appendix if labeled |
| Strong baselines (current JSON in workspace) | **USABLE BUT SHOULD BE STRENGTHENED** | Align n with main study; use real config if claiming gpt-4o-mini baselines |
| Real policy eval + cross-regime + final summary | **READY FOR MANUSCRIPT** (scoped to described slices and models) | Match text to **measured_now** caveats in docs |
| Next-stage (oracle routing, budget, cascade) | **READY FOR MANUSCRIPT** (as in `NEXT_STAGE_EXPERIMENT_RESULTS.md`) | Same |
| Learned routing model eval | **NOT YET ADEQUATE FOR CLAIMS** of strong learned routing | Can report as **negative / small-N** result if framed honestly |

---

## Relation to strict export success

`python3 scripts/generate_paper_tables.py --strict` and `python3 scripts/generate_paper_figures.py --strict` succeeding means **all required input paths exist and parse**. It does **not** mean every number is publication-final. Use this file together with `docs/EXPERIMENT_ARTIFACT_STATUS.md` and `docs/PAPER_ARTIFACT_GENERATION_STATUS.md`.
