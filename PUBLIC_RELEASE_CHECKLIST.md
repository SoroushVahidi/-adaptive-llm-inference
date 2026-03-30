# Public Release Checklist

This document records what was cleaned before making the repository public and
lists items a maintainer must verify before flipping the visibility switch.

---

## What Was Cleaned (This PR)

### Removed from tracking

- `archive.zip` — duplicate binary archive (not source code or data)
- `archive (1).zip` — duplicate binary archive
- `AGENTS.md` — internal AI-agent instruction file (Copilot/Cursor Cloud
  operational notes; not relevant to external readers)

### Moved to `docs/internal/`

Internal AI-agent planning logs and working notes that would confuse external
readers were moved from `docs/` to `docs/internal/`:

- `CODEX_CRITICAL_STATE_REVIEW.md`
- `CODEX_NEXT_EXPERIMENT_RECOMMENDATION.md`
- `TONIGHT_THINKING_QUESTION.md`
- `OPEN_QUESTIONS_NOW.md`
- `DECISION_POINT_SUMMARY.md`
- `NEXT_REPO_RECOVERY_PLAN.md`
- `FULL_REPO_AUDIT.md`, `FULL_PROJECT_STATE_AUDIT.md`, `REPO_AUDIT_REPORT.md`
- `MAIN_BRANCH_STATUS_AUDIT.md`, `FINAL_PRE_RUN_AUDIT.md`
- `PAPER_EXPORT_RUN_STATUS.md`, `PAPER_ARTIFACT_GENERATION_STATUS.md`
- `CURRENT_STATE_SUMMARY.md`

### Strengthened `.gitignore`

Added entries for:
- Archive files (`*.zip`, `*.tar.gz`, etc.)
- Additional virtual-environment patterns (`env/`, `ENV/`)
- Editor files (`.vscode/`, `.idea/`, `*.swp`, `.DS_Store`, etc.)
- Model checkpoints (`*.ckpt`, `*.pt`, `*.bin`, `*.safetensors`)
- Temporary logs and debug folders
- Jupyter checkpoint directories
- `AGENTS.md` explicitly excluded

### Added new files

- `.env.example` — template for required environment variables
- `REPRODUCIBILITY.md` — exact commands for manuscript pipeline
- `DATA_AVAILABILITY.md` — dataset provenance and access instructions
- `MANUSCRIPT_ARTIFACTS.md` — inventory of committed tables/figures
- `PUBLIC_RELEASE_CHECKLIST.md` — this file
- `docs/internal/README.md` — explains internal/ folder to curious readers

### Rewrote `README.md`

The README was rewritten to be understandable to external readers with no
prior context.  It now covers: research question, project structure,
installation, offline quick-start, real-model experiments, data summary,
committed outputs, and known limitations.

---

## Security Checks Performed

- [x] Searched all Python, YAML, and JSON files for hardcoded API keys,
  tokens, and credentials — **none found**
- [x] Verified `OPENAI_API_KEY` is read exclusively from environment
  (`os.getenv`) and never logged or committed
- [x] `raw_responses.jsonl` files contain only public GSM8K/MATH500/AIME
  question-answer pairs — **no personal information**
- [x] `provider_metadata.json` files contain only model configuration
  metadata — **no credentials**
- [x] `.env` is gitignored; `.env.example` added as safe template

---

## What a Maintainer Must Verify Before Making Public

### Legal / licence

- [ ] Confirm the MIT licence in `LICENSE` is correct and covers all original
  code.
- [ ] Confirm AIME-2024 problem text in `data/real_aime2024_routing_dataset.csv`
  is permissible to redistribute (AoPS problems are widely published, but
  verify for your institution).
- [ ] Confirm GPQA-Diamond data in `data/gpqa_diamond_normalized.jsonl` can
  be redistributed under your intended licence (check upstream GPQA licence).

### Scientific accuracy

- [ ] Read `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv`
  and confirm these numbers match the manuscript exactly.
- [ ] Read `MANUSCRIPT_ARTIFACTS.md` and confirm no BLOCKED artifacts are
  cited as final in the paper.
- [ ] Ensure `STATE_OF_EVIDENCE.md` caveats are reflected in the paper text
  and no overclaims are present in `README.md`.

### Code quality

- [ ] Run `pytest` and confirm 612 tests pass.
- [ ] Run `ruff check src/ tests/ scripts/` — pre-existing minor issues are
  documented; no new issues should be introduced.
- [ ] Check that every script referenced in `REPRODUCIBILITY.md` actually
  exists and runs without import errors (dummy-model path only).

### Documentation consistency

- [ ] Verify all cross-references in `README.md`, `REPRODUCIBILITY.md`,
  `DATA_AVAILABILITY.md`, and `MANUSCRIPT_ARTIFACTS.md` point to files that
  exist.
- [ ] Read `docs/internal/README.md` to confirm the internal docs folder is
  clearly labelled.

### External dependencies

- [ ] Decide whether to include `external/tale/` and `external/best_route/`
  placeholder READMEs or remove them if the external repos will never be
  cloned.
- [ ] Confirm that the GitHub Actions workflow
  `.github/workflows/test-openai-key.yml` is acceptable to expose publicly
  (it tests a secret via `workflow_dispatch`; the secret itself is never
  printed).

---

## Recommended Visibility Decision

**The repository is safe to make public** once the legal, scientific accuracy,
and documentation consistency checks above are passed.

### Remaining risks before public release

1. **AIME and GPQA data licensing** — verify redistribution rights.
2. **Simulated sweep tables** — two paper tables are BLOCKED; confirm whether
   the manuscript needs them or they are appendix-only.
3. **Modest policy gains** — the routing policy improvements over the
   `reasoning_greedy` baseline are small (1–3 pp) on easy regimes.  The paper
   must not overclaim beyond what the committed data shows.
4. **Single-model scope** — all results are for `gpt-4o-mini`; the README
   states this limitation clearly.

---

## Files That a Human Should Inspect One Last Time

| File | Why |
|------|-----|
| `LICENSE` | Confirm it covers all content |
| `data/real_aime2024_routing_dataset.csv` | Check redistribution rights |
| `data/gpqa_diamond_normalized.jsonl` | Check redistribution rights |
| `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv` | Match to paper numbers |
| `docs/STATE_OF_EVIDENCE.md` | Honest evidence audit — cross-check with paper claims |
| `docs/BASELINE_TRACKER.md` | Confirm external-baseline status is accurate |
| `outputs/paper_tables/export_manifest.json` | Review BLOCKED list |
| `.github/workflows/test-openai-key.yml` | Confirm safe to expose |
