# AGENTS.md

## Cursor Cloud specific instructions

### Repository Overview

**adaptive-llm-inference** — Adaptive test-time compute allocation for LLM reasoning under budget constraints. Python research codebase with a modular pipeline: datasets → models → baselines/allocators → evaluation.

**Start here:** Read [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) for the full research goal, target venue (EAAI), baseline families, novelty positioning, and implementation plan. Read [`docs/BASELINE_TRACKER.md`](docs/BASELINE_TRACKER.md) for the status of every baseline.

**Mainline reference:** Treat `origin/main` (or a branch based on it) as the real codebase state. If local `main` shows only a handful of files, realign it to `origin/main` before using `main` as a reference.

### Development Environment

- **Python 3.12** (system `python3`); no `python` symlink — always use `python3`.
- Install: `pip install -e ".[dev]"` from repo root.
- Add `$HOME/.local/bin` to `PATH` if `pytest` / `ruff` are not found after install.

### Key Commands

| Task | Command |
|------|---------|
| Install deps | `pip install -e ".[dev]"` |
| Run tests | `pytest` |
| Lint | `ruff check src/ tests/ scripts/` |
| Auto-fix lint | `ruff check --fix src/ tests/ scripts/` |
| Run experiment | `python3 scripts/run_experiment.py --config configs/<name>.yaml` |

### Architecture Notes

- **Native baselines** live in `src/baselines/` (greedy, best-of-N, self-consistency).
- **External baselines** (TALE, BEST-Route) wrap official author code via thin adapters in `src/baselines/external/`. Official repos go under `external/<name>/.repo`.
- The experiment runner (`scripts/run_experiment.py`) uses an allocator + baseline to process queries. Allocators decide per-query sample counts; baselines execute the sampling strategy.
- GSM8K is auto-downloaded on first run to `data/` (requires network). Subsequent runs use the HuggingFace cache.
- The dummy model is controlled by `correct_prob` and `seed` in config; useful for deterministic pipeline testing.
- `outputs/` and `data/` are gitignored.
