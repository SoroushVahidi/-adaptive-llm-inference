# AGENTS.md

## Cursor Cloud specific instructions

### Repository Overview

**adaptive-llm-inference** — Adaptive test-time compute allocation for LLM reasoning under budget constraints. Python research codebase with a modular pipeline: datasets → models → baselines/allocators → evaluation.

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

### Structure

See `README.md` for full project structure. Core code lives under `src/`, configs in `configs/`, experiment runner in `scripts/`.

### Notes

- GSM8K is auto-downloaded on first run to `data/` (requires network). Subsequent runs use the HuggingFace cache.
- The dummy model is controlled by `correct_prob` and `seed` in config; useful for deterministic pipeline testing.
- `outputs/` and `data/` are gitignored.
