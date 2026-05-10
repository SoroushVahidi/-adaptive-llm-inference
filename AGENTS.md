# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python research codebase (no web server, no Docker, no database).
All experiments run as CLI scripts; there are no services to start.

### Key commands

| Task | Command |
|------|---------|
| Install deps | `pip install -e ".[dev]"` |
| Lint | `python3 -m ruff check src/ tests/ scripts/` |
| Auto-fix lint | `python3 -m ruff check --fix src/ tests/ scripts/` |
| Tests | `python3 -m pytest` |
| Offline experiment | `python3 scripts/run_experiment.py --config configs/greedy.yaml` |

### Gotchas

- **`ruff` binary not on PATH**: Use `python3 -m ruff` instead of bare `ruff`.
- **`python` vs `python3`**: This environment only has `python3`. One pre-existing test (`test_token_budget_export_integration.py`) fails because it invokes `python` (not `python3`). This is a known repo issue, not an environment problem.
- **Pre-existing test failures**: 2 tests fail on `main` (`test_multi_action_oracle_eval_smoke` assertion count, `test_final_manuscript_export_includes_token_budget_router` missing `python` binary). These are not caused by environment setup.
- **No OpenAI API key needed for offline work**: All dummy-model experiments, tests, and policy evaluations run fully offline. Only `real_llm` experiments and dataset building need `OPENAI_API_KEY` set in `.env`.
- **Manuscript artifact generation** (`scripts/generate_final_manuscript_artifacts.py`) requires pre-built token-budget router outputs that depend on the OpenAI API; it will fail without those files.
- **Dataset auto-download**: GSM8K and MATH500 datasets auto-download from HuggingFace on first use. This works without `HF_TOKEN` but the token speeds up downloads.
