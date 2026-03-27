# AGENTS.md

## Cursor Cloud specific instructions

### Repository Overview

This is the **adaptive-llm-inference** repository — an MIT-licensed Python project by Soroush Vahidi. As of initial setup, the repo contains only a `LICENSE` file with no application code, dependencies, or configuration.

### Development Environment

- **Python 3.12.3** is available system-wide at `python3`.
- No virtual environment or dependency file (`requirements.txt`, `pyproject.toml`, `setup.py`) exists yet. When one is added, the update script should be updated to install from it.
- No linter, test framework, or build system is configured yet.

### Running Services

There are no services to run. When application code is added, update this section with startup instructions.

### Notes for Future Agents

- If `requirements.txt` or `pyproject.toml` is added, run `pip install -r requirements.txt` or `pip install -e .` accordingly.
- If a linter (e.g. `ruff`, `flake8`) or test runner (e.g. `pytest`) is added, document the commands here.
- The repo name suggests this will be an ML/LLM project; common dependencies may include `torch`, `transformers`, `numpy`, etc.
