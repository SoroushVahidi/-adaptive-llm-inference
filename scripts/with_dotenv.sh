#!/usr/bin/env bash
# Load repo-root .env into the environment, then run the given command.
# Usage (from repo root):
#   bash scripts/with_dotenv.sh python3 scripts/test_openai_key.py
#   bash scripts/with_dotenv.sh python scripts/run_build_real_routing_dataset.py --paired-outcomes --dataset gpqa_diamond --subset-size 5
#
# Requires: .env in the repository root (copy from .env.example). Never commit .env.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
ENV_FILE="${REPO_ROOT}/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing ${ENV_FILE}" >&2
  echo "Copy .env.example to .env and set OPENAI_API_KEY (and optional HF_TOKEN)." >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
exec "$@"
