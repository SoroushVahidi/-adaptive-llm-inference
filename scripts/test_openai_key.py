#!/usr/bin/env python3
"""Minimal OpenAI API auth probe. Reads OPENAI_API_KEY from the environment only.

Uses ``requests`` for a single GET to ``/v1/models``. Never prints the key.
Exit 0 on HTTP 2xx; exit 1 if the key is missing or the call fails.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.utils.repo_env import try_load_repo_dotenv  # noqa: E402

try_load_repo_dotenv()

MODELS_URL = "https://api.openai.com/v1/models"
TIMEOUT_SEC = 30


def _redact_blob(text: str, max_len: int = 120) -> str:
    if len(text) > max_len:
        text = text[:max_len] + "…"
    return re.sub(r"sk-[a-zA-Z0-9_-]{10,}", "[REDACTED_TOKEN]", text)


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        print("OPENAI_API_KEY not found in environment (empty or unset).")
        return 1

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.get(MODELS_URL, headers=headers, timeout=TIMEOUT_SEC)
    except requests.RequestException as exc:
        print("Key detected in environment. API request failed before a normal HTTP response.")
        print(f"Error class: {type(exc).__name__} ({_redact_blob(str(exc))})")
        return 1

    status = resp.status_code
    if 200 <= status < 300:
        print(
            f"Key detected in environment. API call returned HTTP {status} "
            "(authentication succeeded)."
        )
        return 0

    print(f"Key detected in environment. API call returned HTTP {status}.")
    try:
        err_json = resp.json()
        err_msg = err_json.get("error", {})
        if isinstance(err_msg, dict):
            msg = err_msg.get("message", str(err_msg))
        else:
            msg = str(err_msg)
        print(f"Error class: HTTP {status} ({_redact_blob(msg)})")
    except (json.JSONDecodeError, ValueError):
        print(f"Body preview (redacted): {_redact_blob(resp.text)}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
