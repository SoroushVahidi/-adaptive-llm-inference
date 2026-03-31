"""Load repository-root ``.env`` into ``os.environ`` when ``python-dotenv`` is installed."""

from __future__ import annotations

from pathlib import Path

_LOADED = False


def try_load_repo_dotenv() -> None:
    """Load ``<repo>/.env`` if the file exists and ``python-dotenv`` is available.

    Uses ``override=False`` so variables already set in the environment (e.g. CI
    or an explicit ``export``) take precedence over ``.env``.
    """
    global _LOADED
    if _LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    _LOADED = True
