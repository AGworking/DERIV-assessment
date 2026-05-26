"""Tiny .env loader, called once at the top of pipeline.py.

Uses python-dotenv if installed (the normal path — it's in requirements.txt).
Falls back to a small hand-rolled parser if python-dotenv isn't available, so
the pipeline still runs on a clean checkout that hasn't installed deps yet.

This is intentionally non-fatal: if `.env` doesn't exist, we silently move on
and the pipeline uses real environment variables / defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.paths import ROOT


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass  # Fall through to the minimal parser below.

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Don't overwrite values already set in the real environment.
        if key and key not in os.environ:
            os.environ[key] = value
