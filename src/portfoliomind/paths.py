"""Filesystem paths used across the agent.

Two pieces of state are persisted on disk between runs:

* ``SESSION_DIR`` — Playwright Chromium cookies (InvestingPro + XTB)
* ``SCREENSHOT_DIR`` — pre-trade screenshots (XTB)

Both default to project-local directories but are overridable via env so the
operator can point at e.g. an encrypted mount in production.
"""

from __future__ import annotations

import os
from pathlib import Path

from .time_utils import BOGOTA_TZ  # noqa: F401  (re-exported for convenience)


def resolve_path(env_var: str, default: str) -> Path:
    """Read an env var and resolve to an absolute Path, creating the dir if missing.

    Resolution order:
      1. env var value (if set and non-empty)
      2. ``default`` (relative paths are resolved against CWD)

    The directory is created with ``parents=True, exist_ok=True`` so callers
    never need to worry about a missing parent. Returns the absolute path.
    """
    raw = os.environ.get(env_var, "").strip() or default
    p = Path(raw).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def session_dir() -> Path:
    """Where Playwright cookies are persisted between runs."""
    return resolve_path("SESSION_DIR", "./sessions")


def screenshot_dir() -> Path:
    """Where pre-trade screenshots are saved."""
    return resolve_path("SCREENSHOT_DIR", "./screenshots")


__all__ = ["session_dir", "screenshot_dir", "resolve_path"]
