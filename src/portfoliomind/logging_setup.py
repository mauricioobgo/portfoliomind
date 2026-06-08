"""Structured logging for the agent.

Output format:
    ``2026-06-08T08:30:00-05:00 | INFO  | portfoliomind.sheets.client:42 | message``

One log line = one event. No multi-line stack traces in the stream output —
they go to stderr via the default exception handler. The format is chosen so
the AGENT_LOG tab can ingest log lines one-for-one without re-parsing.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

from .time_utils import now_bogota

_DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)-5s | %(name)s:%(lineno)d | %(message)s"
_DEFAULT_DATEFMT: Final[str] = "%Y-%m-%dT%H:%M:%S%z"


class _BogotaFormatter(logging.Formatter):
    """Formatter that stamps every record with the current Bogota wall time."""

    def formatTime(self, record, datefmt=None):  # noqa: N802 (logging API)
        # Override asctime so we get Bogota tz, not local time.
        return now_bogota().strftime(datefmt or _DEFAULT_DATEFMT)


_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once. Idempotent — safe to call from CLIs."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_BogotaFormatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quiet noisy 3rd-party loggers unless the user opts in via env.
    for noisy in ("google.auth", "googleapiclient", "urllib3"):
        logging.getLogger(noisy).setLevel("WARNING")

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Setup is lazy: callers don't need to call setup first,
    but if a CLI calls :func:`setup_logging` it will reconfigure the root handler."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
