"""Timezone-aware timestamps for the agent.

All log lines, sheet timestamps, and order records use ``America/Bogota`` (UTC-5
year-round — Colombia does not observe DST). Never use ``datetime.now()``
without a tz: a naive timestamp in a multi-region sheet is a footgun.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final
from zoneinfo import ZoneInfo

BOGOTA_TZ: Final[ZoneInfo] = ZoneInfo("America/Bogota")


def now_bogota() -> datetime:
    """Return the current wall time in ``America/Bogota`` with tzinfo attached."""
    return datetime.now(tz=BOGOTA_TZ)


def utc_now() -> datetime:
    """Return the current UTC time with tzinfo attached."""
    return datetime.now(tz=timezone.utc)


def iso_now() -> str:
    """ISO 8601 timestamp in Bogota time, e.g. ``2026-06-08T08:30:00-05:00``."""
    return now_bogota().isoformat(timespec="seconds")


__all__ = ["BOGOTA_TZ", "now_bogota", "utc_now", "iso_now"]
