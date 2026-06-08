"""Scheduler for the PortfolioMind agent.

The scheduler drives two recurring jobs on a weekday / daily cadence:

* :func:`portfoliomind.scheduler.jobs.morning_run` — InvestingPro scrape →
  strategy engine picks → operator approval → XTB orders. Fires Mon–Fri at
  08:30 America/Bogota. Skips weekends and configured market holidays.
* :func:`portfoliomind.scheduler.jobs.refresh_returns` — pull current prices
  for everything in :data:`~portfoliomind.sheets.schema.RETURNS_TRACKER` via
  ``yfinance``, update Current Price / Current Value / Unrealized P&L ($) /
  Unrealized P&L (%) / Days Held. Prunes rows for tickers that no longer
  exist. Fires daily at 16:30 America/Bogota.

The cron triggers live in :mod:`portfoliomind.scheduler.loop` and the CLI
entry point is :mod:`scripts.run_scheduler`. The cron job itself is
registered with the Hermes cron scheduler (under the ``portfoliomind``
profile) — see :file:`AGENTS.md` and ``scripts/register_cron.py``.
"""

from __future__ import annotations

from .jobs import (
    HolidayCalendar,
    bogota_weekday,
    morning_run,
    refresh_returns,
)
from .loop import (
    DEFAULT_MORNING_HOUR,
    DEFAULT_MORNING_MINUTE,
    DEFAULT_RETURNS_HOUR,
    DEFAULT_RETURNS_MINUTE,
    build_scheduler,
)

__all__ = [
    "morning_run",
    "refresh_returns",
    "HolidayCalendar",
    "bogota_weekday",
    "build_scheduler",
    "DEFAULT_MORNING_HOUR",
    "DEFAULT_MORNING_MINUTE",
    "DEFAULT_RETURNS_HOUR",
    "DEFAULT_RETURNS_MINUTE",
]
