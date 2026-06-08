"""APScheduler wiring for the PortfolioMind scheduler.

We use :class:`apscheduler.schedulers.background.BackgroundScheduler` with
two cron triggers, both pinned to :data:`portfoliomind.time_utils.BOGOTA_TZ`
so the host's local time (UTC, in the production Docker image) is
irrelevant.

The triggers are exported as build functions so tests can introspect
the cron fields without standing up a real scheduler. The
:class:`apscheduler.schedulers.background.BackgroundScheduler` instance
is built by :func:`build_scheduler` and is meant to be ``.start()``ed
by the daemon CLI in :mod:`scripts.run_scheduler`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.triggers.cron import CronTrigger

from ..logging_setup import get_logger
from ..time_utils import BOGOTA_TZ
from .jobs import morning_run, refresh_returns

log = get_logger(__name__)


# --- Schedule constants -----------------------------------------------------

# Morning job fires Mon-Fri at 08:30 Bogota local. ``day_of_week='mon-fri'``
# is the APScheduler way of saying "skip Sat/Sun". The 08:30 hour matches
# the v4 spec exactly; the agent has until 09:00 Bogota to be ready
# before the XTB market opens.
DEFAULT_MORNING_HOUR: int = 8
DEFAULT_MORNING_MINUTE: int = 30

# Returns refresh fires daily at 16:30 Bogota (after the US market close
# at 16:00 ET == 15:00 Bogota in winter / 15:00 Bogota in summer — the
# extra 1.5h gives yfinance time to publish the closing print).
DEFAULT_RETURNS_HOUR: int = 16
DEFAULT_RETURNS_MINUTE: int = 30


@dataclass(frozen=True)
class ScheduleConfig:
    """Tunable knobs for the cron schedule.

    All defaults match the v4 spec. Tests can override any field by
    instantiating this class explicitly. The CLI exposes ``--morning-hh``,
    ``--morning-mm``, ``--returns-hh``, ``--returns-mm`` for ad-hoc
    overrides.
    """

    morning_hour: int = DEFAULT_MORNING_HOUR
    morning_minute: int = DEFAULT_MORNING_MINUTE
    returns_hour: int = DEFAULT_RETURNS_HOUR
    returns_minute: int = DEFAULT_RETURNS_MINUTE


def build_morning_trigger(cfg: ScheduleConfig) -> CronTrigger:
    """The Mon-Fri 08:30 Bogota-local cron trigger for the morning job.

    Exposed as a separate function so a test can assert that the trigger
    is wired to the right timezone + the right day-of-week without
    spinning up the full scheduler.
    """
    return CronTrigger(
        hour=cfg.morning_hour,
        minute=cfg.morning_minute,
        day_of_week="mon-fri",
        timezone=BOGOTA_TZ,
    )


def build_returns_trigger(cfg: ScheduleConfig) -> CronTrigger:
    """The daily 16:30 Bogota-local cron trigger for the returns refresh."""
    return CronTrigger(
        hour=cfg.returns_hour,
        minute=cfg.returns_minute,
        timezone=BOGOTA_TZ,
    )


def build_scheduler(
    cfg: Optional[ScheduleConfig] = None,
    *,
    scheduler_factory: Optional[type[BaseScheduler]] = None,
) -> BaseScheduler:
    """Build a scheduler with both jobs registered.

    Parameters
    ----------
    cfg:
        The :class:`ScheduleConfig` to use. Defaults match the v4 spec.
    scheduler_factory:
        Test seam — pass a :class:`apscheduler.schedulers.base.BaseScheduler`
        subclass to swap out :class:`BackgroundScheduler` for a
        :class:`BlockingScheduler` or a custom test scheduler. The
        returned instance is unstarted — call ``.start()`` on it.

    The two jobs are registered with human-readable ``id``s and coalesced
    to prevent pile-up if a previous run took longer than the trigger
    interval (misfire_grace_time=300s, coalesce=True). On misfire the
    agent logs a warning and skips the run.
    """
    if cfg is None:
        cfg = ScheduleConfig()
    factory = scheduler_factory or BackgroundScheduler
    scheduler = factory(timezone=BOGOTA_TZ)
    scheduler.add_job(
        _safe_morning_run,
        trigger=build_morning_trigger(cfg),
        id="portfoliomind.morning_run",
        name="PortfolioMind morning run (Mon-Fri 08:30 Bogota)",
        replace_existing=True,
        misfire_grace_time=300,
        coalesce=True,
    )
    scheduler.add_job(
        _safe_refresh_returns,
        trigger=build_returns_trigger(cfg),
        id="portfoliomind.refresh_returns",
        name="PortfolioMind returns refresh (daily 16:30 Bogota)",
        replace_existing=True,
        misfire_grace_time=600,
        coalesce=True,
    )
    return scheduler


def _safe_morning_run() -> None:
    """Cron-side wrapper around :func:`morning_run` that swallows
    exceptions. APScheduler's default behavior is to log and continue
    on a job failure, but the v4 spec wants explicit, structured
    error handling so a single morning failure doesn't keep showing
    up as a misfire for 12 hours.
    """
    try:
        outcome = morning_run()
    except Exception as e:  # noqa: BLE001
        log.error("morning_run_unhandled_exception type=%s err=%r", type(e).__name__, str(e)[:300])
        return
    log.info(outcome.summary_line())


def _safe_refresh_returns() -> None:
    """Cron-side wrapper around :func:`refresh_returns`. Same pattern as
    :func:`_safe_morning_run`."""
    try:
        outcome = refresh_returns()
    except Exception as e:  # noqa: BLE001
        log.error("refresh_returns_unhandled_exception type=%s err=%r", type(e).__name__, str(e)[:300])
        return
    log.info(outcome.summary_line())


__all__ = [
    "ScheduleConfig",
    "build_morning_trigger",
    "build_returns_trigger",
    "build_scheduler",
    "DEFAULT_MORNING_HOUR",
    "DEFAULT_MORNING_MINUTE",
    "DEFAULT_RETURNS_HOUR",
    "DEFAULT_RETURNS_MINUTE",
]
