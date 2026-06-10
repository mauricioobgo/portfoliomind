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
from datetime import timezone
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

# Default cron expression for the morning trigger, exposed as a
# module-level constant so the operator-facing script
# (:file:`scripts/register_cron.sh`) can reference a single source of
# truth. The Bogota-local time the operator sees is 08:30 Mon-Fri;
# the container is UTC, and Colombia does not observe DST, so the
# UTC cron is fixed at ``30 13 * * 1-5`` year-round.
#
# (Bogota is UTC-5 year-round: 08:30 Bogota == 13:30 UTC. The
# ``day_of_week='mon-fri'`` qualifier in :func:`build_morning_trigger`
# encodes the Mon-Fri restriction at APScheduler level, not in the
# cron string. The cron string here is a *reference* for
# ``register_cron.sh`` which generates a Hermes-level ``hermes cron
# create`` command; APScheduler itself still owns the timezone
# handling via :data:`portfoliomind.time_utils.BOGOTA_TZ`.)
DEFAULT_MORNING_CRON: str = "30 13 * * 1-5"


@dataclass(frozen=True)
class ScheduleConfig:
    """Tunable knobs for the cron schedule.

    All defaults match the v4 spec. Tests can override any field by
    instantiating this class explicitly. The CLI exposes
    ``--morning-cron`` (5-field cron expression, overrides
    ``morning_hour``/``morning_minute`` when set), ``--returns-hh``
    and ``--returns-mm`` for ad-hoc overrides.

    The ``morning_cron`` field is the 5-field cron expression in
    *UTC* (because the container runs UTC). Card 8 added it so the
    operator can move the morning tick without rebuilding — e.g. to
    shift the run from 13:30 UTC to 14:00 UTC during US summer when
    the market effectively opens an hour later. The default
    ``"30 13 * * 1-5"`` is what :func:`build_morning_trigger` builds
    from ``morning_hour=8, morning_minute=30`` plus the
    ``America/Bogota`` timezone.

    When ``morning_cron`` is the empty string (the default), the
    scheduler uses the ``morning_hour``/``morning_minute`` fields
    with the ``day_of_week='mon-fri'`` restriction. When
    ``morning_cron`` is non-empty, it is parsed as a 5-field cron
    expression *in UTC* and used verbatim (the operator is
    responsible for translating to their local time).

    Colombia does not observe DST, so 13:30 UTC = 08:30 Bogota
    year-round. Operators in other timezones should adjust
    ``morning_cron`` accordingly and document the local-time
    equivalent in their :file:`AGENTS.md`.
    """

    morning_hour: int = DEFAULT_MORNING_HOUR
    morning_minute: int = DEFAULT_MORNING_MINUTE
    morning_cron: str = ""  # "" => use hour/minute + mon-fri restriction
    returns_hour: int = DEFAULT_RETURNS_HOUR
    returns_minute: int = DEFAULT_RETURNS_MINUTE


def build_morning_trigger(cfg: ScheduleConfig) -> CronTrigger:
    """The Mon-Fri 08:30 Bogota-local cron trigger for the morning job.

    Exposed as a separate function so a test can assert that the trigger
    is wired to the right timezone + the right day-of-week without
    spinning up the full scheduler.

    When ``cfg.morning_cron`` is non-empty, the trigger is built from
    the raw 5-field cron expression in UTC (the container is UTC).
    The operator is responsible for translating their local time to
    UTC when overriding this. When ``cfg.morning_cron`` is empty
    (the default), the trigger is built from ``morning_hour`` /
    ``morning_minute`` with a ``day_of_week='mon-fri'`` restriction,
    pinned to the ``America/Bogota`` timezone.
    """
    if cfg.morning_cron:
        # Parse the 5-field cron expression in UTC. APScheduler's
        # ``from_crontab`` accepts the standard 5-field syntax with
        # optional whitespace. We use it directly so the
        # ``morning_cron`` override is verbatim — the operator typed
        # exactly what they want.
        return CronTrigger.from_crontab(cfg.morning_cron, timezone=timezone.utc)
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
    "DEFAULT_MORNING_CRON",
]
