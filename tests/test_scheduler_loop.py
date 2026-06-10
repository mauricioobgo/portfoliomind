"""Tests for the card-8 scheduler-loop extensions.

Coverage:

* :func:`build_scheduler` registers BOTH jobs (the existing daily
  returns refresh and the morning run) — verifies card 8 didn't
  accidentally drop the returns job.
* :class:`ScheduleConfig` accepts a new ``morning_cron`` field; the
  default is empty (use the hour/minute path) and a non-empty value
  causes :func:`build_morning_trigger` to build the trigger in UTC
  from the raw 5-field expression.
* :func:`build_morning_trigger` honors the ``morning_cron`` field
  when set; when empty, it uses the hour/minute path with
  ``day_of_week='mon-fri'`` in ``America/Bogota``.
* :data:`DEFAULT_MORNING_CRON` is the canonical "08:30 Bogota Mon-Fri
  in UTC" expression. The constant exists so
  :file:`scripts/register_cron.sh` has a single source of truth.
"""

from __future__ import annotations

from datetime import timezone

import pytest
from apscheduler.triggers.cron import CronTrigger

from portfoliomind.scheduler.loop import (
    DEFAULT_MORNING_CRON,
    DEFAULT_MORNING_HOUR,
    DEFAULT_MORNING_MINUTE,
    ScheduleConfig,
    build_morning_trigger,
    build_returns_trigger,
    build_scheduler,
)
from portfoliomind.time_utils import BOGOTA_TZ


# --- Default schedule ------------------------------------------------------


class TestDefaultSchedule:
    """The default schedule matches the v4 spec: 08:30 Bogota Mon-Fri."""

    def test_default_morning_cron_constant(self):
        """``30 13 * * 1-5`` is 13:30 UTC = 08:30 Bogota (UTC-5
        year-round). The expression is fixed because Colombia does
        not observe DST."""
        assert DEFAULT_MORNING_CRON == "30 13 * * 1-5"

    def test_default_morning_hour_and_minute(self):
        assert DEFAULT_MORNING_HOUR == 8
        assert DEFAULT_MORNING_MINUTE == 30

    def test_default_morning_trigger_in_bogota(self):
        """Without a ``morning_cron`` override, the trigger is
        Bogota-local with ``day_of_week='mon-fri'``."""
        cfg = ScheduleConfig()
        trigger = build_morning_trigger(cfg)
        assert trigger.timezone == BOGOTA_TZ
        # CronTrigger fields are at positions [5,6,7] for hour/min/sec
        # in 0-indexed; day_of_week is at [4]. We re-render to a
        # string for the assertion.
        s = str(trigger)
        assert "day_of_week='mon-fri'" in s
        assert "hour='8'" in s
        assert "minute='30'" in s


# --- morning_cron override -------------------------------------------------


class TestMorningCronOverride:
    """When ``morning_cron`` is set, the trigger is built from the raw
    5-field cron expression in UTC."""

    def test_morning_cron_builds_in_utc(self):
        cfg = ScheduleConfig(morning_cron="30 13 * * 1-5")
        trigger = build_morning_trigger(cfg)
        assert trigger.timezone == timezone.utc
        s = str(trigger)
        # from_crontab stores day_of_week as '1-5' (the 1-5 form).
        assert "day_of_week='1-5'" in s
        assert "hour='13'" in s
        assert "minute='30'" in s

    def test_morning_cron_empty_uses_hour_minute(self):
        """``morning_cron=''`` is the sentinel that defers to
        morning_hour/morning_minute + day_of_week='mon-fri' +
        America/Bogota."""
        cfg = ScheduleConfig(morning_cron="")
        trigger = build_morning_trigger(cfg)
        assert trigger.timezone == BOGOTA_TZ
        s = str(trigger)
        assert "day_of_week='mon-fri'" in s
        assert "hour='8'" in s
        assert "minute='30'" in s

    def test_morning_cron_can_shift_time(self):
        """A custom ``morning_cron`` can move the run time without
        touching the hour/minute defaults."""
        cfg = ScheduleConfig(morning_cron="0 14 * * 1-5")
        trigger = build_morning_trigger(cfg)
        assert trigger.timezone == timezone.utc
        s = str(trigger)
        assert "hour='14'" in s
        assert "minute='0'" in s

    def test_morning_cron_ignores_hour_minute_fields(self):
        """When ``morning_cron`` is set, the hour/minute fields are
        not used by the trigger — but the dataclass still holds
        them for reference / display."""
        cfg = ScheduleConfig(
            morning_hour=99,  # nonsense
            morning_minute=99,  # nonsense
            morning_cron="30 13 * * 1-5",
        )
        trigger = build_morning_trigger(cfg)
        s = str(trigger)
        # The trigger was built from morning_cron, not the
        # hour/minute fields.
        assert "hour='13'" in s
        assert "minute='30'" in s
        assert "hour='99'" not in s

    def test_morning_cron_rejects_invalid_expression(self):
        """A malformed ``morning_cron`` raises from APScheduler at
        trigger build time, not at scheduler-build time. This is
        the documented behavior — the operator needs to see the
        error from the cron expression itself."""
        cfg = ScheduleConfig(morning_cron="not a cron")
        with pytest.raises((ValueError, Exception)):
            build_morning_trigger(cfg)


# --- build_scheduler has BOTH jobs -----------------------------------------


class TestBuildSchedulerHasBothJobs:
    """The card-8 extension must not break the existing two-job
    contract: morning run + returns refresh."""

    def test_build_scheduler_registers_two_jobs(self):
        scheduler = build_scheduler()
        jobs = scheduler.get_jobs()
        assert len(jobs) == 2
        ids = sorted(j.id for j in jobs)
        assert ids == sorted(
            [
                "portfoliomind.morning_run",
                "portfoliomind.refresh_returns",
            ]
        )

    def test_morning_job_uses_morning_trigger(self):
        """The ``morning_run`` job is wired to a cron trigger with
        the expected timezone + day-of-week."""
        scheduler = build_scheduler()
        job = next(j for j in scheduler.get_jobs() if j.id == "portfoliomind.morning_run")
        assert isinstance(job.trigger, CronTrigger)
        # The default trigger is Bogota-local + mon-fri.
        assert job.trigger.timezone == BOGOTA_TZ
        assert "day_of_week='mon-fri'" in str(job.trigger)

    def test_returns_job_uses_returns_trigger(self):
        scheduler = build_scheduler()
        job = next(j for j in scheduler.get_jobs() if j.id == "portfoliomind.refresh_returns")
        assert isinstance(job.trigger, CronTrigger)
        assert job.trigger.timezone == BOGOTA_TZ
        # Daily at 16:30 Bogota.
        assert "hour='16'" in str(job.trigger)
        assert "minute='30'" in str(job.trigger)

    def test_morning_cron_override_propagates_to_scheduler(self):
        """When ``morning_cron`` is set, the ``morning_run`` job's
        trigger reflects it (UTC, raw 5-field)."""
        cfg = ScheduleConfig(morning_cron="30 13 * * 1-5")
        scheduler = build_scheduler(cfg)
        job = next(j for j in scheduler.get_jobs() if j.id == "portfoliomind.morning_run")
        assert job.trigger.timezone == timezone.utc
        assert "day_of_week='1-5'" in str(job.trigger)
        # Returns trigger is unchanged.
        ret_job = next(
            j for j in scheduler.get_jobs() if j.id == "portfoliomind.refresh_returns"
        )
        assert ret_job.trigger.timezone == BOGOTA_TZ

    def test_returns_trigger_independent_of_morning_cron(self):
        """Setting ``morning_cron`` does not affect the returns trigger."""
        cfg = ScheduleConfig(morning_cron="30 13 * * 1-5")
        returns_trigger = build_returns_trigger(cfg)
        # Returns still 16:30 Bogota.
        assert returns_trigger.timezone == BOGOTA_TZ
        assert "hour='16'" in str(returns_trigger)
        assert "minute='30'" in str(returns_trigger)


# --- Returns trigger still uses Bogota-local 16:30 ----------------------


class TestReturnsTrigger:
    """The card 4 returns trigger is unchanged by card 8."""

    def test_default_returns_trigger(self):
        cfg = ScheduleConfig()
        trigger = build_returns_trigger(cfg)
        assert trigger.timezone == BOGOTA_TZ
        assert "hour='16'" in str(trigger)
        assert "minute='30'" in str(trigger)

    def test_returns_trigger_override(self):
        """The hour/minute fields still work for the returns job."""
        cfg = ScheduleConfig(returns_hour=17, returns_minute=15)
        trigger = build_returns_trigger(cfg)
        assert "hour='17'" in str(trigger)
        assert "minute='15'" in str(trigger)


# --- Public surface ------------------------------------------------------


class TestPublicSurface:
    """The card-8 additions to the public surface are exported."""

    def test_default_morning_cron_exported(self):
        """``DEFAULT_MORNING_CRON`` is importable from the loop
        module so ``scripts/register_cron.sh`` has a single source
        of truth."""
        from portfoliomind.scheduler import loop

        assert hasattr(loop, "DEFAULT_MORNING_CRON")
        assert loop.DEFAULT_MORNING_CRON == DEFAULT_MORNING_CRON

    def test_schedule_config_has_morning_cron_field(self):
        """The new ``morning_cron`` field is on the dataclass with a
        default of empty string (preserves card-4 callers)."""
        cfg = ScheduleConfig()
        assert hasattr(cfg, "morning_cron")
        assert cfg.morning_cron == ""

    def test_schedule_config_can_instantiate_with_all_overrides(self):
        """All card-8 fields can be passed positionally/by-keyword."""
        cfg = ScheduleConfig(
            morning_hour=8,
            morning_minute=30,
            morning_cron="",
            returns_hour=16,
            returns_minute=30,
        )
        assert cfg.morning_hour == 8
        assert cfg.morning_cron == ""
        assert cfg.returns_hour == 16
