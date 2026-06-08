"""CLI entry point for the PortfolioMind scheduler.

Two modes:

* ``--once`` — run the morning job once and exit. Used by external cron
  triggers (the hermes cron wrapper, ad-hoc manual runs, and CI smoke
  tests).
* ``--daemon`` — start a long-running BackgroundScheduler that fires
  morning_run() Mon–Fri 08:30 Bogota and refresh_returns() daily
  16:30 Bogota. Stays running until SIGINT/SIGTERM.

Both modes share a common setup path: load env, configure logging,
build the config. The ``--daemon`` mode additionally installs a
SIGINT/SIGTERM handler that calls ``scheduler.shutdown(wait=False)``
so a Ctrl-C in the foreground exits cleanly.

Examples:

    uv run python scripts/run_scheduler.py --once
    uv run python scripts/run_scheduler.py --daemon
    uv run python scripts/run_scheduler.py --daemon --morning-hh 9 --morning-mm 0
"""

from __future__ import annotations

import argparse
import signal
from typing import NoReturn

from portfoliomind.config import ConfigError
from portfoliomind.logging_setup import get_logger, setup_logging
from portfoliomind.scheduler.jobs import morning_run
from portfoliomind.scheduler.loop import (
    DEFAULT_MORNING_HOUR,
    DEFAULT_MORNING_MINUTE,
    DEFAULT_RETURNS_HOUR,
    DEFAULT_RETURNS_MINUTE,
    ScheduleConfig,
    build_scheduler,
)

log = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="portfoliomind-run_scheduler",
        description="PortfolioMind scheduler (morning run + returns refresh).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--once",
        action="store_true",
        help="Run the morning job once and exit (for external cron triggers).",
    )
    mode.add_argument(
        "--daemon",
        action="store_true",
        help="Start the long-running scheduler; fires jobs on their cron schedule.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Root logger level (default: INFO).",
    )
    parser.add_argument(
        "--morning-hh",
        type=int,
        default=DEFAULT_MORNING_HOUR,
        help="Override the morning job's hour-of-day (Bogota local).",
    )
    parser.add_argument(
        "--morning-mm",
        type=int,
        default=DEFAULT_MORNING_MINUTE,
        help="Override the morning job's minute-of-hour (Bogota local).",
    )
    parser.add_argument(
        "--returns-hh",
        type=int,
        default=DEFAULT_RETURNS_HOUR,
        help="Override the returns refresh's hour-of-day (Bogota local).",
    )
    parser.add_argument(
        "--returns-mm",
        type=int,
        default=DEFAULT_RETURNS_MINUTE,
        help="Override the returns refresh's minute-of-hour (Bogota local).",
    )
    return parser.parse_args(argv)


def _run_once() -> int:
    """Run the morning job once and return the shell exit code.

    Exit codes:
      0 — ran, or skipped (weekend / holiday / no platform modules).
      2 — config error (the agent can't proceed without env).
      3 — sheets error (the agent can't reach the report sheet).
      4 — morning job ran but reported errors.
    """
    log.info("scheduler --once: starting")
    try:
        outcome = morning_run()
    except ConfigError as e:
        log.error("scheduler --once: config error: %s", e)
        return 2
    except Exception as e:  # noqa: BLE001
        log.error(
            "scheduler --once: unhandled error type=%s err=%r",
            type(e).__name__,
            str(e)[:300],
        )
        return 3
    log.info("scheduler --once: %s", outcome.summary_line())
    if outcome.status == "failed":
        return 4
    return 0


def _run_daemon(args: argparse.Namespace) -> int:
    """Start the BackgroundScheduler and block until SIGINT/SIGTERM."""
    cfg = ScheduleConfig(
        morning_hour=args.morning_hh,
        morning_minute=args.morning_mm,
        returns_hour=args.returns_hh,
        returns_minute=args.returns_mm,
    )
    log.info(
        "scheduler --daemon: starting "
        "(morning=%02d:%02d returns=%02d:%02d Bogota)",
        cfg.morning_hour,
        cfg.morning_minute,
        cfg.returns_hour,
        cfg.returns_minute,
    )
    scheduler = build_scheduler(cfg)
    scheduler.start()
    log.info("scheduler started")

    # Block the main thread. APScheduler's BackgroundScheduler runs in a
    # worker thread; without this loop the main thread would exit and
    # the scheduler would die with it. The signal handler flips the
    # event so we wake up cleanly.
    import threading

    stop = threading.Event()

    def _handle_signal(signum, _frame):  # noqa: ANN001
        log.info("scheduler --daemon: caught signal=%d, shutting down", signum)
        try:
            scheduler.shutdown(wait=False)
        finally:
            stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        stop.wait()
    except KeyboardInterrupt:
        log.info("scheduler --daemon: KeyboardInterrupt, shutting down")
        scheduler.shutdown(wait=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(level=args.log_level)
    if args.once:
        return _run_once()
    return _run_daemon(args)


def _entry() -> NoReturn:
    """Console-script entry point wrapper. Exits with the integer return
    code from :func:`main`."""
    raise SystemExit(main())


if __name__ == "__main__":
    _entry()
