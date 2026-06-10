"""Job bodies for the PortfolioMind scheduler.

The scheduler (see :mod:`portfoliomind.scheduler.loop`) registers two
recurring jobs:

* :func:`morning_run` — fires Mon–Fri 08:30 America/Bogota. Drives the
  InvestingPro scrape → strategy pick → operator approval → XTB order
  pipeline. The InvestingPro and XTB modules are imported lazily inside
  :func:`morning_run`; if a module isn't installed yet (e.g. card 2 / card 3
  haven't landed) the job logs a ``not_implemented`` line to
  ``AGENT_LOG`` and exits cleanly so the schedule keeps ticking.

* :func:`refresh_returns` — fires daily 16:30 America/Bogota. Pulls current
  prices for every ticker in ``RETURNS_TRACKER`` via ``yfinance``, updates
  the derived columns (Current Price, Current Value, Unrealized P&L,
  Days Held), and prunes rows for tickers that ``yfinance`` can no longer
  resolve.

All times are Bogota-local. The cron triggers live in
:mod:`portfoliomind.scheduler.loop`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Optional, Protocol

from ..config import PortfoliomindConfig
from ..logging_setup import get_logger
from ..sheets.client import SheetsClient
from ..sheets.schema import AGENT_LOG, RETURNS_TRACKER
from ..time_utils import BOGOTA_TZ, iso_now, now_bogota

log = get_logger(__name__)


# --- Bogota weekday + holiday helpers ---------------------------------------


def bogota_weekday(today: Optional[datetime] = None) -> int:
    """Return the Bogota-local weekday as ``0=Monday .. 6=Sunday``.

    Wrapping :meth:`datetime.weekday` so callers don't have to remember
    the timezone dance. The default argument is computed lazily so a
    test can pass a fixed :class:`datetime` without monkeypatching
    :func:`datetime.now`.
    """
    if today is None:
        today = now_bogota()
    return today.weekday()


@dataclass(frozen=True)
class HolidayCalendar:
    """A tiny holiday calendar — set of dates (Bogota-local) on which the
    morning job must be skipped.

    We don't try to track every market holiday; this is the minimal
    "skip these days" affordance for the agent. The agent operator
    configures which dates to skip via ``PORTFOLIOMIND_HOLIDAYS`` (comma-
    separated ``YYYY-MM-DD``). Colombian market holidays are the obvious
    default; the spec leaves the exact list to the operator.
    """

    skipped_dates: frozenset[date] = field(default_factory=frozenset)

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> "HolidayCalendar":
        """Build a calendar from ``PORTFOLIOMIND_HOLIDAYS``.

        Format: comma-separated ``YYYY-MM-DD`` list, e.g.
        ``"2026-01-01,2026-05-01,2026-07-20"``. Bad entries are logged
        and skipped, not fatal — the agent never wants a typo to take
        the morning job offline.
        """
        if env is None:
            import os
            env = dict(os.environ)
        raw = (env.get("PORTFOLIOMIND_HOLIDAYS") or "").strip()
        if not raw:
            return cls(skipped_dates=frozenset())
        out: set[date] = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out.add(date.fromisoformat(token))
            except ValueError:
                log.warning(
                    "holiday_parse_skip token=%r reason=invalid_iso8601_date", token
                )
        return cls(skipped_dates=frozenset(out))

    def is_holiday(self, day: date) -> bool:
        return day in self.skipped_dates


def is_morning_trading_day(
    today: Optional[datetime] = None,
    *,
    calendar: HolidayCalendar = HolidayCalendar(),
) -> bool:
    """True if the morning job should fire on the given Bogota-local date.

    Weekday check: Mon–Fri (weekday 0–4). Holiday check: delegated to the
    supplied :class:`HolidayCalendar` (default = no holidays).
    """
    if today is None:
        today = now_bogota()
    if bogota_weekday(today) >= 5:
        return False
    return not calendar.is_holiday(today.date())


# --- Lazy platform runner protocol -----------------------------------------


class _PlatformRunner(Protocol):
    """Callable signature the morning job expects from card 2 / card 3.

    Card 4 defines this protocol; card 2 / card 3 will implement it. The
    morning job does not import the platform modules at module load time
    so the absence of those modules is non-fatal — the job logs
    ``not_implemented`` and exits cleanly.
    """

    def __call__(self, ctx: "MorningContext") -> "MorningResult": ...


class _SheetsLike(Protocol):
    """Structural type for the Sheets operations the scheduler needs.

    Production code passes a real :class:`~portfoliomind.sheets.client.SheetsClient`.
    Tests pass a fake with the same method shapes. We can't reuse
    :class:`SheetsClient` directly because the production class has a
    lot of surface area (sheet creation, batch updates, etc.) that the
    scheduler never touches, and forcing tests to mock all of it would
    be noise.
    """

    def read_range(
        self, sheet_id: str, tab_name: str, range_a1: str
    ) -> list[list[str]]: ...

    def write_range(
        self,
        sheet_id: str,
        tab_name: str,
        range_a1: str,
        values: list[list[Any]],
    ) -> None: ...

    def append_rows(
        self, sheet_id: str, tab_name: str, values: list[list[Any]]
    ) -> int: ...


@dataclass
class MorningContext:
    """The bag of state passed to a platform runner.

    Kept intentionally small: a runner should be able to do its full job
    given only the config + sheets + the current Bogota timestamp. The
    scheduler fills it in once per tick and passes it to each runner.
    """

    config: Optional[PortfoliomindConfig]
    sheets: _SheetsLike
    sheet_id: str
    today: datetime  # Bogota-local, tzinfo attached
    log_to_sheet: Callable[[str, str], None]  # (level, message) -> writes AGENT_LOG row


@dataclass
class MorningResult:
    """The bag of state returned by a platform runner.

    ``skipped`` covers the case where the runner decided not to do work
    (e.g. no new picks from InvestingPro, or operator said "no" to every
    pick — both legitimate no-ops). ``error`` is the human-readable failure
    reason when the runner raised; the morning job will log it to
    AGENT_LOG and to the Discord home channel via the alert hook.
    """

    runner: str  # "card2" / "card3" / etc
    picks_scraped: int = 0
    orders_placed: int = 0
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""

    def ok(self) -> bool:
        return not self.error


def _noop_log(level: str, message: str) -> None:
    """A log-to-sheet function that silently no-ops if the sheets client
    is missing. Used when we run in a unit-test context."""
    log.log(
        {"INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}.get(
            level.upper(), logging.INFO
        ),
        message,
    )


def _append_agent_log(
    sheets: _SheetsLike, sheet_id: str, level: str, module: str, message: str
) -> None:
    """Append a row to the AGENT_LOG tab. Best-effort: a Sheets failure
    here must not break the morning job.

    The format matches :data:`~portfoliomind.sheets.schema.AGENT_LOG`:
    ``Timestamp, Level, Module, Message``.
    """
    row = [iso_now(), level.upper(), module, message]
    try:
        sheets.append_rows(sheet_id, AGENT_LOG, [row])
    except Exception as e:  # noqa: BLE001  (best-effort)
        log.warning(
            "agent_log_append_failed sheet_id=%s err_type=%s err=%r",
            sheet_id,
            type(e).__name__,
            str(e)[:200],
        )


# --- morning_run ------------------------------------------------------------


@dataclass
class MorningOutcome:
    """What :func:`morning_run` returns to the cron driver / Discord alerter.

    Fields are designed so the Discord alert builder can format a single
    human-readable line without re-reading the run. ``status`` is one of
    ``"ran"`` / ``"skipped_weekend"`` / ``"skipped_holiday"`` /
    ``"no_platform_modules"`` / ``"failed"``.
    """

    status: str
    started_at: str
    finished_at: str
    picks_scraped: int = 0
    orders_placed: int = 0
    errors: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        if self.status == "ran":
            return (
                f"morning_run OK: picks={self.picks_scraped} orders={self.orders_placed} "
                f"errors={len(self.errors)}"
            )
        if self.status == "skipped_weekend":
            return "morning_run SKIP: weekend"
        if self.status == "skipped_holiday":
            return "morning_run SKIP: holiday"
        if self.status == "no_platform_modules":
            return "morning_run SKIP: card 2/3 modules not implemented yet"
        if self.status == "failed":
            return f"morning_run FAIL: {len(self.errors)} error(s); first={self.errors[0]!r}"
        return f"morning_run {self.status}"


def _try_import_card2() -> Optional[_PlatformRunner]:
    """Try to import the card 2 (InvestingPro) runner. Returns ``None``
    if the module is missing so :func:`morning_run` can degrade
    gracefully.

    Card 2 is expected to register a callable at
    ``portfoliomind.investingpro.runner.run_morning``. When that lands,
    card 4 picks it up automatically; until then, the morning job logs
    ``not_implemented`` and exits cleanly.
    """
    try:
        from ..investingpro import runner as inv_runner  # type: ignore[import-not-found]
    except ImportError:
        return None
    return getattr(inv_runner, "run_morning", None)


def _try_import_card3() -> Optional[_PlatformRunner]:
    """Try to import the card 3 (XTB) runner. Same pattern as card 2 —
    see :func:`_try_import_card2`."""
    try:
        from ..xtb import runner as xtb_runner  # type: ignore[import-not-found]
    except ImportError:
        return None
    return getattr(xtb_runner, "run_morning", None)


def _try_import_strategy() -> Optional[_PlatformRunner]:
    """Try to import the card 8 (strategy) runner.

    The strategy runner is the third leg of the morning pipeline: it
    scores the universe (card 6), sizes the candidates (card 7), posts
    to Discord for operator approval (card 7), and persists the
    approved subset to ``APPROVED_TRADES`` for the XTB runner to pick
    up on its next tick.

    Same lazy-import pattern as card 2/3: if the module isn't installed
    (e.g. card 8 hasn't shipped) the job logs ``not_implemented`` and
    exits cleanly. The card 8 module's :func:`run_morning` is also
    defensively ``not_implemented``-aware at the strategy-layer: even
    if card 8 ships, the inner signals/sizer/approval imports may
    still be pending, and the runner will degrade gracefully.
    """
    try:
        from ..strategy_runner import run_morning as strategy_run  # type: ignore[import-not-found]
    except ImportError:
        return None
    return strategy_run


def morning_run(
    *,
    config: Optional[PortfoliomindConfig] = None,
    sheets: Optional[_SheetsLike] = None,
    sheet_id: Optional[str] = None,
    today: Optional[datetime] = None,
    calendar: Optional[HolidayCalendar] = None,
) -> MorningOutcome:
    """Drive the morning pipeline for the current Bogota-local day.

    The function is the seam between the scheduler and the platform
    runners. It is deliberately defensive: every failure mode is
    converted into a :class:`MorningOutcome` rather than an exception so
    the cron wrapper can always format a Discord alert.

    Parameters
    ----------
    config:
        The :class:`PortfoliomindConfig` to use. Built from the env when
        omitted. Pass an explicit instance in tests.
    sheets:
        A pre-built :class:`SheetsClient`. Built from ``config`` when
        omitted.
    today:
        Override the "current Bogota time" — useful in tests and
        when a retry needs to pretend it's a different day.
    calendar:
        The :class:`HolidayCalendar` to honor. Defaults to the env-
        configured calendar.

    Returns
    -------
    :class:`MorningOutcome`
        Always returns one. Never raises.
    """
    started_at = iso_now()
    if today is None:
        today = now_bogota()
    if calendar is None:
        calendar = HolidayCalendar.from_env()

    if bogota_weekday(today) >= 5:
        return MorningOutcome(
            status="skipped_weekend",
            started_at=started_at,
            finished_at=iso_now(),
        )
    if calendar.is_holiday(today.date()):
        return MorningOutcome(
            status="skipped_holiday",
            started_at=started_at,
            finished_at=iso_now(),
        )

    # Lazy config + sheets build so a unit test can pass them in and
    # skip env entirely. The order: if both config and sheets are
    # missing, try the env. Otherwise honor what's provided. sheet_id
    # is derived from config when not given.
    if config is None and sheets is None:
        try:
            config = PortfoliomindConfig.from_env()
        except Exception as e:  # noqa: BLE001
            return MorningOutcome(
                status="failed",
                started_at=started_at,
                finished_at=iso_now(),
                errors=[f"config load failed: {type(e).__name__}: {e}"],
            )
    if config is not None and sheet_id is None:
        sheet_id = config.google_sheet_id
    if sheet_id is None:
        sheet_id = ""
    if sheets is None and config is not None:
        try:
            sheets = SheetsClient.from_config(config)
        except Exception as e:  # noqa: BLE001
            return MorningOutcome(
                status="failed",
                started_at=started_at,
                finished_at=iso_now(),
                errors=[f"sheets client build failed: {type(e).__name__}: {e}"],
            )
    if sheets is None:
        return MorningOutcome(
            status="failed",
            started_at=started_at,
            finished_at=iso_now(),
            errors=["morning_run: no SheetsClient provided and no config to build one from"],
        )

    log_to_sheet = lambda level, msg: _append_agent_log(  # noqa: E731
        sheets, sheet_id, level, "scheduler.jobs", msg
    )

    # Lazy-import the platform runners. If neither is present yet, the
    # job logs a clear "not implemented" line and exits cleanly. This
    # lets card 4 ship ahead of cards 2/3 without breaking the schedule.
    # The card 8 (strategy) runner is also lazy-imported: when the
    # card-6/7 modules are still missing the strategy runner returns
    # ``status='not_implemented'`` cleanly so the morning job keeps
    # ticking. See ``_try_import_strategy`` and
    # ``portfoliomind.strategy_runner`` for details.
    inv_runner = _try_import_card2()
    xtb_runner = _try_import_card3()
    strategy_runner = _try_import_strategy()
    if inv_runner is None and xtb_runner is None and strategy_runner is None:
        msg = "morning_run: no runners (card 2/3/8 modules) registered — skipping"
        log.info(msg)
        log_to_sheet("INFO", msg)
        return MorningOutcome(
            status="no_platform_modules",
            started_at=started_at,
            finished_at=iso_now(),
        )

    ctx = MorningContext(
        config=config,
        sheets=sheets,
        sheet_id=sheet_id,
        today=today,
        log_to_sheet=log_to_sheet,
    )

    errors: list[str] = []
    picks_scraped = 0
    orders_placed = 0

    if inv_runner is not None:
        try:
            res: MorningResult = inv_runner(ctx)
            picks_scraped += res.picks_scraped
            if not res.ok():
                errors.append(f"card2:{res.error}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"card2 raised: {type(e).__name__}: {e}")

    # Card 8 (strategy) runs BEFORE card 3 (XTB execution) so that the
    # approved trades are persisted to APPROVED_TRADES first; the XTB
    # runner then reads them on the same tick. The strategy runner
    # returns a StrategyResult (its own dataclass with the same
    # shape) which we unwrap into the morning summary fields. The
    # ``picks_scraped`` field is repurposed to count the candidates
    # score_universe produced; ``orders_placed`` counts the rows
    # persisted to APPROVED_TRADES.
    if strategy_runner is not None:
        try:
            # The strategy runner's return type is StrategyResult, not
            # MorningResult — it has the same fields plus an extra
            # ``errors`` list. We can't reuse the _PlatformRunner
            # Protocol because it pins the return type to
            # MorningResult, so we cast to Any and unwrap manually.
            strat_res: Any = strategy_runner(ctx)
            picks_scraped += getattr(strat_res, "picks_scraped", 0)
            orders_placed += getattr(strat_res, "orders_placed", 0)
            if not getattr(strat_res, "ok", lambda: True)():
                err_text = getattr(strat_res, "error", "") or (
                    strat_res.errors[0] if getattr(strat_res, "errors", None) else ""
                )
                if err_text:
                    errors.append(f"card8:{err_text}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"card8 raised: {type(e).__name__}: {e}")

    if xtb_runner is not None:
        try:
            res = xtb_runner(ctx)
            orders_placed += res.orders_placed
            if not res.ok():
                errors.append(f"card3:{res.error}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"card3 raised: {type(e).__name__}: {e}")

    if errors:
        # The job ran to completion but produced errors — log them and
        # surface as a soft failure (the cron wrapper can still alert).
        for e in errors:
            log_to_sheet("ERROR", e)
        log.error("morning_run partial-failure errors=%d", len(errors))
        return MorningOutcome(
            status="failed",
            started_at=started_at,
            finished_at=iso_now(),
            picks_scraped=picks_scraped,
            orders_placed=orders_placed,
            errors=errors,
        )

    msg = f"morning_run OK picks={picks_scraped} orders={orders_placed}"
    log.info(msg)
    log_to_sheet("INFO", msg)
    return MorningOutcome(
        status="ran",
        started_at=started_at,
        finished_at=iso_now(),
        picks_scraped=picks_scraped,
        orders_placed=orders_placed,
    )


# --- refresh_returns --------------------------------------------------------


@dataclass(frozen=True)
class TickerRow:
    """A single row from ``RETURNS_TRACKER`` after the first parse pass.

    Holds the original index (for re-writing) and the columns we need
    to update in place. Date math uses :class:`datetime.date` so a
    timestamp like ``"2026-06-08T08:30:00-05:00"`` from
    :func:`portfoliomind.time_utils.iso_now` parses cleanly via
    :meth:`datetime.fromisoformat`.
    """

    row_index: int  # 1-indexed; row 1 is the header
    ticker: str
    entry_date: date
    entry_price: float
    qty: float
    dividend_received: float = 0.0

    @classmethod
    def from_row(cls, row_index: int, row: list[str]) -> "TickerRow":
        """Parse a raw row from the sheet. Raises :class:`ValueError` on
        malformed input — caller decides whether to skip or hard-fail.
        """
        ticker = (row[0] or "").strip()
        if not ticker:
            raise ValueError(f"empty ticker at row {row_index}")
        entry_date_raw = (row[4] or "").strip()  # 0=Ticker,4=Entry Date
        try:
            entry_date = _parse_date(entry_date_raw)
        except ValueError as e:
            raise ValueError(
                f"bad entry_date {entry_date_raw!r} for ticker {ticker!r}: {e}"
            ) from e
        try:
            entry_price = float(row[5])  # 5=Entry Price
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"bad entry_price {row[5]!r} for ticker {ticker!r}: {e}"
            ) from e
        try:
            qty = float(row[7])  # 7=Qty
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"bad qty {row[7]!r} for ticker {ticker!r}: {e}"
            ) from e
        # Dividend is optional; default 0.
        try:
            dividend = float(row[15]) if len(row) > 15 and row[15] else 0.0
        except (TypeError, ValueError):
            dividend = 0.0
        return cls(
            row_index=row_index,
            ticker=ticker,
            entry_date=entry_date,
            entry_price=entry_price,
            qty=qty,
            dividend_received=dividend,
        )


def _parse_date(s: str) -> date:
    """Parse a date that came out of a Sheets cell. Accepts both bare
    ``YYYY-MM-DD`` and full ISO 8601 timestamps. Returns a :class:`date`.
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("empty date")
    # ISO 8601 first.
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        pass
    # Common fallbacks.
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date format: {s!r}")


@dataclass
class RefreshOutcome:
    """What :func:`refresh_returns` returns to the cron driver."""

    status: str  # "ran" / "skipped" / "failed" / "no_sheet"
    tickers_refreshed: int = 0
    tickers_pruned: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def summary_line(self) -> str:
        if self.status == "ran":
            return (
                f"refresh_returns OK refreshed={self.tickers_refreshed} "
                f"pruned={self.tickers_pruned} errors={len(self.errors)}"
            )
        if self.status == "skipped":
            return "refresh_returns SKIP"
        if self.status == "no_sheet":
            return "refresh_returns SKIP: no tickers in RETURNS_TRACKER"
        if self.status == "failed":
            return (
                f"refresh_returns FAIL errors={len(self.errors)} "
                f"first={self.errors[0]!r}"
            )
        return f"refresh_returns {self.status}"


def _yfinance_lookup(
    tickers: list[str], *, price_fetcher: Optional[Callable[[str], Optional[float]]] = None
) -> dict[str, Optional[float]]:
    """Look up current prices for a list of tickers.

    ``price_fetcher`` is the injection point for tests. In production we
    use :func:`_default_yfinance_lookup`, which calls ``yfinance`` once
    per ticker (the simple, deterministic, mockable shape).

    Returns a dict ``{ticker: price_or_None}``. ``None`` means "yfinance
    could not resolve the ticker" and is the signal for the caller to
    prune the row.
    """
    if price_fetcher is not None:
        return {t: price_fetcher(t) for t in tickers}
    return _default_yfinance_lookup(tickers)


def _default_yfinance_lookup(tickers: list[str]) -> dict[str, Optional[float]]:
    """Production yfinance path. Lazy-imports ``yfinance`` so the
    scheduler can be imported in environments where ``yfinance`` is not
    installed (CI without network, for example).
    """
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "yfinance is required for refresh_returns in production; install with `uv add yfinance`"
        ) from e
    out: dict[str, Optional[float]] = {}
    for t in tickers:
        try:
            data = yf.Ticker(t).history(period="1d", auto_adjust=False)
        except Exception as e:  # noqa: BLE001
            log.warning("yfinance_call_failed ticker=%s err_type=%s", t, type(e).__name__)
            out[t] = None
            continue
        if data is None or data.empty:
            out[t] = None
            continue
        try:
            # ``Close`` is the most reliable column. yfinance returns it
            # as a float for the most recent row.
            price = float(data["Close"].iloc[-1])
        except (KeyError, IndexError, TypeError, ValueError):
            out[t] = None
            continue
        out[t] = price
    return out


def refresh_returns(
    *,
    config: Optional[PortfoliomindConfig] = None,
    sheets: Optional[_SheetsLike] = None,
    sheet_id: Optional[str] = None,
    today: Optional[datetime] = None,
    price_fetcher: Optional[Callable[[str], Optional[float]]] = None,
) -> RefreshOutcome:
    """Pull current prices and update ``RETURNS_TRACKER`` in place.

    Behavior:
    1. Read all rows of ``RETURNS_TRACKER`` (header is row 1).
    2. For each ticker, look up the current price.
    3. Update Current Price (col G), Current Value (col J),
       Unrealized P&L $ (col K), Unrealized P&L % (col L),
       Days Held (col M), and Total Return (col Q).
    4. Prune rows for tickers that yfinance can no longer resolve
       (writing the surviving rows to a contiguous block + clearing
       the tail).

    Parameters
    ----------
    config, sheets, sheet_id:
        Override hooks. The function uses ``config.google_sheet_id``
        when ``sheet_id`` is not provided. Tests pass an explicit
        ``sheet_id`` (or pass both ``config`` and ``sheet_id``) to
        avoid going through env.
    today:
        Override the "current Bogota time" — useful in tests.
    price_fetcher:
        Test seam — a callable ``ticker -> price | None`` that replaces
        :func:`_default_yfinance_lookup` when set.

    Returns
    -------
    :class:`RefreshOutcome`
        Always returns one. Never raises.
    """
    started_at = iso_now()
    if today is None:
        today = now_bogota()
    if config is None and sheets is None:
        # Both missing — try the env.
        try:
            config = PortfoliomindConfig.from_env()
        except Exception as e:  # noqa: BLE001
            return RefreshOutcome(
                status="failed",
                started_at=started_at,
                finished_at=iso_now(),
                errors=[f"config load failed: {type(e).__name__}: {e}"],
            )
    if config is not None and sheet_id is None:
        sheet_id = config.google_sheet_id
    if sheet_id is None:
        sheet_id = ""
    if sheets is None and config is not None:
        try:
            sheets = SheetsClient.from_config(config)
        except Exception as e:  # noqa: BLE001
            return RefreshOutcome(
                status="failed",
                started_at=started_at,
                finished_at=iso_now(),
                errors=[f"sheets client build failed: {type(e).__name__}: {e}"],
            )
    if sheets is None:
        return RefreshOutcome(
            status="failed",
            started_at=started_at,
            finished_at=iso_now(),
            errors=["refresh_returns: no SheetsClient provided and no config to build one from"],
        )

    log_to_sheet = lambda level, msg: _append_agent_log(  # noqa: E731
        sheets, sheet_id or "", level, "scheduler.jobs", msg
    )

    try:
        rows = sheets.read_range(sheet_id, RETURNS_TRACKER, "A2:R")
    except Exception as e:  # noqa: BLE001
        msg = f"refresh_returns: read_range failed: {type(e).__name__}: {e}"
        log_to_sheet("ERROR", msg)
        log.error(msg)
        return RefreshOutcome(
            status="failed",
            started_at=started_at,
            finished_at=iso_now(),
            errors=[msg],
        )

    if not rows:
        return RefreshOutcome(
            status="no_sheet",
            started_at=started_at,
            finished_at=iso_now(),
        )

    # Parse every row. Bad rows are pruned (we don't have a way to fix
    # malformed entries; better to drop them and let the operator
    # re-enter on the next manual run).
    parsed: list[TickerRow] = []
    parse_errors: list[str] = []
    for i, row in enumerate(rows, start=2):  # 2 = first data row
        # Normalize the row to the canonical 19-column width. Sheets can
        # return shorter rows for trailing-empty cells.
        row = list(row) + [""] * (19 - len(row))
        try:
            parsed.append(TickerRow.from_row(i, row))
        except ValueError as e:
            # The error text is shaped for the AGENT_LOG tab — it
            # starts with "parse-skip:" so the on-call human can grep
            # for parse issues specifically.
            err = f"parse-skip: row {i}: {e}"
            parse_errors.append(err)
            log.warning("refresh_returns parse_skip row=%d err=%s", i, e)
    if parse_errors:
        for e in parse_errors:
            log_to_sheet("WARNING", f"refresh_returns: {e}")

    if not parsed:
        return RefreshOutcome(
            status="no_sheet",
            started_at=started_at,
            finished_at=iso_now(),
            errors=parse_errors,
        )

    # yfinance lookup (or the test fetcher).
    tickers = [r.ticker for r in parsed]
    try:
        prices = _yfinance_lookup(tickers, price_fetcher=price_fetcher)
    except Exception as e:  # noqa: BLE001
        msg = f"refresh_returns: price lookup failed: {type(e).__name__}: {e}"
        log_to_sheet("ERROR", msg)
        log.error(msg)
        return RefreshOutcome(
            status="failed",
            started_at=started_at,
            finished_at=iso_now(),
            errors=parse_errors + [msg],
        )

    # Build the new row data + track which tickers are pruned.
    new_rows: list[list[str]] = []
    pruned: list[str] = []
    for r in parsed:
        price = prices.get(r.ticker)
        if price is None or price <= 0:
            pruned.append(r.ticker)
            continue
        # Pad the original row to 19 columns so we can update by index.
        original = list(rows[r.row_index - 2]) + [""] * (19 - len(rows[r.row_index - 2]))
        new_row = _update_ticker_row(
            original=original,
            row=r,
            current_price=price,
            today=today,
        )
        new_rows.append(new_row)

    # Write the updated rows back as a contiguous block starting at
    # A2 (row 1 is the header). We need to:
    # 1. Overwrite the existing block with the new data.
    # 2. Clear any rows that remain after the new block (pruned-from-
    #    middle case keeps the sheet compact).
    try:
        if new_rows:
            end_col = _col_letter(19)
            sheets.write_range(
                sheet_id, RETURNS_TRACKER, f"A2:{end_col}{1 + len(new_rows)}", new_rows
            )
        # Clear the tail (rows after the new block).
        if len(new_rows) < len(rows):
            tail_start = 2 + len(new_rows)
            tail_end = 1 + len(rows)
            sheets.write_range(
                sheet_id,
                RETURNS_TRACKER,
                f"A{tail_start}:S{tail_end}",
                [[""] * 19] * (tail_end - tail_start + 1),
            )
    except Exception as e:  # noqa: BLE001
        msg = f"refresh_returns: write_range failed: {type(e).__name__}: {e}"
        log_to_sheet("ERROR", msg)
        log.error(msg)
        return RefreshOutcome(
            status="ran",
            started_at=started_at,
            finished_at=iso_now(),
            tickers_refreshed=len(new_rows),
            tickers_pruned=len(pruned),
            errors=parse_errors + [msg],
        )

    summary = (
        f"refresh_returns OK refreshed={len(new_rows)} pruned={len(pruned)} "
        f"parse_errors={len(parse_errors)}"
    )
    if pruned:
        log_to_sheet("INFO", f"refresh_returns: pruned {len(pruned)} ticker(s): {','.join(pruned)}")
    log_to_sheet("INFO", summary)
    log.info(summary)
    return RefreshOutcome(
        status="ran",
        started_at=started_at,
        finished_at=iso_now(),
        tickers_refreshed=len(new_rows),
        tickers_pruned=len(pruned),
        errors=parse_errors,
    )


def _update_ticker_row(
    *,
    original: list[str],
    row: TickerRow,
    current_price: float,
    today: datetime,
) -> list[str]:
    """Compute the new values for a single ``RETURNS_TRACKER`` row.

    Updates columns G (Current Price), J (Current Value), K (Unrealized
    P&L $), L (Unrealized P&L %), M (Days Held), Q (Total Return), and
    R (Status) based on the freshly fetched price. All other columns
    pass through unchanged.
    """
    # Make a 19-element copy (A..S).
    out = list(original) + [""] * (19 - len(original))
    out[6] = f"{current_price:.4f}"  # G = Current Price
    entry_value = row.entry_price * row.qty
    current_value = current_price * row.qty
    out[9] = f"{current_value:.2f}"  # J = Current Value
    pnl_dollars = current_value - entry_value
    out[10] = f"{pnl_dollars:.2f}"  # K = Unrealized P&L $
    pnl_pct = (pnl_dollars / entry_value * 100.0) if entry_value else 0.0
    out[11] = f"{pnl_pct:.2f}"  # L = Unrealized P&L %
    days_held = (today.date() - row.entry_date).days
    out[12] = str(max(days_held, 0))  # M = Days Held
    # Q = Total Return = (pnl + dividend) / entry_value. We don't track
    # dividends per-day, so we use the cumulative dividend received
    # recorded on the row.
    total_return_pct = (
        ((pnl_dollars + row.dividend_received) / entry_value * 100.0) if entry_value else 0.0
    )
    out[16] = f"{total_return_pct:.2f}"  # Q = Total Return
    # R = Status. We use a simple OPEN/CLOSED convention: "OPEN" if
    # the position still has a row, "CLOSED" if it's been pruned (and
    # the operator has manually closed it). Refresh never sets CLOSED.
    if not out[18] or out[18].strip() == "":
        out[18] = "OPEN"
    return out


def _col_letter(n: int) -> str:
    """1-indexed column number -> spreadsheet column letter."""
    if n < 1:
        raise ValueError(f"Column number must be >= 1, got {n}")
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(ord("A") + rem) + result
    return result


# --- Public surface ---------------------------------------------------------


__all__ = [
    "bogota_weekday",
    "BOGOTA_TZ",
    "HolidayCalendar",
    "is_morning_trading_day",
    "MorningContext",
    "MorningResult",
    "MorningOutcome",
    "morning_run",
    "TickerRow",
    "RefreshOutcome",
    "refresh_returns",
]
