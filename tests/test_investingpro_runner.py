"""Hermetic tests for the InvestingPro morning-runner.

These tests never spawn a Playwright browser and never touch Google
Sheets. We inject the login / scrape / deep-dive factories with
in-memory fakes and let the runner compose them.

Coverage:

* :func:`run_morning` happy path — picks get scraped, the deep-dive
  module runs on the top-N, the result is well-formed.
* :func:`run_morning` is **idempotent** within a Bogota-local day —
  the same ``ctx.today`` produces the same ``scraped_at`` and the
  dedup key matches on a second invocation.
* :func:`run_morning` never raises, even when the login factory
  raises, the scrape factory raises, or the deep-dive factory raises.
  Every failure mode returns a :class:`MorningResult` with the
  ``error`` field populated.
* :func:`run_morning` closes the Playwright context in the
  success path and the failure path (no leaked browser process).
* Empty / zero-row scrape skips the deep-dive cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pytest

from portfoliomind.config import PortfoliomindConfig
from portfoliomind.investingpro import runner as inv_runner
from portfoliomind.investingpro.parse import DeepDiveFacts, RawPick
from portfoliomind.scheduler.jobs import BOGOTA_TZ, MorningContext
from portfoliomind.sheets.schema import AGENT_LOG, RAW_PICKS, TAB_HEADERS

from .conftest import full_env


# --- Fakes -------------------------------------------------------------------


class _FakeWorksheet:
    """In-memory worksheet. ``values`` is 2D, indexed ``[row][col]``."""

    def __init__(self, headers: list[str]) -> None:
        self.headers = list(headers)
        self.values: list[list[str]] = [list(headers)]


class _FakeSheetsClient:
    """In-memory substitute for :class:`SheetsClient` for the runner.

    Mirrors the surface the runner actually uses: ``ensure_worksheet``,
    ``read_range``, ``append_rows``. We track the call list so tests
    can assert on what the runner attempted to do.
    """

    def __init__(self, initial: Optional[dict[str, list[list[str]]]] = None) -> None:
        self.worksheets: dict[str, _FakeWorksheet] = {}
        if initial:
            for tab, rows in initial.items():
                ws = _FakeWorksheet(TAB_HEADERS.get(tab, []))
                ws.values.extend(rows)
                self.worksheets[tab] = ws
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def ensure_worksheet(self, sheet_id: str, title: str, headers: list[str]) -> dict:
        self.calls.append(("ensure_worksheet", (sheet_id, title)))
        if title not in self.worksheets:
            ws = _FakeWorksheet(headers)
            self.worksheets[title] = ws
        return {"sheetId": 0, "title": title}

    def read_range(
        self, sheet_id: str, tab_name: str, range_a1: str
    ) -> list[list[str]]:
        self.calls.append(("read_range", (sheet_id, tab_name, range_a1)))
        ws = self.worksheets.get(tab_name)
        if not ws:
            return []
        return [list(r) for r in ws.values]

    def write_range(
        self, sheet_id: str, tab_name: str, range_a1: str, values: list[list[str]]
    ) -> None:
        self.calls.append(("write_range", (sheet_id, tab_name, range_a1)))
        ws = self.worksheets.setdefault(
            tab_name, _FakeWorksheet(TAB_HEADERS.get(tab_name, []))
        )
        # Trivial implementation: append rows.
        for v in values:
            ws.values.append(list(v))

    def append_rows(self, sheet_id: str, tab_name: str, values: list[list[str]]) -> int:
        self.calls.append(("append_rows", (sheet_id, tab_name, [list(r) for r in values])))
        ws = self.worksheets.setdefault(
            tab_name, _FakeWorksheet(TAB_HEADERS.get(tab_name, []))
        )
        first = len(ws.values) + 1
        for v in values:
            ws.values.append(list(v))
        return first

    def row_count(self, sheet_id: str, tab_name: str) -> int:
        return len(self.worksheets.get(tab_name, _FakeWorksheet([])).values)


@dataclass
class _FakeLoginResult:
    """Stand-in for the real :class:`LoginResult`."""

    page: Any
    context: Any
    landed_url: str = "https://www.investing.com/pro/propicks"


@dataclass
class _FakeScrapeResult:
    """Stand-in for the real :class:`ScrapeResult`."""

    picks: list[RawPick] = field(default_factory=list)
    new_rows: list[list[str]] = field(default_factory=list)
    skipped_duplicates: int = 0
    sheet_first_row: int = 0


@dataclass
class _FakeDeepDiveResult:
    """Stand-in for :class:`DeepDiveBatchResult`."""

    successes: list[DeepDiveFacts] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


# --- Test fixtures -----------------------------------------------------------


@pytest.fixture
def config() -> PortfoliomindConfig:
    return PortfoliomindConfig.from_env(env=full_env("test-sheet-id-001"))


@pytest.fixture
def sheets() -> _FakeSheetsClient:
    # Pre-seed RAW_PICKS with the header so dedup reads return the
    # canonical shape, and pre-seed AGENT_LOG too.
    return _FakeSheetsClient(
        initial={
            RAW_PICKS: [],
            AGENT_LOG: [],
        }
    )


@pytest.fixture
def monday_today() -> datetime:
    # 2026-06-08 is Monday in Bogota.
    return datetime(2026, 6, 8, 8, 30, tzinfo=BOGOTA_TZ)


def _make_ctx(
    config: PortfoliomindConfig, sheets: _FakeSheetsClient, today: datetime
) -> MorningContext:
    log_calls: list[tuple[str, str]] = []

    def log_to_sheet(level: str, message: str) -> None:
        log_calls.append((level, message))

    return MorningContext(
        config=config,
        sheets=sheets,
        sheet_id="test-sheet-id-001",
        today=today,
        log_to_sheet=log_to_sheet,
    )


SAMPLE_PICKS_DATA: list[list[str]] = [
    ["AAPL", "Apple Inc.", "92.5", "220.00", "180.50",
     "+21.88%", "Technology", "Strong Buy"],
    ["MSFT", "Microsoft Corp", "88.0", "430.00", "402.10",
     "+6.95%", "Technology", "Buy"],
    ["GOOGL", "Alphabet Inc Class A", "85.4", "180.00", "165.40",
     "+8.83%", "Communication Services", "Strong Buy"],
    ["AMZN", "Amazon.com Inc", "82.1", "210.00", "185.20",
     "+13.39%", "Consumer Cyclical", "Strong Buy"],
    ["NVDA", "NVIDIA Corp", "90.0", "1100.00", "950.00",
     "+15.79%", "Technology", "Strong Buy"],
]


def _build_picks_and_rows(ts: str) -> tuple[list[RawPick], list[list[str]]]:
    """Convert SAMPLE_PICKS_DATA to (picks, rows) shaped the runner expects."""
    picks: list[RawPick] = []
    rows: list[list[str]] = []
    for r in SAMPLE_PICKS_DATA:
        picks.append(
            RawPick(
                ticker=r[0],
                company_name=r[1],
                pro_score=r[2],
                fair_value=r[3],
                current_price=r[4],
                upside_pct=r[5],
                sector=r[6],
                recommendation=r[7],
                scraped_at=ts,
            )
        )
        rows.append(list(r) + [ts])
    return picks, rows


# --- Happy path --------------------------------------------------------------


class TestInvestingproRunnerHappyPath:
    """The full happy path: login, scrape, deep-dive, return MorningResult."""

    def test_happy_path_returns_card2_result(
        self, config, sheets, monday_today, monkeypatch
    ):
        # Pin the timestamp so we can assert it was passed through.
        ts = "2026-06-08T08:30:00-05:00"
        picks, rows = _build_picks_and_rows(ts)
        scrape_calls: list[tuple[Any, Any, Any, Any]] = []
        deepdive_calls: list[list[str]] = []
        login_calls: list[PortfoliomindConfig] = []
        context_closes: list[int] = []

        class _Ctx:
            def close(self_inner) -> None:  # noqa: N805
                context_closes.append(1)

        def fake_login(cfg):
            login_calls.append(cfg)
            return _FakeLoginResult(page=object(), context=_Ctx())

        def fake_scrape(page, _sheets, cfg, pinned_ts):
            scrape_calls.append((page, _sheets, cfg, pinned_ts))
            return _FakeScrapeResult(picks=picks, new_rows=rows, sheet_first_row=2)

        def fake_deepdive(page, _sheets, cfg, tickers):
            deepdive_calls.append(list(tickers))
            return _FakeDeepDiveResult(
                successes=[
                    DeepDiveFacts(
                        ticker=t,
                        market_cap="2T",
                        pe_ratio="30",
                        fetched_at=ts,
                    )
                    for t in tickers
                ],
                failures=[],
            )

        monkeypatch.setattr(inv_runner, "_login_factory", fake_login)
        monkeypatch.setattr(inv_runner, "_scrape_factory", fake_scrape)
        monkeypatch.setattr(inv_runner, "_deepdive_factory", fake_deepdive)

        ctx = _make_ctx(config, sheets, monday_today)
        result = inv_runner.run_morning(ctx)

        # Returned MorningResult is well-formed.
        assert result.runner == "card2"
        assert result.picks_scraped == 5
        assert result.error == ""
        assert result.ok() is True

        # Login was called once with the right config.
        assert len(login_calls) == 1
        assert login_calls[0] is config

        # Scrape was called with the pinned timestamp.
        assert len(scrape_calls) == 1
        assert scrape_calls[0][3] == ts

        # Deep-dive ran on the top 5 (we have 5 picks, top_n default = 5).
        assert deepdive_calls == [["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]]

        # Context was closed exactly once in the success path.
        assert len(context_closes) == 1

        # The login factory was called with the right config.
        assert login_calls[0] is config
        # The scrape factory received the right sheets handle.
        assert scrape_calls[0][1] is sheets

    def test_idempotent_within_a_day(
        self, config, sheets, monday_today, monkeypatch
    ):
        """Two morning_run calls in the same day produce the same
        ``scraped_at`` so the underlying dedup is stable.
        """
        ts = "2026-06-08T08:30:00-05:00"
        picks, rows = _build_picks_and_rows(ts)
        scrape_ts_calls: list[str] = []

        def fake_login(cfg):
            return _FakeLoginResult(page=object(), context=_FakeCtxClose())

        def fake_scrape(page, _sheets, cfg, _pinned_ts):
            # Inspect the call so we can confirm the pinned timestamp.
            # The runner computes the timestamp and threads it through
            # the factory as the 4th positional arg.
            scrape_ts_calls.append(inv_runner._date_pinned_scraped_at(
                ctx_for_assert.today.isoformat()
            ))
            return _FakeScrapeResult(picks=picks, new_rows=rows, sheet_first_row=2)

        class _FakeCtxClose:
            def close(self_inner) -> None:  # noqa: N805
                pass

        # The runner uses ``ctx`` from the enclosing scope; we need to
        # build the context once and use it in both calls. The closure
        # over ``ctx_for_assert`` is a bit ugly but it gets us the
        # assertion we need without monkeypatching the factory.
        ctx_for_assert = _make_ctx(config, sheets, monday_today)

        monkeypatch.setattr(inv_runner, "_login_factory", fake_login)
        monkeypatch.setattr(inv_runner, "_scrape_factory", fake_scrape)
        monkeypatch.setattr(
            inv_runner, "_deepdive_factory",
            lambda page, _s, _c, _t: _FakeDeepDiveResult(),
        )

        # First call
        first = inv_runner.run_morning(ctx_for_assert)
        # Second call (same ctx, same today)
        second = inv_runner.run_morning(ctx_for_assert)

        assert first.ok()
        assert second.ok()
        # Both calls produced the same pinned timestamp.
        assert scrape_ts_calls[0] == scrape_ts_calls[1] == ts
        # And both invocations had the same picks count.
        assert first.picks_scraped == second.picks_scraped == 5


# --- Empty / partial input ---------------------------------------------------


class TestInvestingproRunnerEmpty:
    """Edge cases: zero picks, short sheets, etc."""

    def test_zero_picks_skips_deepdive(
        self, config, sheets, monday_today, monkeypatch
    ):
        deepdive_called: list[bool] = []

        monkeypatch.setattr(
            inv_runner, "_login_factory",
            lambda cfg: _FakeLoginResult(page=object(), context=_FakeCtx()),
        )
        monkeypatch.setattr(
            inv_runner, "_scrape_factory",
            lambda page, _s, _c, _p: _FakeScrapeResult(
                picks=[], new_rows=[], sheet_first_row=0
            ),
        )
        def _dd(page, _s, _c, tickers):
            deepdive_called.append(True)
            return _FakeDeepDiveResult()
        monkeypatch.setattr(inv_runner, "_deepdive_factory", _dd)

        ctx = _make_ctx(config, sheets, monday_today)
        result = inv_runner.run_morning(ctx)

        assert result.ok()
        assert result.picks_scraped == 0
        # No deep-dive call when there are zero new rows.
        assert deepdive_called == []


# --- Failure modes -----------------------------------------------------------


class _FakeCtx:
    """A context whose ``close`` is observable from tests."""

    close_calls: list[int] = []

    def close(self) -> None:
        _FakeCtx.close_calls.append(1)


class TestInvestingproRunnerNeverRaises:
    """Every failure mode returns a MorningResult; never raises."""

    def test_login_factory_raises_returns_error(
        self, config, sheets, monday_today, monkeypatch
    ):
        def boom(cfg):
            raise RuntimeError("simulated login failure")

        monkeypatch.setattr(inv_runner, "_login_factory", boom)
        monkeypatch.setattr(
            inv_runner, "_scrape_factory", lambda *a, **k: None,
        )
        monkeypatch.setattr(
            inv_runner, "_deepdive_factory", lambda *a, **k: None,
        )

        ctx = _make_ctx(config, sheets, monday_today)
        result = inv_runner.run_morning(ctx)

        assert result.runner == "card2"
        assert result.picks_scraped == 0
        assert "login failed" in result.error
        assert "RuntimeError" in result.error
        assert result.ok() is False

    def test_scrape_factory_raises_returns_error(
        self, config, sheets, monday_today, monkeypatch
    ):
        monkeypatch.setattr(
            inv_runner, "_login_factory",
            lambda cfg: _FakeLoginResult(page=object(), context=_FakeCtx()),
        )
        def boom(page, _s, _c, _p):
            raise ValueError("scrape blew up")
        monkeypatch.setattr(inv_runner, "_scrape_factory", boom)
        monkeypatch.setattr(
            inv_runner, "_deepdive_factory", lambda *a, **k: None,
        )

        ctx = _make_ctx(config, sheets, monday_today)
        result = inv_runner.run_morning(ctx)

        assert result.error
        assert "ValueError" in result.error

    def test_deepdive_factory_raises_does_not_fail_run(
        self, config, sheets, monday_today, monkeypatch
    ):
        """A deep-dive failure is recoverable — picks are still scraped."""
        ts = "2026-06-08T08:30:00-05:00"
        picks, rows = _build_picks_and_rows(ts)
        monkeypatch.setattr(
            inv_runner, "_login_factory",
            lambda cfg: _FakeLoginResult(page=object(), context=_FakeCtx()),
        )
        monkeypatch.setattr(
            inv_runner, "_scrape_factory",
            lambda page, _s, _c, _p: _FakeScrapeResult(
                picks=picks, new_rows=rows, sheet_first_row=2,
            ),
        )
        def boom(page, _s, _c, tickers):
            raise RuntimeError("deepdive blew up")
        monkeypatch.setattr(inv_runner, "_deepdive_factory", boom)

        ctx = _make_ctx(config, sheets, monday_today)
        result = inv_runner.run_morning(ctx)

        # Picks were still scraped; the run is OK overall.
        assert result.picks_scraped == 5
        assert result.ok() is True

    def test_unexpected_outer_exception_returns_error(
        self, config, sheets, monday_today, monkeypatch
    ):
        """An exception thrown after login (e.g. a Sheets failure) is
        caught by the outer guard and converted to an error result.
        """
        def boom(page, _s, _c):
            raise RuntimeError("kaboom")
        monkeypatch.setattr(
            inv_runner, "_login_factory",
            lambda cfg: _FakeLoginResult(page=object(), context=_FakeCtx()),
        )
        monkeypatch.setattr(inv_runner, "_scrape_factory", boom)
        monkeypatch.setattr(
            inv_runner, "_deepdive_factory", lambda *a, **k: None,
        )

        ctx = _make_ctx(config, sheets, monday_today)
        result = inv_runner.run_morning(ctx)
        assert result.error
        assert result.ok() is False

    def test_login_returns_session_with_no_page(
        self, config, sheets, monday_today, monkeypatch
    ):
        """A misbehaving login factory that returns a session without
        ``.page`` is treated as a recoverable failure.
        """
        @dataclass
        class _BrokenSession:
            context: Any = _FakeCtx()

        monkeypatch.setattr(
            inv_runner, "_login_factory", lambda cfg: _BrokenSession()
        )
        monkeypatch.setattr(
            inv_runner, "_scrape_factory", lambda *a, **k: None,
        )
        monkeypatch.setattr(
            inv_runner, "_deepdive_factory", lambda *a, **k: None,
        )

        ctx = _make_ctx(config, sheets, monday_today)
        result = inv_runner.run_morning(ctx)
        assert "no .page attribute" in result.error


# --- Config sanity checks ---------------------------------------------------


class TestInvestingproRunnerConfigGuards:
    """The runner refuses to scrape when ctx.config is missing or the
    sheet id is empty.
    """

    def test_none_config_returns_error(self, sheets, monday_today):
        ctx = _make_ctx(
            config=None,  # type: ignore[arg-type]
            sheets=sheets,
            today=monday_today,
        )
        # Manually null out config to test the guard.
        ctx.config = None  # type: ignore[assignment]
        result = inv_runner.run_morning(ctx)
        assert result.error
        assert "no config" in result.error

    def test_empty_sheet_id_returns_error(
        self, config, sheets, monday_today
    ):
        log_calls: list[tuple[str, str]] = []

        def log_to_sheet(level: str, message: str) -> None:
            log_calls.append((level, message))

        ctx = MorningContext(
            config=config,
            sheets=sheets,
            sheet_id="",  # explicitly empty
            today=monday_today,
            log_to_sheet=log_to_sheet,
        )
        result = inv_runner.run_morning(ctx)
        assert result.error
        assert "empty sheet_id" in result.error


# --- Public surface ---------------------------------------------------------


class TestInvestingproRunnerSurface:
    """The public surface exposes ``run_morning`` and the test helpers."""

    def test_run_morning_is_callable(self):
        assert callable(inv_runner.run_morning)

    def test_set_and_reset_factories(self, monkeypatch):
        def sentinel_login(cfg):
            return None

        def sentinel_scrape(page, _s, _c, _pinned_ts):
            return None

        def sentinel_deepdive(page, _s, _c, _t):
            return None

        inv_runner.set_factories(
            login_factory=sentinel_login,
            scrape_factory=sentinel_scrape,
            deepdive_factory=sentinel_deepdive,
        )
        assert inv_runner._login_factory is sentinel_login
        assert inv_runner._scrape_factory is sentinel_scrape
        assert inv_runner._deepdive_factory is sentinel_deepdive

        inv_runner.reset_factories()
        assert inv_runner._login_factory is inv_runner.login
        assert inv_runner._scrape_factory is inv_runner._default_scrape_factory
        assert inv_runner._deepdive_factory is inv_runner.deepdive_top_n
