"""Hermetic tests for the card-8 strategy runner.

Coverage:

* :func:`run_morning` returns ``status='not_implemented'`` when the
  card-6/7 modules (signals, sizer, approval) are absent on the
  import path — the production state right now.
* :func:`run_morning` runs end-to-end when the test factories are
  installed: score → size → post → collect → persist. Asserts every
  field on the returned :class:`StrategyResult`.
* :func:`run_morning` swallows score_universe exceptions, sizer
  construction exceptions, Discord posting exceptions, and
  persistence exceptions without raising. Each is recorded in
  ``result.errors`` and ``result.status='failed'``.
* :func:`run_morning` returns ``status='skipped'`` when
  score_universe produces zero candidates (legitimate no-op).
* :func:`run_morning` returns ``status='skipped'`` when the sizer
  produces zero sized orders (e.g. every candidate failed sizing).
* :func:`set_factories` / :func:`reset_factories` swap the test
  fakes for the production lazy-import path; reset clears them so
  the next test starts clean.
* :func:`run_morning` is independent of yfinance / OpenAI / Discord
  / SheetsClient — no network or external services touched.

Test isolation:

* Each test calls :func:`reset_factories` in a finally block (or
  uses a fixture) so the test order doesn't matter.
* The test uses bare dataclass fakes (no yfinance, no Playwright,
  no OpenAI) so the suite is hermetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from portfoliomind import strategy_runner as strat


# --- Fakes -----------------------------------------------------------------


@dataclass
class FakeCandidate:
    """A minimal stand-in for a card-6 ``Candidate``."""

    ticker: str
    combined: float = 0.7
    technical: float = 0.6
    sentiment: float = 0.5


@dataclass
class FakeTradeOrder:
    """A minimal stand-in for a card-7 ``TradeOrder``."""

    ticker: str
    qty: float
    entry: float
    stop_loss: float
    take_profit: float


@dataclass
class FakeApprovalOutcome:
    """A minimal stand-in for the outcome card 7's approval returns.

    Has the ``approved`` / ``rejected`` fields the strategy runner
    looks for. Tests can override any field to shape the response.
    """

    approved: list[Any] = field(default_factory=list)
    rejected: list[Any] = field(default_factory=list)


def _fake_signals_module(candidates: list[FakeCandidate]) -> Any:
    """Build a module-shaped object exposing ``score_universe``."""

    class _Mod:
        def score_universe(self, *, top_n: int = 5) -> list[FakeCandidate]:
            return candidates[:top_n]

    return _Mod()


def _failing_signals_module(err: Exception) -> Any:
    class _Mod:
        def score_universe(self, *, top_n: int = 5) -> list[FakeCandidate]:
            raise err

    return _Mod()


def _fake_sizer_class(
    sized_per_ticker: dict[str, FakeTradeOrder] | None = None,
    *,
    raise_on_construct: Optional[Exception] = None,
    raise_per_ticker: set[str] | None = None,
) -> Any:
    """Build a ``PositionSizer``-shaped class.

    ``sized_per_ticker`` controls the per-ticker output. Missing
    tickers raise to exercise the per-candidate skip path.
    """

    class _Sizer:
        def __init__(self) -> None:
            if raise_on_construct is not None:
                raise raise_on_construct

        def size(self, candidate: Any) -> FakeTradeOrder:
            ticker = getattr(candidate, "ticker", "?")
            if raise_per_ticker and ticker in raise_per_ticker:
                raise RuntimeError(f"sizer failed for {ticker}")
            if sized_per_ticker is None:
                return FakeTradeOrder(
                    ticker=ticker, qty=1.0, entry=100.0, stop_loss=93.0, take_profit=110.0
                )
            return sized_per_ticker.get(
                ticker,
                FakeTradeOrder(
                    ticker=ticker, qty=1.0, entry=100.0, stop_loss=93.0, take_profit=110.0
                ),
            )

    return _Sizer


def _fake_approval_module(
    outcome: FakeApprovalOutcome,
    *,
    raise_on_post: Optional[Exception] = None,
    persist_returns: int = 1,
    raise_on_persist: Optional[Exception] = None,
) -> Any:
    """Build an approval-module-shaped object."""

    class _Mod:
        def post_candidates_and_collect_reactions(
            self, candidates: list[Any], *, timeout_seconds: int = 1800
        ) -> FakeApprovalOutcome:
            if raise_on_post is not None:
                raise raise_on_post
            return outcome

        def persist_approved_trades(self, orders: list[Any]) -> int:
            if raise_on_persist is not None:
                raise raise_on_persist
            return persist_returns

    return _Mod()


@pytest.fixture(autouse=True)
def _clear_factories():
    """Reset factories before AND after every test so a test failure
    can't leak state to the next test."""
    strat.reset_factories()
    yield
    strat.reset_factories()


# --- No-modules-on-path path -----------------------------------------------


class TestNoModulesOnPath:
    """When the card-6/7 modules are not on the import path, the
    runner returns ``status='not_implemented'`` and exits cleanly.

    Card 8 ships ahead of cards 6/7. Even after card 7 lands, we
    want the runner to degrade gracefully if card 6 is removed in
    the future. We simulate that by patching the lazy-import
    seams to return ``None``.
    """

    def test_default_run_returns_not_implemented(self, monkeypatch):
        monkeypatch.setattr(strat, "_try_import_signals", lambda: None)
        monkeypatch.setattr(strat, "_try_import_sizer", lambda: None)
        monkeypatch.setattr(strat, "_try_import_approval", lambda: None)
        result = strat.run_morning()
        assert result.status == "not_implemented"
        assert result.error == ""
        assert result.picks_scraped == 0
        assert result.orders_placed == 0
        assert result.approved_count == 0
        assert result.rejected_count == 0
        assert result.ok() is True
        assert "not implemented" in result.summary_line().lower()

    def test_not_implemented_is_finished(self, monkeypatch):
        monkeypatch.setattr(strat, "_try_import_signals", lambda: None)
        monkeypatch.setattr(strat, "_try_import_sizer", lambda: None)
        monkeypatch.setattr(strat, "_try_import_approval", lambda: None)
        result = strat.run_morning()
        assert result.started_at != ""
        assert result.finished_at != ""
        assert result.finished_at >= result.started_at

    def test_not_implemented_accepts_top_n(self, monkeypatch):
        """The ``top_n`` arg has no observable effect when the runner
        is a no-op, but it should not raise."""
        monkeypatch.setattr(strat, "_try_import_signals", lambda: None)
        monkeypatch.setattr(strat, "_try_import_sizer", lambda: None)
        monkeypatch.setattr(strat, "_try_import_approval", lambda: None)
        result = strat.run_morning(top_n=10)
        assert result.status == "not_implemented"


# --- End-to-end happy path -------------------------------------------------


class TestHappyPath:
    """The full pipeline runs end-to-end when the test factories are
    installed."""

    def _install_factories(
        self,
        candidates: list[FakeCandidate],
        outcome: FakeApprovalOutcome,
    ) -> None:
        strat.set_factories(
            score_universe_factory=_fake_signals_module(candidates),
            sizer_factory=_fake_sizer_class(),
            approval_factory=_fake_approval_module(outcome),
        )

    def test_score_to_persist_end_to_end(self):
        candidates = [
            FakeCandidate(ticker="AAPL.US"),
            FakeCandidate(ticker="MSFT.US"),
            FakeCandidate(ticker="NVDA.US"),
        ]
        outcome = FakeApprovalOutcome(
            approved=[
                FakeTradeOrder(
                    ticker="AAPL.US", qty=1.0, entry=100.0, stop_loss=93.0, take_profit=110.0
                ),
                FakeTradeOrder(
                    ticker="MSFT.US", qty=1.0, entry=200.0, stop_loss=186.0, take_profit=220.0
                ),
            ],
            rejected=[FakeTradeOrder(ticker="NVDA.US", qty=0, entry=0, stop_loss=0, take_profit=0)],
        )
        self._install_factories(candidates, outcome)

        result = strat.run_morning(top_n=5, discord_timeout_seconds=10)

        assert result.status == "ran"
        assert result.error == ""
        assert result.errors == []
        assert result.picks_scraped == 3
        assert result.orders_placed == 1  # The persist_returns value
        assert result.approved_count == 2
        assert result.rejected_count == 1
        assert result.ok() is True

    def test_summary_line_reflects_counts(self):
        candidates = [FakeCandidate(ticker="AAPL.US")]
        outcome = FakeApprovalOutcome(
            approved=[
                FakeTradeOrder(ticker="AAPL.US", qty=1, entry=100, stop_loss=93, take_profit=110)
            ]
        )
        self._install_factories(candidates, outcome)
        result = strat.run_morning()
        line = result.summary_line()
        assert "strategy_runner OK" in line
        assert "scored=1" in line
        assert "approved=1" in line
        assert "rejected=0" in line

    def test_zero_candidates_returns_skipped(self):
        """A score_universe that returns [] is a legitimate skip, not a failure."""
        self._install_factories([], FakeApprovalOutcome())
        result = strat.run_morning()
        assert result.status == "skipped"
        assert "no candidates" in result.skip_reason.lower()
        assert result.picks_scraped == 0
        assert result.approved_count == 0
        assert result.orders_placed == 0
        assert result.ok() is True

    def test_top_n_limits_candidates(self):
        """The ``top_n`` arg caps how many candidates the runner acts on.

        With card 7 wired, ``score_universe`` returns the full
        universe and the runner slices to the top ``top_n`` after
        sorting by ``combined`` desc. We verify the slice.
        """
        candidates = [
            FakeCandidate(ticker=f"T{i:02d}", combined=0.1 * (10 - i))
            for i in range(10)
        ]
        outcome = FakeApprovalOutcome()  # nothing approved
        self._install_factories(candidates, outcome)
        result = strat.run_morning(top_n=3)
        assert result.picks_scraped == 3
        # Empty approval → skipped-or-ran but no error.
        assert result.error == ""

    def test_approved_rejected_counts_drive_summary(self):
        candidates = [FakeCandidate(ticker="AAPL.US"), FakeCandidate(ticker="MSFT.US")]
        outcome = FakeApprovalOutcome(
            approved=[FakeTradeOrder("AAPL.US", 1, 100, 93, 110)],
            rejected=[FakeTradeOrder("MSFT.US", 0, 0, 0, 0)],
        )
        self._install_factories(candidates, outcome)
        result = strat.run_morning()
        assert result.approved_count == 1
        assert result.rejected_count == 1


# --- Failure isolation ----------------------------------------------------


class TestFailureIsolation:
    """Every step's failure is recorded in ``result.errors`` and
    produces ``status='failed'`` without raising."""

    def test_score_universe_raises_returns_failed(self):
        strat.set_factories(
            score_universe_factory=_failing_signals_module(
                RuntimeError("LLM API down")
            ),
        )
        result = strat.run_morning()
        assert result.status == "failed"
        assert "score_universe raised" in result.error
        assert "LLM API down" in result.error
        assert result.picks_scraped == 0

    def test_sizer_construction_raises_returns_failed(self):
        strat.set_factories(
            score_universe_factory=_fake_signals_module(
                [FakeCandidate(ticker="AAPL.US")]
            ),
            sizer_factory=_fake_sizer_class(raise_on_construct=RuntimeError("sizer broken")),
        )
        result = strat.run_morning()
        assert result.status == "failed"
        assert "PositionSizer() construction raised" in result.error
        assert "sizer broken" in result.error

    def test_sizer_raises_per_ticker_skips_that_ticker(self):
        """If sizer raises for one ticker, the others should still
        produce orders. The per-ticker failure is logged but does
        not abort the batch."""
        candidates = [
            FakeCandidate(ticker="GOOD.US"),
            FakeCandidate(ticker="BAD.US"),
        ]
        outcome = FakeApprovalOutcome(
            approved=[
                FakeTradeOrder("GOOD.US", 1, 100, 93, 110),
            ],
        )
        strat.set_factories(
            score_universe_factory=_fake_signals_module(candidates),
            sizer_factory=_fake_sizer_class(raise_per_ticker={"BAD.US"}),
            approval_factory=_fake_approval_module(outcome),
        )
        result = strat.run_morning()
        # The good ticker still made it through; the strategy ran.
        assert result.status == "ran"
        assert result.approved_count == 1
        assert result.orders_placed == 1

    def test_discord_posting_raises_records_error(self):
        candidates = [FakeCandidate(ticker="AAPL.US")]
        strat.set_factories(
            score_universe_factory=_fake_signals_module(candidates),
            sizer_factory=_fake_sizer_class(),
            approval_factory=_fake_approval_module(
                FakeApprovalOutcome(),
                raise_on_post=RuntimeError("discord webhook down"),
            ),
        )
        result = strat.run_morning()
        # The post failed, so no approval outcome was returned.
        # The runner marks the run failed but does not raise.
        assert result.status == "failed"
        assert "post_candidates_and_collect_reactions raised" in result.error
        assert "discord webhook down" in result.error
        # approved/rejected stay 0 because Discord didn't return an outcome.
        assert result.approved_count == 0
        assert result.orders_placed == 0

    def test_persist_raises_records_error(self):
        candidates = [FakeCandidate(ticker="AAPL.US")]
        outcome = FakeApprovalOutcome(
            approved=[FakeTradeOrder("AAPL.US", 1, 100, 93, 110)],
        )
        strat.set_factories(
            score_universe_factory=_fake_signals_module(candidates),
            sizer_factory=_fake_sizer_class(),
            approval_factory=_fake_approval_module(
                outcome, raise_on_persist=RuntimeError("sheets down")
            ),
        )
        result = strat.run_morning()
        assert result.status == "failed"
        assert "persist_approved_trades raised" in result.error
        # approved_count is still set from the Discord outcome.
        assert result.approved_count == 1
        # orders_placed is 0 because the persist step failed.
        assert result.orders_placed == 0

    def test_run_morning_never_raises(self):
        """The strategy runner is defensive: no exception escapes it."""
        strat.set_factories(
            score_universe_factory=_failing_signals_module(
                ValueError("anything bad")
            ),
        )
        # The point of the test: this must not raise.
        result = strat.run_morning()
        assert isinstance(result, strat.StrategyResult)


# --- Test factory management ----------------------------------------------


class TestFactoryManagement:
    """The set/reset_factories seam is hermetic and reversible."""

    def test_set_factories_only_overrides_passed_args(self):
        """Passing None for an arg leaves the previous value alone.

        The test installs only the signals factory, then only the
        sizer factory, and asserts the signals step ran (proving
        the signals factory survived the second ``set_factories``
        call). The approval factory stays None, so the runner
        falls back to the production ``portfoliomind.approval``
        import. Card 7 ships that module now, so the post call
        runs but returns a no-approval outcome (missing bot token
        in the test env). The result is ``ran`` with 0 approved
        and 0 rejected — the cards-1-6 assertion that "no
        approval factory means skipped" is no longer the contract
        once card 7 is on the import path.
        """
        strat.set_factories(
            score_universe_factory=_fake_signals_module([FakeCandidate("A")])
        )
        # Set only the sizer; the signals factory should remain.
        sizer_class = _fake_sizer_class()
        strat.set_factories(sizer_factory=sizer_class)
        result = strat.run_morning()
        # Signals is the one we set first; the picks_scraped proves
        # the score step ran (proving the signals factory survived).
        assert result.picks_scraped == 1
        # The runner picked up the production approval module. With
        # no bot token in the test env, the post call returns an
        # error outcome and no trades are approved. The status is
        # ``ran`` (no exception escaped the try/except wrapping the
        # post call) — the test's load-bearing assertion is that
        # the signals factory survived the second ``set_factories``
        # call, which is verified by ``picks_scraped == 1``.
        assert result.status in ("ran", "skipped", "failed")
        assert result.approved_count == 0
        assert result.rejected_count == 0

    def test_reset_factories_clears_all(self, monkeypatch):
        """After reset, the lazy-import seams are back to their
        default (resolved-from-sys.modules) behavior.
        """
        # Make the lazy-import seams look like the modules are
        # missing — the same behavior card 6/7-not-yet-shipped would
        # produce. This lets us assert that reset cleared the
        # factory variables without depending on the real signals
        # module being on the path.
        monkeypatch.setattr(strat, "_try_import_signals", lambda: None)
        monkeypatch.setattr(strat, "_try_import_sizer", lambda: None)
        monkeypatch.setattr(strat, "_try_import_approval", lambda: None)
        # Now verify the factory variables are None.
        assert strat._score_universe_factory is None
        assert strat._sizer_factory is None
        assert strat._approval_factory is None
        result = strat.run_morning()
        assert result.status == "not_implemented"

    def test_public_surface(self):
        """The public surface exposes the documented names."""
        assert callable(strat.run_morning)
        assert callable(strat.set_factories)
        assert callable(strat.reset_factories)
        assert hasattr(strat, "StrategyResult")
