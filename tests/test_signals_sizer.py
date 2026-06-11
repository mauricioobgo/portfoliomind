"""Hermetic tests for :mod:`portfoliomind.signals.sizer`."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from portfoliomind.sheets.schema import APPROVED_TRADES, TAB_HEADERS
from portfoliomind.signals.sizer import (
    DEFAULT_EQUITY,
    EQUITY_ENV_VAR,
    MAX_POSITION_FRACTION,
    PositionSizer,
    SizingError,
    TradeOrder,
)


@dataclass
class StubCandidate:
    ticker: str = "AAPL"
    last_close: float = 100.0
    p_bullish: float = 0.70
    vol_20d: float = 0.01
    patterns: list[str] = field(default_factory=lambda: ["golden_cross"])


# --- Happy path -----------------------------------------------------------------


def test_sizes_a_strong_candidate():
    sizer = PositionSizer(equity=10_000.0)
    order = sizer.size(StubCandidate())
    assert isinstance(order, TradeOrder)
    assert order.side == "BUY"
    assert order.ticker == "AAPL"
    # p=0.7, R:R=2 → kelly=0.55; quarter-kelly 0.1375 capped at 10% → $1000 → 10 shares
    assert order.qty == 10.0
    assert order.allocation == pytest.approx(1000.0)
    # stop_pct = clamp(3×0.01, 2%, 8%) = 3%
    assert order.sl == pytest.approx(97.0)
    assert order.tp == pytest.approx(106.0)
    assert order.sl < order.entry_price < order.tp
    assert "p_bullish=0.700" in order.note


def test_allocation_respects_hard_cap():
    sizer = PositionSizer(equity=100_000.0)
    order = sizer.size(StubCandidate(p_bullish=0.95))
    assert order.allocation <= 100_000.0 * MAX_POSITION_FRACTION + 1e-9


def test_stop_pct_clamped_for_high_vol():
    sizer = PositionSizer(equity=10_000.0)
    order = sizer.size(StubCandidate(vol_20d=0.10))  # 3σ = 30% → clamped to 8%
    assert order.sl == pytest.approx(92.0)


def test_missing_vol_uses_midpoint_stop():
    sizer = PositionSizer(equity=10_000.0)
    order = sizer.size(StubCandidate(vol_20d=0.0))  # midpoint of [2%, 8%] = 5%
    assert order.sl == pytest.approx(95.0)


def test_approved_row_matches_sheet_headers():
    order = PositionSizer(equity=10_000.0).size(StubCandidate())
    row = order.to_approved_row()
    assert len(row) == len(TAB_HEADERS[APPROVED_TRADES])
    assert row[1] == "AAPL"
    assert row[2] == "Stock"


def test_etf_instrument_type():
    order = PositionSizer(equity=50_000.0).size(StubCandidate(ticker="SPY", last_close=400.0))
    assert order.instrument_type == "ETF"


# --- Rejections -----------------------------------------------------------------------


def test_no_edge_raises_sizing_error():
    # p=0.30 at R:R 2 → kelly = 0.30 - 0.35 < 0
    with pytest.raises(SizingError, match="no positive edge"):
        PositionSizer(equity=10_000.0).size(StubCandidate(p_bullish=0.30))


def test_too_expensive_raises_sizing_error():
    with pytest.raises(SizingError, match="exceeds"):
        PositionSizer(equity=10_000.0).size(StubCandidate(last_close=5000.0))


def test_missing_price_raises_sizing_error():
    with pytest.raises(SizingError, match="last_close"):
        PositionSizer(equity=10_000.0).size(StubCandidate(last_close=0.0))


def test_missing_ticker_raises_sizing_error():
    with pytest.raises(SizingError, match="ticker"):
        PositionSizer(equity=10_000.0).size(StubCandidate(ticker=""))


def test_bad_equity_rejected():
    with pytest.raises(SizingError):
        PositionSizer(equity=-5.0)


# --- Equity resolution -------------------------------------------------------------------


def test_equity_from_env(monkeypatch):
    monkeypatch.setenv(EQUITY_ENV_VAR, "50000")
    assert PositionSizer().equity == 50_000.0


def test_equity_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(EQUITY_ENV_VAR, raising=False)
    assert PositionSizer().equity == DEFAULT_EQUITY


def test_equity_default_when_env_garbage(monkeypatch):
    monkeypatch.setenv(EQUITY_ENV_VAR, "lots of money")
    assert PositionSizer().equity == DEFAULT_EQUITY


# --- Kelly math ----------------------------------------------------------------------------


def test_kelly_formula():
    sizer = PositionSizer(equity=10_000.0, reward_risk=2.0)
    assert sizer.kelly(0.5) == pytest.approx(0.25)
    assert sizer.kelly(1.0) == pytest.approx(1.0)
    # break-even probability at R:R 2 is 1/3
    assert sizer.kelly(1.0 / 3.0) == pytest.approx(0.0, abs=1e-9)
