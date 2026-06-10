"""Exhaustive unit tests for :mod:`portfoliomind.signals.sizer`.

The sizer is the highest-stakes code in the project — real money on
the line. The card 7 spec lists these cases; we cover them and a
handful of edge cases (zero qty, negative input, malformed signal).

All tests are hermetic: no yfinance, no Discord, no Sheets. The
``entry_price_fetcher`` is injected as a constant so each test
controls the price directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from portfoliomind.signals.commissions import InstrumentType
from portfoliomind.signals.sizer import PositionSizer, RejectReason, TradeOrder


# --- Fakes --------------------------------------------------------------


@dataclass(frozen=True)
class FakeSignal:
    """A minimal card-6 Signal stand-in for sizer tests.

    The sizer reads ``ticker``, ``combined``, ``error``,
    ``asof_date``, and the reason/confidence fields. Everything else
    is unused.
    """

    ticker: str
    combined: float = 0.65
    technical: float = 0.6
    sentiment: float = 0.5
    confidence: float = 0.7
    reasons: list[str] = field(default_factory=lambda: ["test signal"])
    error: str = ""
    asof_date: str = "2026-06-10"


def _sizer(
    *,
    entry_price: Optional[float] = 100.0,
    per_trade_cap: float = 200.0,
    max_open_positions: int = 5,
    sl_pct: float = 0.07,
    tp_pct: float = 0.14,
    max_commission_pct: float = 0.05,
    allow_fractional: bool = True,
) -> PositionSizer:
    """Build a sizer with a fixed entry price. ``None`` simulates no-data."""
    fetcher = (lambda _t: entry_price) if entry_price is not None else (lambda _t: None)
    return PositionSizer(
        per_trade_cap=per_trade_cap,
        max_open_positions=max_open_positions,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        max_commission_pct=max_commission_pct,
        allow_fractional=allow_fractional,
        entry_price_fetcher=fetcher,
        verbose=False,
    )


# --- Happy path: ETF (no $8 minimum) ----------------------------------


class TestHappyPathETF:
    """The spec's happy path uses $100 entry; with stocks that triggers
    the $8 minimum + 5% commission rejection, so the spec test is
    really about an ETF. The card 7 spec is internally inconsistent
    here (the $200 cap + $8 minimum + 5% rule + 2-share happy path
    all collide for a stock); we run the same numbers against an ETF
    to validate the math."""

    def test_happy_path_etf_qty_sl_tp(self):
        # SPY is an ETF in our universe. $100 entry, $200 cap → 2 shares.
        # ETF commission is $0 (under free tier). 5% threshold = $10.
        # Commission ($0) ≤ threshold ($10) → PASS.
        sizer = _sizer()
        sig = FakeSignal(ticker="SPY")
        out = sizer.size(sig, open_position_count=0)
        assert isinstance(out, TradeOrder)
        assert out.ticker == "SPY"
        assert out.qty == 2.0
        assert out.entry == 100.0
        assert out.sl == pytest.approx(93.0)  # 100 * (1 - 0.07)
        assert out.tp == pytest.approx(114.0)  # 100 * (1 + 0.14)
        assert out.notional == pytest.approx(200.0)
        assert out.commission_rt == pytest.approx(0.0)
        assert out.r_r_ratio == pytest.approx(2.0)
        assert out.instrument is InstrumentType.US_ETF


# --- $200 per-trade cap enforcement -----------------------------------


class TestPerTradeCap:
    def test_cap_below_one_share_rejects(self):
        # NVDA at $500 with $200 cap → qty = floor(200/500) = 0.
        sizer = _sizer(entry_price=500.0, per_trade_cap=200.0)
        out = sizer.size(FakeSignal(ticker="NVDA"))
        assert isinstance(out, RejectReason)
        assert "per-trade cap" in out.reason
        assert out.ticker == "NVDA"

    def test_cap_exactly_one_share(self):
        # Stock at $200 with $200 cap → qty = 1.
        sizer = _sizer(entry_price=200.0, per_trade_cap=200.0)
        out = sizer.size(FakeSignal(ticker="AAPL"))
        # 1 share × $200 = $200. Commission = max($8, 0.08%×$200) = $8.
        # Round-trip = $16. 5% threshold = $10. → REJECT.
        # The $200 cap is the *ceiling*; a $200 stock with $200 cap
        # still hits the commission rejection. We expect a reject.
        assert isinstance(out, RejectReason)
        assert "commission" in out.reason.lower()

    def test_cap_respected_for_cheap_stock(self):
        # Stock at $20 with $200 cap → qty = 10. Notional = $200.
        # Commission = max($8, 0.08%×$200) = $8. RT = $16. 5% × $200 = $10. REJECT.
        sizer = _sizer(entry_price=20.0, per_trade_cap=200.0)
        out = sizer.size(FakeSignal(ticker="AAPL"))
        assert isinstance(out, RejectReason)
        assert "commission" in out.reason.lower()

    def test_cap_lifted_allows_high_price(self):
        # Stock at $400 with $1000 cap → qty = floor(1000/400) = 2.
        # Notional = $800. Commission = 0.08% × $800 = $0.64 → min $8.
        # RT = $16. 5% × $800 = $40. $16 < $40 → PASS.
        sizer = _sizer(entry_price=400.0, per_trade_cap=1000.0)
        out = sizer.size(FakeSignal(ticker="AAPL"))
        assert isinstance(out, TradeOrder)
        assert out.qty == 2.0
        assert out.notional == pytest.approx(800.0)


# --- $8 minimum commission rejection (>5% round-trip) ----------------


class TestMinimumCommissionRejection:
    def test_aapl_at_100_rejected_by_commission(self):
        # AAPL (stock) at $100 with $200 cap → qty = 2, notional = $200.
        # Commission = max($8, 0.08%×$200) = $8. RT = $16. Threshold = $10. REJECT.
        sizer = _sizer(entry_price=100.0, per_trade_cap=200.0)
        out = sizer.size(FakeSignal(ticker="AAPL"))
        assert isinstance(out, RejectReason)
        assert "commission" in out.reason.lower()
        assert "16" in out.reason  # the round-trip dollar amount

    def test_reject_message_includes_thresholds(self):
        sizer = _sizer(entry_price=100.0, per_trade_cap=200.0)
        out = sizer.size(FakeSignal(ticker="AAPL"))
        assert isinstance(out, RejectReason)
        assert "threshold" in out.reason
        assert "5%" in out.reason


# --- ETF 0% acceptance (under $100k monthly volume) ------------------


class TestETFZeroCommission:
    def test_etf_no_commission_charged(self):
        # SPY at $100, $200 cap → 2 shares, $200 notional, $0 commission.
        sizer = _sizer(entry_price=100.0, per_trade_cap=200.0)
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, TradeOrder)
        assert out.commission_rt == 0.0

    def test_etf_above_free_tier_with_cap(self):
        # SPY at $600, $200 cap → qty = 0.3333 (fractional allowed for ETFs),
        # notional $199.98, commission $0 (under $100k free tier) → PASS.
        # The "above free tier" comment is misleading here — the sizer caps
        # the trade below the free tier automatically. This is a feature:
        # the per-trade cap is a stronger constraint than the free tier for
        # any reasonable price.
        sizer = _sizer(entry_price=600.0, per_trade_cap=200.0)
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, TradeOrder)
        assert out.commission_rt == 0.0


# --- Max-5-position cap (6th approval rejected) ----------------------


class TestMaxOpenPositions:
    def test_at_capacity_rejects(self):
        sizer = _sizer(entry_price=100.0, max_open_positions=5)
        out = sizer.size(FakeSignal(ticker="SPY"), open_position_count=5)
        assert isinstance(out, RejectReason)
        assert "max_open_positions" in out.reason

    def test_one_below_capacity_passes_etf(self):
        sizer = _sizer(entry_price=100.0, max_open_positions=5)
        out = sizer.size(FakeSignal(ticker="SPY"), open_position_count=4)
        assert isinstance(out, TradeOrder)

    def test_explicit_zero_positions_passes(self):
        sizer = _sizer(entry_price=100.0, max_open_positions=5)
        out = sizer.size(FakeSignal(ticker="SPY"), open_position_count=0)
        assert isinstance(out, TradeOrder)


# --- R:R < 2:1 rejection ----------------------------------------------


class TestRRRatio:
    def test_default_rr_exactly_two(self):
        sizer = _sizer()
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, TradeOrder)
        assert out.r_r_ratio == pytest.approx(2.0)

    def test_tight_sl_low_rr_rejects(self):
        # sl_pct=0.05, tp_pct=0.04 → R/R = 4/5 = 0.8 < 2.0.
        sizer = _sizer(sl_pct=0.05, tp_pct=0.04)
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, RejectReason)
        assert "R/R" in out.reason

    def test_equal_sl_tp_rejects(self):
        # 1:1 R/R is below 2:1.
        sizer = _sizer(sl_pct=0.05, tp_pct=0.05)
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, RejectReason)

    def test_higher_rr_passes(self):
        # 3:1 R/R passes the floor.
        sizer = _sizer(sl_pct=0.05, tp_pct=0.15)
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, TradeOrder)
        assert out.r_r_ratio == pytest.approx(3.0)


# --- Entry price source ----------------------------------------------


class TestEntryPriceSource:
    def test_none_price_rejects(self):
        sizer = _sizer(entry_price=None)
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, RejectReason)
        assert "entry price unavailable" in out.reason

    def test_zero_price_rejects(self):
        sizer = _sizer(entry_price=0.0)
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, RejectReason)
        assert "non-positive" in out.reason

    def test_negative_price_rejects(self):
        sizer = _sizer(entry_price=-5.0)
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, RejectReason)

    def test_fetcher_called_with_uppercase_ticker(self):
        called_with: list[str] = []

        def fetcher(t: str) -> Optional[float]:
            called_with.append(t)
            return 100.0

        sizer = PositionSizer(
            per_trade_cap=200.0,
            max_open_positions=5,
            sl_pct=0.07,
            tp_pct=0.14,
            max_commission_pct=0.05,
            entry_price_fetcher=fetcher,
            verbose=False,
        )
        sizer.size(FakeSignal(ticker="aapl"))  # lowercase input
        assert called_with == ["AAPL"]


# --- Signal error -----------------------------------------------------


class TestSignalErrors:
    def test_signal_with_error_rejected(self):
        sizer = _sizer()
        sig = FakeSignal(ticker="AAPL", error="yfinance down")
        out = sizer.size(sig)
        assert isinstance(out, RejectReason)
        assert "signal has error" in out.reason
        assert "yfinance down" in out.reason

    def test_missing_ticker_rejected(self):
        sizer = _sizer()
        # The dataclass requires a ticker, so we make one.
        sig = FakeSignal(ticker="")
        out = sizer.size(sig)
        assert isinstance(out, RejectReason)
        assert "ticker" in out.reason.lower()


# --- qty precision: whole shares for stocks, fractional for ETFs -----


class TestQtyPrecision:
    def test_stock_whole_shares(self):
        # AAPL at $150 with $200 cap → qty = floor(200/150) = 1.
        sizer = _sizer(entry_price=150.0, per_trade_cap=200.0, allow_fractional=True)
        out = sizer.size(FakeSignal(ticker="AAPL"))
        # 1 × $150 = $150. Commission = max($8, 0.08%×$150) = $8. RT=$16. 5%×$150=$7.50. REJECT.
        assert isinstance(out, RejectReason)

    def test_etf_fractional_allowed(self):
        # SPY at $150 with $200 cap → raw = 1.3333, fractional → 1.3333.
        sizer = _sizer(entry_price=150.0, per_trade_cap=200.0, allow_fractional=True)
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, TradeOrder)
        assert out.qty == pytest.approx(1.3333, rel=1e-3)
        # Notional ≈ 200.
        assert out.notional == pytest.approx(200.0, rel=1e-3)

    def test_stock_fractional_disabled(self):
        # allow_fractional=False → stocks round down (whole shares).
        sizer = _sizer(entry_price=150.0, per_trade_cap=200.0, allow_fractional=False)
        out = sizer.size(FakeSignal(ticker="AAPL"))
        # 1 share × $150 = $150. Commission=$8 → REJECT (5% of $150 = $7.50).
        assert isinstance(out, RejectReason)


# --- Sizer never raises ---------------------------------------------


class TestNeverRaises:
    def test_sizer_swallows_fetcher_exception(self):
        def broken_fetcher(_t: str) -> Optional[float]:
            raise RuntimeError("yfinance kaboom")

        sizer = PositionSizer(
            per_trade_cap=200.0,
            max_open_positions=5,
            sl_pct=0.07,
            tp_pct=0.14,
            max_commission_pct=0.05,
            entry_price_fetcher=broken_fetcher,
            verbose=False,
        )
        out = sizer.size(FakeSignal(ticker="SPY"))
        # The fetcher raises, the sizer converts to None, then rejects.
        assert isinstance(out, RejectReason)


# --- Construction validation ----------------------------------------


class TestConstruction:
    def test_default_construction(self):
        sizer = PositionSizer(verbose=False)
        assert sizer.per_trade_cap == 200.0
        assert sizer.max_open_positions == 5
        assert sizer.sl_pct == 0.07
        assert sizer.tp_pct == 0.14
        assert sizer.max_commission_pct == 0.05

    def test_invalid_per_trade_cap(self):
        with pytest.raises(ValueError, match="per_trade_cap"):
            PositionSizer(per_trade_cap=0, verbose=False)

    def test_invalid_max_open_positions(self):
        with pytest.raises(ValueError, match="max_open_positions"):
            PositionSizer(max_open_positions=0, verbose=False)

    def test_invalid_sl_pct(self):
        with pytest.raises(ValueError, match="sl_pct"):
            PositionSizer(sl_pct=0, verbose=False)
        with pytest.raises(ValueError, match="sl_pct"):
            PositionSizer(sl_pct=1, verbose=False)

    def test_invalid_tp_pct(self):
        with pytest.raises(ValueError, match="tp_pct"):
            PositionSizer(tp_pct=0, verbose=False)
        with pytest.raises(ValueError, match="tp_pct"):
            PositionSizer(tp_pct=1, verbose=False)

    def test_invalid_max_commission_pct(self):
        with pytest.raises(ValueError, match="max_commission_pct"):
            PositionSizer(max_commission_pct=0, verbose=False)
        with pytest.raises(ValueError, match="max_commission_pct"):
            PositionSizer(max_commission_pct=1, verbose=False)


# --- from_config factory --------------------------------------------


class TestFromConfig:
    def test_from_config_happy(self):
        @dataclass(frozen=True)
        class FakeCfg:
            xtb_per_trade_cap: float = 250.0
            xtb_max_open_positions: int = 3
            xtb_sl_pct: float = 0.08
            xtb_tp_pct: float = 0.16
            xtb_max_commission_pct: float = 0.04

        sizer = PositionSizer.from_config(FakeCfg())  # type: ignore[arg-type]
        assert sizer.per_trade_cap == 250.0
        assert sizer.max_open_positions == 3
        assert sizer.sl_pct == 0.08
        assert sizer.tp_pct == 0.16
        assert sizer.max_commission_pct == 0.04

    def test_from_config_uses_defaults_when_attrs_missing(self):
        @dataclass(frozen=True)
        class EmptyCfg:
            pass

        sizer = PositionSizer.from_config(EmptyCfg())  # type: ignore[arg-type]
        assert sizer.per_trade_cap == 200.0
        assert sizer.max_open_positions == 5


# --- TradeOrder.to_dict ---------------------------------------------


class TestTradeOrderToDict:
    def test_to_dict_round_trip(self):
        sizer = _sizer()
        out = sizer.size(FakeSignal(ticker="SPY"))
        assert isinstance(out, TradeOrder)
        d = out.to_dict()
        assert d["ticker"] == "SPY"
        assert d["qty"] == 2.0
        assert d["entry"] == 100.0
        assert d["instrument"] == "us_etf"  # .value of the enum
        assert "signal_date" in d


# --- RejectReason is frozen + tiny ---------------------------------


class TestRejectReason:
    def test_only_ticker_and_reason(self):
        r = RejectReason(ticker="AAPL", reason="nope")
        assert r.ticker == "AAPL"
        assert r.reason == "nope"
