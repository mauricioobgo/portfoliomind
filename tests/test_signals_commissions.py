"""Exhaustive unit tests for :mod:`portfoliomind.signals.commissions`.

The card 7 spec lists the exact XTB math:

* US stock: ``max($8, 0.08% * notional)`` per side
* US ETF: 0% up to $100k monthly volume, then 0.08% per side
* Round-trip = 2 × one-way

This module is **pure math** — no network, no I/O, no clock. The
tests are hermetic; they construct an :class:`XTBCommissionModel`
directly and assert the exact dollar amounts.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from portfoliomind.signals.commissions import (
    MONEY_QUANTUM,
    US_ETF_FREE_TIER,
    US_ETF_PCT_OF_NOTIONAL,
    US_STOCK_MIN_COMMISSION,
    US_STOCK_PCT_OF_NOTIONAL,
    InstrumentType,
    XTBCommissionModel,
    default_model,
)


# --- One-way, US stocks --------------------------------------------------


class TestUSStockOneWay:
    model = XTBCommissionModel()

    def test_minimum_under_10k(self):
        # $1,000 notional: 0.08% = $0.80; min $8 wins.
        assert self.model.one_way(1000, InstrumentType.US_STOCK) == Decimal("8.00")

    def test_minimum_under_10k_alt(self):
        # $9,999: 0.08% = $7.99; min $8 wins.
        assert self.model.one_way(9999, InstrumentType.US_STOCK) == Decimal("8.00")

    def test_percentage_at_exactly_10k(self):
        # $10,000: 0.08% = $8.00; min $8 ties. Result is $8.00.
        assert self.model.one_way(10_000, InstrumentType.US_STOCK) == Decimal("8.00")

    def test_percentage_dominates_above_10k(self):
        # $20,000: 0.08% = $16.00; min $8 — percentage wins.
        assert self.model.one_way(20_000, InstrumentType.US_STOCK) == Decimal("16.00")

    def test_large_notional(self):
        # $1,000,000: 0.08% = $800.
        assert self.model.one_way(1_000_000, InstrumentType.US_STOCK) == Decimal("800.00")


# --- Round-trip, US stocks -----------------------------------------------


class TestUSStockRoundTrip:
    model = XTBCommissionModel()

    def test_round_trip_doubles_one_way(self):
        # $20,000: one-way $16, round-trip $32.
        rt = self.model.round_trip(20_000, InstrumentType.US_STOCK)
        assert rt == Decimal("32.00")

    def test_round_trip_at_minimum(self):
        # $1,000: one-way $8, round-trip $16.
        rt = self.model.round_trip(1_000, InstrumentType.US_STOCK)
        assert rt == Decimal("16.00")


# --- One-way, US ETFs (free tier) ---------------------------------------


class TestUSETFOneWay:
    model = XTBCommissionModel()

    def test_etf_under_free_tier(self):
        # $50,000: 0% (under $100k free tier).
        assert self.model.one_way(50_000, InstrumentType.US_ETF) == Decimal("0.00")

    def test_etf_exactly_at_free_tier(self):
        # $100,000: at the boundary, still 0% (the spec is "up to $100k").
        assert self.model.one_way(100_000, InstrumentType.US_ETF) == Decimal("0.00")

    def test_etf_above_free_tier(self):
        # $200,000: billable = $200,000 - $100,000 = $100,000; fee = 0.08% × $100,000 = $80.
        assert self.model.one_way(200_000, InstrumentType.US_ETF) == Decimal("80.00")

    def test_etf_just_above_free_tier(self):
        # $100,001: billable = $1; fee = 0.08% × $1 = $0.0008 → quantize to $0.00.
        assert self.model.one_way(100_001, InstrumentType.US_ETF) == Decimal("0.00")


# --- US ETF monthly volume tracking -------------------------------------


class TestUSETFMonthlyVolume:
    model = XTBCommissionModel()

    def test_zero_volume_used(self):
        # Fresh month: $200k notional → $80.
        assert (
            self.model.one_way(200_000, InstrumentType.US_ETF, monthly_volume_used=0)
            == Decimal("80.00")
        )

    def test_volume_partially_consumed(self):
        # $60k already traded this month: free tier remaining = $40k.
        # $200k notional: billable = $200k - $40k = $160k; fee = 0.08% × $160k = $128.
        assert (
            self.model.one_way(200_000, InstrumentType.US_ETF, monthly_volume_used=60_000)
            == Decimal("128.00")
        )

    def test_volume_fully_consumed(self):
        # $100k+ already used → no free tier left.
        # $200k notional: billable = $200k; fee = 0.08% × $200k = $160.
        assert (
            self.model.one_way(200_000, InstrumentType.US_ETF, monthly_volume_used=100_000)
            == Decimal("160.00")
        )

    def test_volume_over_consumed(self):
        # $150k used (over the $100k tier): free tier is clamped to 0.
        # $200k notional: billable = $200k; fee = 0.08% × $200k = $160.
        assert (
            self.model.one_way(200_000, InstrumentType.US_ETF, monthly_volume_used=150_000)
            == Decimal("160.00")
        )

    def test_volume_consumes_for_small_trade(self):
        # $80k used, $50k trade: free tier remaining = $20k.
        # Billable = $50k - $20k = $30k; fee = 0.08% × $30k = $24.
        assert (
            self.model.one_way(50_000, InstrumentType.US_ETF, monthly_volume_used=80_000)
            == Decimal("24.00")
        )


# --- Round-trip spec test case (US ETF $200k) --------------------------


class TestCard7SpecAcceptance:
    """The exact cases the card 7 spec lists as acceptance criteria."""

    def test_us_stock_1000_one_way(self):
        # US stock $1000 notional → $8 minimum
        m = XTBCommissionModel()
        assert m.one_way(1000, InstrumentType.US_STOCK) == Decimal("8.00")

    def test_us_stock_20000_one_way(self):
        # US stock $20,000 notional → $16 (0.08%)
        m = XTBCommissionModel()
        assert m.one_way(20_000, InstrumentType.US_STOCK) == Decimal("16.00")

    def test_us_etf_50000_one_way(self):
        # US ETF $50,000 notional → $0 (under free tier)
        m = XTBCommissionModel()
        assert m.one_way(50_000, InstrumentType.US_ETF) == Decimal("0.00")

    def test_us_etf_200000_round_trip(self):
        # US ETF $200,000 notional → $160 (above $100k threshold)
        # Above free tier by $100k; 0.08% × $100k = $80 one-way; $160 round-trip.
        m = XTBCommissionModel()
        assert m.round_trip(200_000, InstrumentType.US_ETF) == Decimal("160.00")


# --- Edge cases ---------------------------------------------------------


class TestEdgeCases:
    model = XTBCommissionModel()

    def test_zero_notional_us_stock(self):
        # 0 notional: 0.08% × 0 = 0; min $8 still applies → $8.
        # This is intentional: a $0 trade isn't a valid trade, but the
        # commission model is the *pricing* function, not the *gating*
        # function. The sizer rejects 0-qty trades before computing
        # commission.
        assert self.model.one_way(0, InstrumentType.US_STOCK) == Decimal("8.00")

    def test_zero_notional_us_etf(self):
        # 0 notional ETF: 0% of 0 is 0.
        assert self.model.one_way(0, InstrumentType.US_ETF) == Decimal("0.00")

    def test_negative_notional_rejected(self):
        with pytest.raises(ValueError, match="notional must be >= 0"):
            self.model.one_way(-100, InstrumentType.US_STOCK)

    def test_negative_volume_rejected(self):
        with pytest.raises(ValueError, match="monthly_volume_used must be >= 0"):
            self.model.one_way(100, InstrumentType.US_ETF, monthly_volume_used=-1)

    def test_unknown_instrument_rejected(self):
        with pytest.raises(ValueError, match="unknown instrument type"):
            # Pass a value outside the enum by hand.
            self.model.one_way(100, "us_option")  # type: ignore[arg-type]

    def test_money_quantum(self):
        # All commissions come back at 2-decimal precision.
        for v in [1.0, 7.99, 8.005, 100.123, 999.999]:
            c = self.model.one_way(v, InstrumentType.US_STOCK)
            assert c.as_tuple().exponent == -2  # type: ignore[attr-defined]

    def test_rounding_half_up(self):
        # 0.08% of $123,456 = $98.7648 → $98.76 (rounded half-up).
        c = self.model.one_way(123_456, InstrumentType.US_STOCK)
        assert c == Decimal("98.76")

    def test_decimal_input_accepted(self):
        # Decimal inputs are valid; results are still quantized to 2 dp.
        c = self.model.one_way(Decimal("10000"), InstrumentType.US_STOCK)
        assert c == Decimal("8.00")


# --- Module-level singletons -------------------------------------------


class TestModuleSurface:
    def test_constants_present(self):
        assert US_STOCK_MIN_COMMISSION == Decimal("8.00")
        assert US_STOCK_PCT_OF_NOTIONAL == Decimal("0.0008")
        assert US_ETF_FREE_TIER == Decimal("100000.00")
        assert US_ETF_PCT_OF_NOTIONAL == Decimal("0.0008")
        assert MONEY_QUANTUM == Decimal("0.01")

    def test_instrument_type_string_value(self):
        # The spec says: "future-proof: a small enum, not a string".
        # We use a str-valued Enum so the .value round-trips through
        # JSON / Sheets cleanly. Verify both.
        assert InstrumentType.US_STOCK == "us_stock"
        assert InstrumentType.US_ETF == "us_etf"
        assert InstrumentType.US_STOCK.value == "us_stock"
        assert isinstance(InstrumentType.US_STOCK, str)

    def test_default_model_is_singleton(self):
        m1 = default_model()
        m2 = default_model()
        assert m1 is m2
        # The default uses the standard XTB constants.
        assert m1.us_stock_min == US_STOCK_MIN_COMMISSION
        assert m1.us_etf_free_tier == US_ETF_FREE_TIER
