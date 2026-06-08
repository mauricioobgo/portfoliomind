"""Unit tests for OrderSpec validation.

These tests are pure-Python — no Playwright, no network, no Google Sheets.
They cover the PortfolioMind v4 iron rules:

  * SL is mandatory (must be > 0, must be finite).
  * TP is mandatory (must be > 0, must be finite).
  * SL and TP are on the correct side of the entry price for the order side.
  * Qty must be > 0 and finite.
  * Entry price may be 0 (market) or > 0, but must be finite and non-negative.

If any of these tests fail, the agent is one click away from sending an
unprotected order. Don't relax them.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pytest

from portfoliomind.xtb.order import (
    OrderSide,
    OrderSpec,
    ValidationError,
    validate_order,
)
from portfoliomind.xtb.screenshot import screenshot_basename_for


# --- Tiny factories ----------------------------------------------------------


def _buy(qty: float = 10, entry: float = 100.0, sl: float = 95.0, tp: float = 110.0) -> OrderSpec:
    return OrderSpec(ticker="AAPL.US", side=OrderSide.BUY, qty=qty, entry_price=entry, sl=sl, tp=tp)


def _sell(qty: float = 10, entry: float = 100.0, sl: float = 105.0, tp: float = 90.0) -> OrderSpec:
    return OrderSpec(ticker="AAPL.US", side=OrderSide.SELL, qty=qty, entry_price=entry, sl=sl, tp=tp)


# --- Happy paths -------------------------------------------------------------


class TestHappyPath:
    def test_buy_with_sl_and_tp_validates(self):
        # Should not raise.
        validate_order(_buy())

    def test_sell_with_sl_and_tp_validates(self):
        # Should not raise.
        validate_order(_sell())

    def test_buy_with_market_order_validates(self):
        # entry_price=0 is allowed (market order); SL/TP sanity check is
        # skipped when there's no entry anchor, but SL and TP themselves
        # are still mandatory and must be > 0.
        spec = _buy(entry=0.0, sl=95.0, tp=110.0)
        validate_order(spec)

    def test_checked_convenience_constructs_a_validated_spec(self):
        spec = OrderSpec.checked(
            ticker="EURUSD",
            side="buy",  # case-insensitive string
            qty=1.0,
            entry_price=1.10,
            sl=1.09,
            tp=1.12,
        )
        assert spec.side is OrderSide.BUY
        # And the spec round-trips through validate_order.
        validate_order(spec)

    def test_checked_accepts_lowercase_side(self):
        spec = OrderSpec.checked(
            ticker="X", side="sell", qty=1, entry_price=100, sl=105, tp=90
        )
        assert spec.side is OrderSide.SELL

    def test_checked_accepts_uppercase_side(self):
        spec = OrderSpec.checked(
            ticker="X", side="BUY", qty=1, entry_price=100, sl=95, tp=110
        )
        assert spec.side is OrderSide.BUY


# --- SL mandatory ------------------------------------------------------------


class TestStopLossMandatory:
    def test_sl_zero_raises(self):
        with pytest.raises(ValidationError, match=r"Stop-loss.*mandatory"):
            validate_order(_buy(sl=0.0))

    def test_sl_none_raises(self):
        # OrderSpec is a frozen dataclass; we cannot set sl=None in a sane
        # way, so we test via the validator with a spec where sl is
        # 0.0 (the closest thing to "missing"). 0.0 already fails above.
        # This test pins the message text.
        with pytest.raises(ValidationError, match=r"sl=0\.0"):
            validate_order(_buy(sl=0.0))

    def test_sl_negative_raises(self):
        # Negative SL is treated as "missing" by the validator (only
        # positive finite numbers are accepted).
        with pytest.raises(ValidationError, match=r"Stop-loss"):
            validate_order(_buy(sl=-5.0))

    def test_sl_nan_raises(self):
        with pytest.raises(ValidationError, match=r"Stop-loss"):
            validate_order(_buy(sl=float("nan")))

    def test_sl_infinity_raises(self):
        with pytest.raises(ValidationError, match=r"Stop-loss"):
            validate_order(_buy(sl=float("inf")))

    def test_sl_string_raises(self):
        # A non-numeric string that float() cannot parse. Numeric strings
        # like "95" would be silently coerced to 95.0; the validator
        # explicitly rejects that as ambiguous.
        spec = OrderSpec("AAPL.US", OrderSide.BUY, 1, 100, sl="ninety-five", tp=110)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match=r"Stop-loss"):
            validate_order(spec)


# --- TP mandatory ------------------------------------------------------------


class TestTakeProfitMandatory:
    def test_tp_zero_raises(self):
        with pytest.raises(ValidationError, match=r"Take-profit.*mandatory"):
            validate_order(_buy(tp=0.0))

    def test_tp_negative_raises(self):
        with pytest.raises(ValidationError, match=r"Take-profit"):
            validate_order(_buy(tp=-1.0))

    def test_tp_nan_raises(self):
        with pytest.raises(ValidationError, match=r"Take-profit"):
            validate_order(_buy(tp=float("nan")))

    def test_tp_infinity_raises(self):
        with pytest.raises(ValidationError, match=r"Take-profit"):
            validate_order(_buy(tp=float("inf")))


# --- SL/TP side-of-entry sanity check ---------------------------------------


class TestSideOfEntrySanity:
    def test_buy_sl_above_entry_raises(self):
        with pytest.raises(ValidationError, match=r"BUY order.*sl.*below"):
            validate_order(_buy(entry=100, sl=105, tp=110))

    def test_buy_tp_below_entry_raises(self):
        with pytest.raises(ValidationError, match=r"BUY order.*tp.*above"):
            validate_order(_buy(entry=100, sl=95, tp=90))

    def test_sell_sl_below_entry_raises(self):
        with pytest.raises(ValidationError, match=r"SELL order.*sl.*above"):
            validate_order(_sell(entry=100, sl=95, tp=90))

    def test_sell_tp_above_entry_raises(self):
        with pytest.raises(ValidationError, match=r"SELL order.*tp.*below"):
            validate_order(_sell(entry=100, sl=105, tp=110))

    def test_buy_with_market_order_skips_side_check(self):
        # entry=0, so the side-of-entry check is skipped, but SL and TP
        # are still > 0 so the rule passes.
        spec = _buy(entry=0, sl=0.01, tp=0.02)
        validate_order(spec)

    def test_buy_sl_equal_to_entry_raises(self):
        # Strictly less than: equal-to-entry would mean "stop at the
        # price you just paid", which is a no-op stop.
        with pytest.raises(ValidationError, match=r"BUY order.*sl.*below"):
            validate_order(_buy(entry=100, sl=100, tp=110))

    def test_sell_sl_equal_to_entry_raises(self):
        with pytest.raises(ValidationError, match=r"SELL order.*sl.*above"):
            validate_order(_sell(entry=100, sl=100, tp=90))


# --- Qty and entry price -----------------------------------------------------


class TestQty:
    def test_qty_zero_raises(self):
        with pytest.raises(ValidationError, match=r"qty must be"):
            validate_order(_buy(qty=0))

    def test_qty_negative_raises(self):
        with pytest.raises(ValidationError, match=r"qty must be"):
            validate_order(_buy(qty=-5))

    def test_qty_nan_raises(self):
        with pytest.raises(ValidationError, match=r"qty must be"):
            validate_order(_buy(qty=float("nan")))

    def test_qty_infinity_raises(self):
        with pytest.raises(ValidationError, match=r"qty must be"):
            validate_order(_buy(qty=float("inf")))

    def test_qty_string_raises(self):
        # A non-numeric string that float() cannot parse.
        spec = OrderSpec("AAPL.US", OrderSide.BUY, qty="ten", entry_price=100, sl=95, tp=110)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match=r"qty must be"):
            validate_order(spec)

    def test_qty_fractional_allowed(self):
        # Fractional shares (US brokers allow this since 2020).
        validate_order(_buy(qty=0.5))


class TestEntryPrice:
    def test_entry_zero_is_market_order(self):
        # Allowed but SL/TP still must be on their own merits.
        validate_order(_buy(entry=0, sl=95, tp=110))

    def test_entry_negative_raises(self):
        with pytest.raises(ValidationError, match=r"entry_price must be"):
            validate_order(_buy(entry=-1.0))

    def test_entry_nan_raises(self):
        with pytest.raises(ValidationError, match=r"entry_price must be"):
            validate_order(_buy(entry=float("nan")))


# --- Ticker ------------------------------------------------------------------


class TestTicker:
    def test_empty_ticker_raises(self):
        with pytest.raises(ValidationError, match=r"ticker must be"):
            validate_order(OrderSpec("", OrderSide.BUY, 1, 100, 95, 110))

    def test_whitespace_ticker_raises(self):
        with pytest.raises(ValidationError, match=r"ticker must be"):
            validate_order(OrderSpec("   ", OrderSide.BUY, 1, 100, 95, 110))

    def test_ticker_non_string_raises(self):
        spec = OrderSpec(123, OrderSide.BUY, 1, 100, 95, 110)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match=r"ticker must be"):
            validate_order(spec)


# --- Side enum ---------------------------------------------------------------


class TestSide:
    def test_side_must_be_enum(self):
        # Bypass the dataclass type check by passing a non-OrderSide value
        # in via a freshly built spec. (The dataclass won't enforce this
        # at runtime, so the validator has to.)
        spec = OrderSpec("AAPL.US", "BUY", 1, 100, 95, 110)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match=r"side must be"):
            validate_order(spec)


# --- Checked() input validation ---------------------------------------------


class TestCheckedInputValidation:
    @pytest.mark.parametrize("bad_side", ["", "HOLD", "long", "short", "foo"])
    def test_invalid_side_strings_rejected(self, bad_side: str):
        with pytest.raises(ValidationError, match=r"Invalid side"):
            OrderSpec.checked("AAPL.US", bad_side, 1, 100, 95, 110)

    def test_non_string_non_enum_side_rejected(self):
        with pytest.raises(ValidationError, match=r"side must be str or OrderSide"):
            OrderSpec.checked("AAPL.US", 123, 1, 100, 95, 110)  # type: ignore[arg-type]

    def test_checked_does_not_swallow_sl_errors(self):
        with pytest.raises(ValidationError, match=r"Stop-loss"):
            OrderSpec.checked("AAPL.US", "BUY", 1, 100, sl=0, tp=110)

    def test_checked_does_not_swallow_tp_errors(self):
        with pytest.raises(ValidationError, match=r"Take-profit"):
            OrderSpec.checked("AAPL.US", "BUY", 1, 100, sl=95, tp=0)


# --- Immutability ------------------------------------------------------------


class TestImmutability:
    def test_spec_is_frozen(self):
        spec = _buy()
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.qty = 999  # type: ignore[misc]

    def test_spec_is_hashable(self):
        # Frozen dataclasses are hashable by default.
        spec = _buy()
        assert hash(spec) == hash(_buy())
        # And we can use it in a set.
        assert {_buy(), _buy()} == {spec}


# --- Property-based sanity (a few hand-picked) ------------------------------


@pytest.mark.parametrize(
    "side,entry,sl,tp",
    [
        # BUY: SL below entry, TP above entry
        (OrderSide.BUY, 100, 95, 110),
        (OrderSide.BUY, 50.25, 49.50, 52.00),
        (OrderSide.BUY, 0.99, 0.95, 1.05),
        # SELL: SL above entry, TP below entry
        (OrderSide.SELL, 100, 105, 90),
        (OrderSide.SELL, 50.25, 51.00, 49.00),
    ],
)
def test_valid_specs_round_trip(side: OrderSide, entry: float, sl: float, tp: float):
    spec = OrderSpec("AAPL.US", side, 1, entry, sl, tp)
    validate_order(spec)  # should not raise


# --- Math: infinity, NaN, and friends (defense in depth) ---------------------


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, "not a number", None])
def test_sl_rejects_non_finite_or_non_numeric(bad_value: Any):
    spec = OrderSpec("AAPL.US", OrderSide.BUY, 1, 100, sl=bad_value, tp=110)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match=r"Stop-loss"):
        validate_order(spec)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, "not a number", None])
def test_tp_rejects_non_finite_or_non_numeric(bad_value: Any):
    spec = OrderSpec("AAPL.US", OrderSide.BUY, 1, 100, sl=95, tp=bad_value)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match=r"Take-profit"):
        validate_order(spec)


# --- Screenshot basename helper --------------------------------------------


class TestScreenshotBasename:
    """Sanity check for the canonical filename format. The card body
    specifies ``SCREENSHOT_DIR/xtb_<ticker>_<side>_<ts>.png``."""

    def test_canonical_format(self):
        ts = datetime(2026, 6, 8, 8, 30, 0)
        name = screenshot_basename_for("AAPL.US", "BUY", "pre", timestamp=ts)
        assert name == "xtb_AAPL.US_BUY_pre_20260608T083000.png"

    def test_sell_phase_post(self):
        ts = datetime(2026, 6, 8, 16, 0, 0)
        name = screenshot_basename_for("EURUSD", "SELL", "post", timestamp=ts)
        assert name == "xtb_EURUSD_SELL_post_20260608T160000.png"

    def test_phase_failure_capture(self):
        ts = datetime(2026, 6, 8, 9, 0, 0)
        name = screenshot_basename_for("TSLA.US", "BUY", "post_FAIL", timestamp=ts)
        assert name == "xtb_TSLA.US_BUY_post_FAIL_20260608T090000.png"

    def test_unsafe_chars_in_ticker_sanitized(self):
        ts = datetime(2026, 6, 8, 9, 0, 0)
        # Slashes and spaces would break a filesystem path.
        name = screenshot_basename_for("BRK/B.US", "BUY", "pre", timestamp=ts)
        assert "/" not in name
        assert " " not in name
        assert name.startswith("xtb_BRK_B.US_BUY_pre_")

    def test_unsafe_chars_in_side_sanitized(self):
        ts = datetime(2026, 6, 8, 9, 0, 0)
        # The function is called with a side like "BUY" but defends
        # against garbage anyway.
        name = screenshot_basename_for("X", "BUY/SELL", "pre", timestamp=ts)
        assert "/" not in name
        assert name.startswith("xtb_X_BUY_SELL_pre_")

    def test_uses_now_when_no_timestamp(self):
        # We can't assert on the timestamp itself, but the name should
        # match the canonical pattern.
        name = screenshot_basename_for("AAPL.US", "BUY", "pre")
        import re
        assert re.match(r"^xtb_AAPL\.US_BUY_pre_\d{8}T\d{6}\.png$", name)
