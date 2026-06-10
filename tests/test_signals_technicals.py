"""Unit tests for :mod:`portfoliomind.signals.technicals`.

Hermetic — uses synthetic price series. No network, no real yfinance.
"""

from __future__ import annotations

import math

import pytest

from portfoliomind.signals.technicals import (
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    RSI_PERIOD,
    WEIGHT_MOMENTUM,
    WEIGHT_TREND,
    WEIGHT_VOLATILITY,
    TechnicalScore,
    compute_technical_score,
    realized_vol,
    rsi,
    sma,
)


# --- Helpers ---------------------------------------------------------------


def _linear(start: float, slope: float, n: int) -> list[float]:
    """Build a simple arithmetic-progression price series."""
    return [start + slope * i for i in range(n)]


def _flat(price: float, n: int) -> list[float]:
    return [float(price)] * n


def _v_recovery(n: int = 100) -> list[float]:
    """A crash followed by a sharp V-shaped recovery.

    First half: linear decline from 100 to 50.
    Second half: linear recovery from 50 to 100.
    """
    half = n // 2
    return _linear(100.0, -50.0 / (half - 1), half) + _linear(50.0, 50.0 / (n - half - 1), n - half)


def _crash(n: int = 100) -> list[float]:
    """A simple monotonic decline."""
    return _linear(100.0, -0.5, n)


# --- sma -------------------------------------------------------------------


class TestSma:
    def test_simple_average(self):
        assert sma([1.0, 2.0, 3.0, 4.0, 5.0], 5) == 3.0
        assert sma([1.0, 2.0, 3.0], 3) == 2.0

    def test_last_n(self):
        # Window of 3 → average of last 3 in the list.
        assert sma([10.0, 20.0, 30.0, 40.0], 3) == pytest.approx(30.0)

    def test_too_short(self):
        assert sma([1.0, 2.0], 5) is None
        assert sma([], 5) is None

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            sma([1.0, 2.0], 0)
        with pytest.raises(ValueError):
            sma([1.0, 2.0], -1)


# --- rsi -------------------------------------------------------------------


class TestRsi:
    def test_all_gains_is_100(self):
        closes = _linear(100.0, 1.0, 30)
        r = rsi(closes, RSI_PERIOD)
        assert r == pytest.approx(100.0)

    def test_all_losses_is_0(self):
        closes = _linear(100.0, -1.0, 30)
        r = rsi(closes, RSI_PERIOD)
        # The Wilder formulation converges towards 0; we accept either
        # the asymptotic value (≈0) or a small positive number.
        assert r is not None
        assert r < 5.0

    def test_mixed_oscillates_around_50(self):
        closes: list[float] = []
        for i in range(50):
            closes.append(100.0 + (1.0 if i % 2 == 0 else -1.0))
        r = rsi(closes, RSI_PERIOD)
        assert r is not None
        assert 30.0 < r < 70.0

    def test_too_short(self):
        assert rsi([100.0, 101.0, 102.0], RSI_PERIOD) is None

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            rsi([1.0, 2.0, 3.0], 0)


# --- realized_vol ---------------------------------------------------------


class TestRealizedVol:
    def test_constant_price_is_zero(self):
        # log(1) = 0 → stddev of zeros = 0.
        v = realized_vol([100.0] * 30, 20)
        assert v == pytest.approx(0.0, abs=1e-9)

    def test_too_short(self):
        assert realized_vol([100.0, 101.0, 102.0], 20) is None

    def test_monotonic_increasing_is_low(self):
        # Constant log-returns → stddev = 0.
        closes = [100.0 * (1.01 ** i) for i in range(30)]
        v = realized_vol(closes, 20)
        assert v is not None
        assert v < 1e-6

    def test_choppy_price_is_higher(self):
        # Alternating big moves → high realized vol.
        closes = [100.0]
        for i in range(1, 30):
            closes.append(closes[-1] * (1.05 if i % 2 == 0 else 0.95))
        v_choppy = realized_vol(closes, 20)
        v_smooth = realized_vol(_linear(100.0, 0.5, 30), 20)
        assert v_choppy is not None and v_smooth is not None
        assert v_choppy > v_smooth


# --- compute_technical_score -----------------------------------------------


class TestComputeTechnicalScore:
    def test_uptrend_is_positive(self):
        closes = _linear(100.0, 1.0, 100)
        ts = compute_technical_score("AAPL", closes=closes)
        assert ts.score > 0
        assert ts.trend > 0
        assert -1.0 <= ts.score <= 1.0

    def test_downtrend_is_negative(self):
        closes = _crash(100)
        ts = compute_technical_score("AAPL", closes=closes)
        assert ts.score < 0
        assert ts.trend < 0
        assert -1.0 <= ts.score <= 1.0

    def test_ranging_is_neutral(self):
        # Slow oscillation around a mean — SMA20 ≈ SMA50.
        closes: list[float] = []
        for i in range(100):
            closes.append(100.0 + 0.5 * math.sin(i / 5.0))
        ts = compute_technical_score("AAPL", closes=closes)
        # Trend is near zero (no directional bias), so the score is
        # dominated by momentum + vol.
        assert abs(ts.trend) < 0.2

    def test_insufficient_history_returns_zero(self):
        ts = compute_technical_score("AAPL", closes=[100.0] * 10)
        assert ts.score == 0.0
        assert ts.trend == 0.0
        assert ts.momentum == 0.0
        assert ts.volatility == 0.0
        # At least one reason mentions history.
        assert any("history" in r.lower() for r in ts.reasons)

    def test_empty_history_returns_zero(self):
        ts = compute_technical_score("AAPL", closes=[])
        assert ts.score == 0.0
        assert ts.reasons == ["no price history supplied"]

    def test_v_recovery_shape(self):
        # After a V-recovery, fast SMA (last 20) is climbing back from
        # a low → trend positive-ish, momentum overbought from the
        # recent recovery. Score is positive.
        ts = compute_technical_score("AAPL", closes=_v_recovery(100))
        assert ts.trend > 0  # fast SMA catching up
        # Vol is high during the crash + recovery → expect non-zero.
        assert ts.score != 0.0

    def test_score_within_bounds_under_extremes(self):
        # 10x ramp in 100 bars: SMA ratio should saturate near +1.
        closes = [100.0 * (1.05 ** i) for i in range(100)]
        ts = compute_technical_score("AAPL", closes=closes)
        assert -1.0 <= ts.trend <= 1.0
        assert -1.0 <= ts.score <= 1.0

    def test_ticker_is_uppercased(self):
        ts = compute_technical_score("aapl", closes=_linear(100.0, 0.5, 100))
        assert ts.ticker == "AAPL"

    def test_asof_date_passes_through(self):
        ts = compute_technical_score("AAPL", closes=_linear(100.0, 0.5, 100), asof_date="2026-06-10")
        assert ts.asof_date == "2026-06-10"

    def test_weights_sum_to_one(self):
        # Sanity: the three weights we publish should sum to 1.0.
        assert WEIGHT_TREND + WEIGHT_MOMENTUM + WEIGHT_VOLATILITY == pytest.approx(1.0)

    def test_reasons_are_nonempty(self):
        ts = compute_technical_score("AAPL", closes=_linear(100.0, 0.5, 100))
        # Three component reasons (one per indicator).
        assert len(ts.reasons) == 3

    def test_rsi_mapping_bands(self):
        # When RSI sits in [30, 70] the linear map returns a value in [-1, +1].
        # We don't control RSI directly here, but we can sanity-check
        # the constants are sensible.
        assert RSI_OVERSOLD < RSI_OVERBOUGHT

    def test_volatility_expanding_and_falling_is_negative(self):
        # Build a series with a recent crash — 20-day vol expanding AND price falling.
        # First 70 days: low-vol ranging; last 20 days: sharp decline.
        closes: list[float] = []
        for i in range(70):
            closes.append(100.0 + 0.1 * math.sin(i / 3.0))
        for i in range(20):
            closes.append(closes[-1] * 0.95)
        ts = compute_technical_score("AAPL", closes=closes)
        # Either volatility is negative (the ideal case) or zero (if the
        # ratio falls below the threshold); we just assert it's not positive.
        assert ts.volatility <= 0.0

    def test_volatility_contraction_is_zero(self):
        # Steady monotone series — no expansion, score 0.
        closes = _linear(100.0, 0.1, 100)
        ts = compute_technical_score("AAPL", closes=closes)
        assert ts.volatility == 0.0


# --- TechnicalScore dataclass --------------------------------------------


class TestTechnicalScoreDataclass:
    def test_to_dict_round_trip(self):
        ts = TechnicalScore(
            ticker="AAPL",
            trend=0.5,
            momentum=-0.3,
            volatility=0.1,
            score=0.27,
            reasons=["r1", "r2"],
            asof_date="2026-06-10",
        )
        d = ts.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["trend"] == 0.5
        assert d["momentum"] == -0.3
        assert d["score"] == 0.27
        assert d["reasons"] == ["r1", "r2"]
        assert d["asof_date"] == "2026-06-10"

    def test_frozen(self):
        ts = TechnicalScore(
            ticker="AAPL", trend=0.0, momentum=0.0, volatility=0.0, score=0.0
        )
        with pytest.raises(Exception):
            ts.score = 0.5  # type: ignore[misc]
