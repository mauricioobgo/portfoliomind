"""Hermetic tests for :mod:`portfoliomind.signals.patterns`.

All series are synthetic — no yfinance, no network. Each detector is
exercised with a series engineered to fire it and one that must not.
"""

from __future__ import annotations

import pytest

from portfoliomind.signals.patterns import (
    MIN_PATTERN_BARS,
    P_CEIL,
    PRIOR_P_UP,
    BullishPatterns,
    PatternHit,
    aggregate_probability,
    detect_breakout,
    detect_bullish_patterns,
    detect_golden_cross,
    detect_higher_lows,
    detect_macd_cross,
    detect_rsi_recovery,
    detect_uptrend_stack,
    ema_series,
    macd_histogram,
)


# --- Synthetic series ---------------------------------------------------------


def rising(n: int = 120, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start + step * i for i in range(n)]


def falling(n: int = 120, start: float = 200.0, step: float = 0.5) -> list[float]:
    return [start - step * i for i in range(n)]


def flat(n: int = 120, level: float = 100.0) -> list[float]:
    return [level] * n


def v_shape_recent_cross() -> list[float]:
    """60-bar decline then a 20-bar sharp rise — SMA20 crosses above
    SMA50 a couple of bars before the end (verified against the
    production sma())."""
    return [160.0 - i for i in range(60)] + [101.0 + 2.0 * (i + 1) for i in range(20)]


# --- Pure helpers ----------------------------------------------------------------


def test_ema_series_basics():
    assert ema_series([], 10) == []
    out = ema_series([1.0, 1.0, 1.0], 3)
    assert out == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        ema_series([1.0], 0)


def test_macd_histogram_flat_is_zero():
    hist = macd_histogram(flat(100))
    assert hist, "flat 100-bar series should produce a histogram"
    assert all(abs(h) < 1e-9 for h in hist)


def test_macd_histogram_short_series_empty():
    assert macd_histogram(flat(20)) == []


# --- Individual detectors ----------------------------------------------------------


def test_golden_cross_fires_on_recent_cross():
    hit = detect_golden_cross(v_shape_recent_cross())
    assert hit is not None
    assert hit.name == "golden_cross"
    assert 0.5 < hit.hit_rate < 1.0


def test_golden_cross_silent_on_established_uptrend():
    """A long uptrend crossed ages ago — not a *recent* cross."""
    assert detect_golden_cross(rising(150)) is None


def test_golden_cross_silent_on_downtrend():
    assert detect_golden_cross(falling()) is None


def test_uptrend_stack_fires_on_rising_series():
    hit = detect_uptrend_stack(rising())
    assert hit is not None and hit.name == "uptrend_stack"


def test_uptrend_stack_silent_on_falling_series():
    assert detect_uptrend_stack(falling()) is None


def test_breakout_fires_on_new_high():
    closes = flat(70) + [110.0]
    hit = detect_breakout(closes)
    assert hit is not None and hit.name == "breakout"


def test_breakout_silent_below_prior_high():
    closes = flat(70) + [99.0]
    assert detect_breakout(closes) is None


def test_rsi_recovery_fires_after_oversold_bounce():
    closes = flat(60) + [100.0 - i for i in range(1, 26)] + [75.0 + 2.0 * i for i in range(1, 7)]
    hit = detect_rsi_recovery(closes)
    assert hit is not None and hit.name == "rsi_recovery"


def test_rsi_recovery_silent_without_oversold_dip():
    assert detect_rsi_recovery(rising()) is None


def test_macd_cross_fires_on_fresh_momentum():
    closes = flat(80) + [102.0, 104.0, 106.0, 108.0]
    hit = detect_macd_cross(closes)
    assert hit is not None and hit.name == "macd_cross"


def test_macd_cross_silent_on_flat_series():
    assert detect_macd_cross(flat(100)) is None


def test_higher_lows_fires_on_rising_series():
    hit = detect_higher_lows(rising())
    assert hit is not None and hit.name == "higher_lows"


def test_higher_lows_silent_on_flat_series():
    assert detect_higher_lows(flat()) is None


# --- Probabilistic aggregation --------------------------------------------------------


def test_no_hits_returns_prior():
    assert aggregate_probability([]) == pytest.approx(PRIOR_P_UP, abs=1e-9)


def test_each_hit_raises_the_posterior():
    hit = PatternHit(name="x", hit_rate=0.62, description="")
    p0 = aggregate_probability([])
    p1 = aggregate_probability([hit])
    p2 = aggregate_probability([hit, hit])
    assert p0 < p1 < p2


def test_posterior_is_clamped():
    hits = [PatternHit(name=f"x{i}", hit_rate=0.65, description="") for i in range(50)]
    assert aggregate_probability(hits) <= P_CEIL


def test_sub_half_hit_rate_lowers_the_posterior():
    bear = PatternHit(name="x", hit_rate=0.40, description="")
    assert aggregate_probability([bear]) < PRIOR_P_UP


# --- detect_bullish_patterns (public entry) -------------------------------------------


def test_insufficient_history_returns_prior_without_raising():
    result = detect_bullish_patterns("AAPL", closes=flat(10))
    assert isinstance(result, BullishPatterns)
    assert result.hits == []
    assert result.p_bullish == pytest.approx(PRIOR_P_UP)
    assert any("insufficient" in r for r in result.reasons)


def test_empty_history_never_raises():
    result = detect_bullish_patterns("AAPL", closes=[])
    assert result.p_bullish == pytest.approx(PRIOR_P_UP)


def test_bullish_series_stacks_patterns():
    result = detect_bullish_patterns("NVDA", closes=v_shape_recent_cross(), asof_date="2026-06-11")
    assert result.ticker == "NVDA"
    assert result.asof_date == "2026-06-11"
    assert len(result.hits) >= 2  # at minimum golden cross + uptrend stack
    assert result.p_bullish > PRIOR_P_UP
    assert result.score == pytest.approx(2.0 * result.p_bullish - 1.0)


def test_bearish_series_stays_at_prior():
    result = detect_bullish_patterns("XYZ", closes=falling())
    assert result.hits == []
    assert result.p_bullish == pytest.approx(PRIOR_P_UP)


def test_min_pattern_bars_constant_is_sane():
    assert MIN_PATTERN_BARS >= 50  # SMA50 must be computable


def test_to_dict_roundtrip():
    result = detect_bullish_patterns("AAPL", closes=rising())
    d = result.to_dict()
    assert d["ticker"] == "AAPL"
    assert isinstance(d["patterns"], list)
    assert 0.0 <= d["p_bullish"] <= 1.0
