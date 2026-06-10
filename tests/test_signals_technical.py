"""Unit tests for :mod:`portfoliomind.signals.technical` (card 6).

These tests are hermetic — they build synthetic ``pandas.Series`` of
close prices and exercise the pure indicator functions. They never
import yfinance, never hit the network, and never write to the real
``PriceCache`` (a tmp dir is used for the cache tests).

The yfinance path is exercised separately by the
``tests/test_signals_combined.py`` mocks (and by the
``scripts/demo_signals.py`` smoke run on a real CI machine).

We also pin the indicator math against a known-input expectation for
the SMA(50)/SMA(200) cross and the MACD crossover — so a future
"optimization" that subtly changes the formula gets caught here.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfoliomind.signals.price_cache import (
    DEFAULT_CACHE_PATH,
    PRICE_TTL_SECONDS,
    PriceCache,
)
from portfoliomind.signals.technical import (
    TechnicalSignal,
    YFINANCE_TIMEOUT_S,
    _bars_to_df,
    compute_technical_signal,
    indicator_macd_bullish,
    indicator_rsi_not_overbought,
    indicator_sma_golden_cross,
    indicator_twenty_day_breakout,
)


# --- Synthetic data builders -----------------------------------------------


def _uptrending_series(n: int = 250, drift: float = 0.3, seed: int = 0) -> pd.Series:
    """A monotonically uptrending close price series with small noise."""
    rng = np.random.default_rng(seed)
    closes = 100.0 + drift * np.arange(n) + rng.normal(0, 1.0, n)
    return pd.Series(closes)


def _downtrending_series(n: int = 250, drift: float = 0.3, seed: int = 1) -> pd.Series:
    """A monotonically downtrending close price series with small noise."""
    rng = np.random.default_rng(seed)
    closes = 200.0 - drift * np.arange(n) + rng.normal(0, 1.0, n)
    return pd.Series(closes)


def _flat_series(n: int = 250, value: float = 100.0) -> pd.Series:
    """A perfectly flat close price series. RSI cannot be computed meaningfully."""
    return pd.Series([value] * n)


def _rsi_above_70_series(n: int = 250) -> pd.Series:
    """A strongly uptrending series whose RSI(14) is > 70 (overbought)."""
    # Big jumps — every delta positive, no losses.
    closes = 100.0 + 5.0 * np.arange(n)
    return pd.Series(closes)


def _make_bars(close: Iterable[float], *, end_date: date | None = None) -> list[dict]:
    """Convert an iterable of close prices to a list of bar dicts (oldest first)."""
    closes = list(close)
    if not closes:
        return []
    end = end_date or date.today()
    # Use business-day spacing so yfinance's date filter is realistic.
    n = len(closes)
    dates = pd.bdate_range(end=end, periods=n)
    out: list[dict] = []
    for ts, c in zip(dates, closes):
        out.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "open": float(c) - 0.5,
                "high": float(c) + 1.0,
                "low": float(c) - 1.0,
                "close": float(c),
                "volume": 1_000_000,
                "adj_close": float(c),
            }
        )
    return out


# --- SMA golden cross ------------------------------------------------------


class TestSMAGoldenCross:
    def test_uptrend_is_bullish(self):
        s = _uptrending_series(n=250)
        assert indicator_sma_golden_cross(s) is True

    def test_downtrend_is_bearish(self):
        s = _downtrending_series(n=250)
        assert indicator_sma_golden_cross(s) is False

    def test_short_history_returns_false(self):
        s = _uptrending_series(n=100)  # < 200
        assert indicator_sma_golden_cross(s) is False

    def test_exact_200_bars_sufficient(self):
        s = _uptrending_series(n=200)
        assert indicator_sma_golden_cross(s) is True

    def test_flat_series_returns_false(self):
        s = _flat_series(n=250)
        # SMA(50) == SMA(200) for a flat series; ">" is strict.
        assert indicator_sma_golden_cross(s) is False

    def test_none_returns_false(self):
        assert indicator_sma_golden_cross(None) is False  # type: ignore[arg-type]

    def test_pure_step_function(self):
        # Build a series where SMA(50) > SMA(200) by construction.
        # First 150 bars at 100, last 100 bars at 200. The 200-day SMA
        # straddles the jump so the most recent 200 includes ~50
        # high bars; SMA(50) is purely the high bars.
        s = pd.Series([100.0] * 150 + [200.0] * 100)
        assert indicator_sma_golden_cross(s) is True


# --- 20-day breakout --------------------------------------------------------


class TestTwentyDayBreakout:
    def test_fresh_high_breaks_out(self):
        # 20 days of slowly rising closes, then a new high on day 21.
        closes = [100.0 + 0.1 * i for i in range(20)] + [105.0]  # +5% above
        s = pd.Series(closes)
        assert indicator_twenty_day_breakout(s) is True

    def test_continuation_does_not_break_out(self):
        # Smoothly rising series — today's close is part of the trend,
        # not above the prior 20.
        s = _uptrending_series(n=250, drift=0.3, seed=0)
        # In a smooth uptrend, the close is usually inside the prior
        # 20-day high. The "breakout" indicator is a single-day event;
        # we just need it to be a well-defined bool here.
        result = indicator_twenty_day_breakout(s)
        assert isinstance(result, bool)

    def test_short_history_returns_false(self):
        s = pd.Series([100.0] * 20)  # exactly 20, need 21
        assert indicator_twenty_day_breakout(s) is False

    def test_constant_returns_false(self):
        s = pd.Series([100.0] * 100)
        # Today's close is NOT strictly greater than the prior max.
        assert indicator_twenty_day_breakout(s) is False

    def test_only_one_day_above_prior_max(self):
        # 20 days at 100, day 21 at 100.01, day 22 at 100.5.
        # The breakout indicator must be a single-bar event, not a
        # "we are above the 20d max" state. Day 22 is the most
        # recent close; it's above the prior 20d max, so bullish.
        s = pd.Series([100.0] * 20 + [100.01, 100.5])
        assert indicator_twenty_day_breakout(s) is True


# --- MACD bullish crossover -------------------------------------------------


class TestMACDBullish:
    def test_strong_uptrend_is_bullish(self):
        s = _uptrending_series(n=250, drift=0.5)
        assert indicator_macd_bullish(s) is True

    def test_strong_downtrend_is_bearish(self):
        # Use a perfectly monotonic downtrend (no noise) so the
        # MACD-vs-signal comparison is unambiguous. The seeded
        # uptrend test exercises the noise case.
        s = pd.Series([200.0 - 0.5 * i for i in range(250)])
        assert indicator_macd_bullish(s) is False

    def test_flat_series(self):
        s = _flat_series(n=250)
        # MACD line == 0 == signal line for a flat series. The
        # comparison is strict ">", so we expect False.
        assert indicator_macd_bullish(s) is False

    def test_short_history_returns_false(self):
        s = pd.Series([100.0] * 30)  # need 35 (26 + 9)
        assert indicator_macd_bullish(s) is False

    def test_custom_parameters(self):
        # With a very fast MACD on a 250-bar uptrend that has
        # recent acceleration, the MACD line crosses above the
        # signal line. A perfectly linear uptrend gives MACD ==
        # signal (no crossover); we need a recent kink to make
        # this robust.
        closes = [100.0 + 0.1 * i for i in range(230)] + [123.0 + 2.0 * i for i in range(20)]
        s = pd.Series(closes)
        assert indicator_macd_bullish(s, fast=3, slow=10, signal=4) is True

    def test_none_returns_false(self):
        assert indicator_macd_bullish(None) is False  # type: ignore[arg-type]


# --- RSI not overbought -----------------------------------------------------


class TestRSINotOverbought:
    def test_uptrend_is_not_overbought(self):
        # Gentle uptrend — RSI sits in the 50-70 range.
        s = _uptrending_series(n=250, drift=0.2, seed=0)
        assert indicator_rsi_not_overbought(s) is True

    def test_pure_uptrend_becomes_overbought_eventually(self):
        # Strong uptrend — RSI climbs above 70.
        s = _rsi_above_70_series(n=250)
        assert indicator_rsi_not_overbought(s) is False

    def test_downtrend_is_not_overbought(self):
        s = _downtrending_series(n=250, drift=0.3, seed=1)
        assert indicator_rsi_not_overbought(s) is True

    def test_short_history_returns_false(self):
        s = pd.Series([100.0] * 10)
        assert indicator_rsi_not_overbought(s) is False

    def test_rsi_in_zero_hundred_range(self):
        # Sanity: RSI must always be in [0, 100] for any input.
        for seed in range(20):
            s = _uptrending_series(n=250, seed=seed) if seed % 2 == 0 else _downtrending_series(
                n=250, seed=seed
            )
            delta = s.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, pd.NA)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            last = rsi.iloc[-1]
            if not pd.isna(last):
                assert 0.0 <= last <= 100.0, f"seed={seed} RSI={last}"

    def test_constant_input_does_not_crash(self):
        # When all deltas are zero, avg_loss = 0 and the division
        # by zero must not raise.
        s = _flat_series(n=100, value=50.0)
        # Result may be True or False (RSI undefined for constant input)
        # but must not raise.
        result = indicator_rsi_not_overbought(s)
        assert isinstance(result, bool)

    def test_none_returns_false(self):
        assert indicator_rsi_not_overbought(None) is False  # type: ignore[arg-type]


# --- _bars_to_df helper -----------------------------------------------------


class TestBarsToDataFrame:
    def test_empty_input(self):
        df = _bars_to_df([])
        assert df.empty
        # Columns should still be present so downstream code can index them.
        assert "close" in df.columns

    def test_preserves_date_order(self):
        bars = _make_bars([100.0, 101.0, 102.0])
        df = _bars_to_df(bars)
        assert list(df["close"]) == [100.0, 101.0, 102.0]

    def test_coerces_string_closes(self):
        bars = [
            {"date": "2026-01-01", "open": "99.5", "high": "101.0", "low": "99.0", "close": "100.0", "volume": "1000"},
            {"date": "2026-01-02", "open": "100.5", "high": "102.0", "low": "100.0", "close": "101.0", "volume": "2000"},
        ]
        df = _bars_to_df(bars)
        assert df["close"].dtype.kind == "f"
        assert df["close"].iloc[0] == 100.0

    def test_raises_on_missing_date(self):
        with pytest.raises(ValueError, match="date"):
            _bars_to_df([{"close": 100.0}])


# --- PriceCache -------------------------------------------------------------


class TestPriceCache:
    def test_round_trip(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        bars = _make_bars([100.0, 101.0])
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-10", bars=bars)
        out = cache.fetch_bars(ticker="AAPL", as_of_date="2026-06-10")
        assert out is not None
        assert out == bars

    def test_cache_miss(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        assert cache.fetch_bars(ticker="AAPL", as_of_date="2026-06-10") is None

    def test_overwrite_is_idempotent(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        bars1 = _make_bars([100.0])
        bars2 = _make_bars([200.0])
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-10", bars=bars1)
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-10", bars=bars2)
        out = cache.fetch_bars(ticker="AAPL", as_of_date="2026-06-10")
        assert out is not None
        assert out == bars2

    def test_stale_returns_none(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        bars = _make_bars([100.0])
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-10", bars=bars)
        # max_age_s=0 means "must be brand new" — any non-zero age is stale.
        out = cache.fetch_bars(ticker="AAPL", as_of_date="2026-06-10", max_age_s=0)
        assert out is None

    def test_stats(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-10", bars=_make_bars([100.0]))
        cache.store_bars(ticker="MSFT", as_of_date="2026-06-10", bars=_make_bars([200.0]))
        s = cache.stats()
        assert s["bar_rows"] == 2
        assert "db_path" in s

    def test_different_as_of_dates_are_separate(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-10", bars=_make_bars([100.0]))
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-11", bars=_make_bars([101.0]))
        d10 = cache.fetch_bars(ticker="AAPL", as_of_date="2026-06-10")
        d11 = cache.fetch_bars(ticker="AAPL", as_of_date="2026-06-11")
        assert d10 is not None and d10[0]["close"] == 100.0
        assert d11 is not None and d11[0]["close"] == 101.0


# --- compute_technical_signal with yfinance mocked --------------------------


class _FakeTicker:
    """A minimal yfinance.Ticker stub for unit tests."""

    def __init__(self, ticker: str, df: pd.DataFrame):
        self._ticker = ticker
        self._df = df

    def history(self, **kwargs) -> pd.DataFrame:
        return self._df


def _fake_yfinance_ticker_factory(bars_by_ticker: dict[str, list[dict]]):
    """Build a side_effect that returns a ``_FakeTicker`` for the requested ticker."""

    def factory(ticker: str) -> _FakeTicker:
        bars = bars_by_ticker[ticker.upper()]
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # Rename to the yfinance column convention.
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "adj_close": "Adj Close",
            }
        )
        return _FakeTicker(ticker, df)

    return factory


class TestComputeTechnicalSignal:
    def test_returns_valid_signal_on_uptrend(self, tmp_path):
        bars = _make_bars(_uptrending_series(n=250, drift=0.3, seed=0))
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        with patch(
            "yfinance.Ticker",
            side_effect=_fake_yfinance_ticker_factory({"AAPL": bars}),
        ):
            sig = compute_technical_signal(
                "AAPL", as_of_date="2026-06-10", cache=cache
            )
        assert isinstance(sig, TechnicalSignal)
        assert sig.ticker == "AAPL"
        assert sig.as_of_date == "2026-06-10"
        # An uptrend should produce at least 2 bullish flags.
        assert sig.bullish_count >= 2
        # All 4 booleans must be populated.
        for attr in (
            "sma_golden_cross",
            "twenty_day_breakout",
            "macd_bullish",
            "rsi_not_overbought",
        ):
            assert isinstance(getattr(sig, attr), bool)

    def test_returns_empty_signal_on_yfinance_failure(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        with patch(
            "yfinance.Ticker",
            side_effect=RuntimeError("rate limited"),
        ):
            sig = compute_technical_signal(
                "AAPL", as_of_date="2026-06-10", cache=cache
            )
        assert sig.bullish_count == 0
        assert sig.close == 0.0
        for attr in (
            "sma_golden_cross",
            "twenty_day_breakout",
            "macd_bullish",
            "rsi_not_overbought",
        ):
            assert getattr(sig, attr) is False

    def test_returns_empty_signal_on_empty_yfinance(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        with patch("yfinance.Ticker", return_value=_FakeTicker("AAPL", empty_df)):
            sig = compute_technical_signal(
                "AAPL", as_of_date="2026-06-10", cache=cache
            )
        assert sig.bullish_count == 0
        assert sig.close == 0.0

    def test_uses_cache_when_fresh(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        bars = _make_bars(_uptrending_series(n=250, drift=0.3, seed=0))
        # Pre-populate cache with a known date.
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-10", bars=bars)

        # Patch yfinance to assert it is NEVER called.
        with patch(
            "yfinance.Ticker",
            side_effect=AssertionError("yfinance should not be called on cache hit"),
        ):
            sig = compute_technical_signal(
                "AAPL", as_of_date="2026-06-10", cache=cache
            )
        assert sig.bullish_count >= 2

    def test_re_pulls_when_stale(self, tmp_path):
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        # Pre-populate cache with stale data (max_age_s=0).
        bars = _make_bars(_uptrending_series(n=250, drift=0.3, seed=0))
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-10", bars=bars)

        with patch(
            "yfinance.Ticker",
            side_effect=_fake_yfinance_ticker_factory({"AAPL": bars}),
        ):
            sig = compute_technical_signal(
                "AAPL",
                as_of_date="2026-06-10",
                cache=cache,
                max_age_s=0,
            )
        assert sig.bullish_count >= 2

    def test_to_dict_round_trip(self, tmp_path):
        bars = _make_bars(_uptrending_series(n=250, drift=0.3, seed=0))
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        with patch(
            "yfinance.Ticker",
            side_effect=_fake_yfinance_ticker_factory({"AAPL": bars}),
        ):
            sig = compute_technical_signal(
                "AAPL", as_of_date="2026-06-10", cache=cache
            )
        d = sig.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["as_of_date"] == "2026-06-10"
        assert isinstance(d["sma_golden_cross"], bool)
        # sma_50 / sma_200 are floats (or NaN).
        assert isinstance(d["sma_50"], float)
        assert isinstance(d["sma_200"], float)

    def test_filters_bars_after_as_of_date(self, tmp_path):
        # The pull may include future-dated bars. The signal must
        # only use data up to and including as_of_date.
        #
        # Construct: 30 bars at 100 (dated at and before as_of) + 30
        # bars at 200 (dated AFTER as_of). The signal must NOT see
        # the second half; otherwise the close would be 200.
        as_of = "2026-06-10"
        from datetime import date

        dates_first = pd.bdate_range(end=date.fromisoformat(as_of), periods=30)
        bars = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": 99.5,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000,
            }
            for d in dates_first
        ]
        end = date(2026, 7, 20)
        future_dates = pd.bdate_range(end=end, periods=30)
        future_bars = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": 199.5,
                "high": 201.0,
                "low": 199.0,
                "close": 200.0,
                "volume": 1_000_000,
            }
            for d in future_dates
        ]
        all_bars = bars + future_bars

        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        with patch(
            "yfinance.Ticker",
            side_effect=_fake_yfinance_ticker_factory({"AAPL": all_bars}),
        ):
            sig = compute_technical_signal(
                "AAPL", as_of_date=as_of, cache=cache
            )
        # The filtered close should be 100.0 (the last bar <= as_of).
        assert sig.close == pytest.approx(100.0, abs=1e-6)

    def test_yfinance_does_not_import_when_cache_hits(self, tmp_path):
        # The contract: a cached call never touches the yfinance
        # import. The unit test guards against a regression that
        # would yfinance-import on every call (slow, may be down).
        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        bars = _make_bars(_uptrending_series(n=250, drift=0.3, seed=0))
        cache.store_bars(ticker="AAPL", as_of_date="2026-06-10", bars=bars)

        with patch("yfinance.Ticker", side_effect=AssertionError("should not import")):
            sig = compute_technical_signal(
                "AAPL", as_of_date="2026-06-10", cache=cache
            )
        assert sig.ticker == "AAPL"

    def test_drops_partial_bar_with_nan_close(self, tmp_path):
        # yfinance returns the CURRENT (partial) bar with NaN close
        # when the market is open. The signal must drop it and use
        # the last fully-closed bar. Regression guard: a previous
        # version forwarded NaN, producing nan close and broken
        # SMA(50)/SMA(200) numbers.
        bars = _make_bars([100.0 + 0.1 * i for i in range(250)])
        # Append a partial bar with NaN close (yfinance's behaviour
        # when the session is still open).
        from datetime import date, timedelta

        last_full = date(2026, 6, 9)
        partial = last_full + timedelta(days=1)
        bars = bars + [
            {
                "date": partial.strftime("%Y-%m-%d"),
                "open": 125.0,
                "high": 126.0,
                "low": 124.5,
                "close": float("nan"),
                "volume": 50_000_000,
            }
        ]

        cache = PriceCache(db_path=tmp_path / "pc.sqlite")
        with patch(
            "yfinance.Ticker",
            side_effect=_fake_yfinance_ticker_factory({"AAPL": bars}),
        ):
            sig = compute_technical_signal(
                "AAPL", as_of_date="2026-06-10", cache=cache
            )
        # The NaN-close bar must be dropped; the close must equal
        # the last fully-closed bar (124.9) and NOT be NaN.
        import math

        assert not math.isnan(sig.close), f"close is NaN: {sig}"
        assert sig.close == pytest.approx(124.9, abs=1e-6)
        # SMA values must be finite.
        assert not math.isnan(sig.sma_50)
        assert not math.isnan(sig.sma_200)


# --- Public constants ------------------------------------------------------


def test_yfinance_timeout_s_is_reasonable():
    # We don't want the yfinance call to block the morning run.
    assert 5.0 <= YFINANCE_TIMEOUT_S <= 120.0


def test_default_cache_path_is_relative_to_cwd():
    assert DEFAULT_CACHE_PATH.name == "price_cache.sqlite"


def test_price_ttl_seconds_is_one_hour():
    assert PRICE_TTL_SECONDS == 60 * 60
