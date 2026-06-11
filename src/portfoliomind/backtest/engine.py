"""Walk-forward backtesting for the bullish-pattern strategy (card 10).

The backtest replays a ticker's historical closes bar-by-bar through
the SAME pattern detector and vol-anchored stop logic the live
strategy uses, and measures whether the probabilistic model actually
pays off out of sample.

Why this exists
---------------
The whole strategy rests on ``p_bullish`` — a *claimed* probability of
upside aggregated from pattern hit-rate priors. A backtest is the only
honest way to check that claim. The headline output is the
**calibration gap**: ``avg_p_bullish - realized_win_rate``. A gap near
zero means the priors are well calibrated; a large positive gap means
the model is overconfident and the priors in
:mod:`portfoliomind.signals.patterns` should be lowered.

Method (deliberately conservative, closes-only)
-----------------------------------------------
* At each bar ``i`` we look only at ``closes[:i+1]`` — no look-ahead.
* When :func:`detect_bullish_patterns` reports
  ``p_bullish >= entry_p_threshold`` and we are flat, we enter a long
  at ``closes[i]``.
* The stop/target mirror the live sizer exactly:
  ``stop_pct = clamp(STOP_SIGMAS * vol_20d, MIN_STOP_PCT, MAX_STOP_PCT)``,
  ``sl = entry*(1-stop_pct)``, ``tp = entry*(1+reward_risk*stop_pct)``.
* We walk forward up to ``max_hold`` bars. The first close at/below the
  stop exits as a loss; the first close at/above the target exits as a
  win; otherwise we exit at the horizon close. We only ever hold one
  position at a time (a cash account, no pyramiding).
* Closes-only means we cannot see intrabar highs/lows, so a gap through
  the stop is booked at the *actual* close (which can be worse than the
  stop) — pessimistic on purpose, never optimistic.

Everything is pure Python over a list of closes. The only I/O is the
optional :func:`fetch` (yfinance), which is injected in tests. Nothing
here ever raises on well-formed numeric input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from ..logging_setup import get_logger
from ..signals.patterns import MIN_PATTERN_BARS, detect_bullish_patterns
from ..signals.sizer import MAX_STOP_PCT, MIN_STOP_PCT, REWARD_RISK, STOP_SIGMAS
from ..signals.technicals import VOL_SHORT, fetch_ohlcv, realized_vol
from ..universe import UNIVERSE

log = get_logger(__name__)


#: Default probability gate for a simulated entry — the same threshold
#: the live combiner uses to qualify a candidate.
DEFAULT_ENTRY_P: float = 0.55
#: Default max bars to hold a simulated position before exiting at close.
DEFAULT_MAX_HOLD: int = 20
#: Default lookback window pulled for a backtest. 2 years of daily bars
#: gives ~500 bars — enough setups to be meaningful without overfitting.
DEFAULT_PERIOD: str = "2y"


@dataclass(frozen=True)
class BacktestTrade:
    """One simulated round-trip."""

    entry_index: int
    entry_price: float
    exit_index: int
    exit_price: float
    exit_reason: str  # "tp" | "sl" | "timeout"
    ret: float  # fractional return (exit/entry - 1)
    holding_bars: int
    p_bullish: float
    patterns: tuple[str, ...]

    @property
    def won(self) -> bool:
        return self.ret > 0


@dataclass(frozen=True)
class BacktestResult:
    """Aggregate stats for one ticker's walk-forward backtest.

    The fields the strategy cares about most are ``win_rate``,
    ``expectancy`` (average per-trade return), and ``calibration_gap``
    (how far the model's claimed probability sat above the realized
    win rate).
    """

    ticker: str
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    total_return: float = 0.0
    expectancy: float = 0.0
    profit_factor: Optional[float] = None
    max_drawdown: float = 0.0
    avg_p_bullish: float = 0.0
    calibration_gap: float = 0.0
    per_pattern: dict[str, tuple[int, float]] = field(default_factory=dict)
    trades: list[BacktestTrade] = field(default_factory=list)
    bars: int = 0
    note: str = ""

    def supported(self, *, min_trades: int = 5) -> bool:
        """True when the backtest gives a usable, positive-edge signal."""
        return self.n_trades >= min_trades and self.expectancy > 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "win_rate": round(self.win_rate, 4),
            "avg_return": round(self.avg_return, 4),
            "total_return": round(self.total_return, 4),
            "expectancy": round(self.expectancy, 4),
            "profit_factor": (
                None if self.profit_factor is None else round(self.profit_factor, 3)
            ),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_p_bullish": round(self.avg_p_bullish, 4),
            "calibration_gap": round(self.calibration_gap, 4),
            "per_pattern": {k: [n, round(w, 4)] for k, (n, w) in self.per_pattern.items()},
            "bars": self.bars,
            "note": self.note,
        }

    def summary_line(self) -> str:
        if self.n_trades == 0:
            return f"backtest {self.ticker}: no setups ({self.bars} bars) — {self.note}".strip()
        return (
            f"backtest {self.ticker}: {self.n_trades} trades, "
            f"win_rate={self.win_rate:.0%}, expectancy={self.expectancy:+.2%}, "
            f"total={self.total_return:+.1%}, maxDD={self.max_drawdown:.1%}, "
            f"calibration_gap={self.calibration_gap:+.0%} "
            f"(claimed p={self.avg_p_bullish:.0%})"
        )


# --- Core engine --------------------------------------------------------------


def _stop_pct(closes_window: list[float], *, stop_sigmas: float) -> float:
    """Vol-anchored stop fraction, mirroring the live sizer."""
    vol = realized_vol(closes_window, VOL_SHORT)
    if vol is None or vol <= 0 or not math.isfinite(vol):
        return (MIN_STOP_PCT + MAX_STOP_PCT) / 2.0
    return min(max(stop_sigmas * vol, MIN_STOP_PCT), MAX_STOP_PCT)


def backtest_closes(
    ticker: str,
    closes: list[float],
    *,
    entry_p_threshold: float = DEFAULT_ENTRY_P,
    reward_risk: float = REWARD_RISK,
    max_hold: int = DEFAULT_MAX_HOLD,
    stop_sigmas: float = STOP_SIGMAS,
) -> BacktestResult:
    """Walk ``closes`` and simulate the bullish-pattern strategy.

    Pure: no I/O, never raises on a well-formed float list. Returns a
    zero-trade :class:`BacktestResult` (with a ``note``) when the
    history is too short to evaluate.
    """
    ticker = ticker.upper()
    n = len(closes)
    if n < MIN_PATTERN_BARS + 2:
        return BacktestResult(
            ticker=ticker,
            bars=n,
            note=f"insufficient history (need ≥ {MIN_PATTERN_BARS + 2} closes, got {n})",
        )

    trades: list[BacktestTrade] = []
    i = MIN_PATTERN_BARS
    while i < n - 1:
        window = closes[: i + 1]
        patterns = detect_bullish_patterns(ticker, closes=window)
        if patterns.p_bullish < entry_p_threshold or not patterns.hits:
            i += 1
            continue

        entry = closes[i]
        if entry <= 0 or not math.isfinite(entry):
            i += 1
            continue
        stop_pct = _stop_pct(window, stop_sigmas=stop_sigmas)
        sl = entry * (1.0 - stop_pct)
        tp = entry * (1.0 + reward_risk * stop_pct)

        exit_index = min(i + max_hold, n - 1)
        exit_price = closes[exit_index]
        exit_reason = "timeout"
        for j in range(i + 1, min(i + max_hold + 1, n)):
            c = closes[j]
            if not math.isfinite(c):
                continue
            if c <= sl:
                exit_index, exit_price, exit_reason = j, c, "sl"
                break
            if c >= tp:
                exit_index, exit_price, exit_reason = j, c, "tp"
                break

        ret = exit_price / entry - 1.0
        trades.append(
            BacktestTrade(
                entry_index=i,
                entry_price=entry,
                exit_index=exit_index,
                exit_price=exit_price,
                exit_reason=exit_reason,
                ret=ret,
                holding_bars=exit_index - i,
                p_bullish=patterns.p_bullish,
                patterns=tuple(h.name for h in patterns.hits),
            )
        )
        # Resume scanning after the position closed (no overlap).
        i = exit_index + 1

    return _aggregate(ticker, closes, trades)


def _aggregate(ticker: str, closes: list[float], trades: list[BacktestTrade]) -> BacktestResult:
    """Fold a trade list into a :class:`BacktestResult`."""
    if not trades:
        return BacktestResult(ticker=ticker, bars=len(closes), note="no qualifying setups")

    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t.won)
    rets = [t.ret for t in trades]
    avg_return = sum(rets) / n_trades

    # Compounded equity curve + max drawdown.
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        equity *= 1.0 + t.ret
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    total_return = equity - 1.0

    gross_win = sum(r for r in rets if r > 0)
    gross_loss = -sum(r for r in rets if r < 0)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None

    win_rate = n_wins / n_trades
    avg_p = sum(t.p_bullish for t in trades) / n_trades

    # Per-pattern realized win rate (which setups actually pay).
    per_pattern: dict[str, list[int]] = {}
    for t in trades:
        for name in t.patterns:
            slot = per_pattern.setdefault(name, [0, 0])  # [count, wins]
            slot[0] += 1
            slot[1] += 1 if t.won else 0
    per_pattern_rate = {
        name: (cnt, wins / cnt if cnt else 0.0) for name, (cnt, wins) in per_pattern.items()
    }

    return BacktestResult(
        ticker=ticker,
        n_trades=n_trades,
        n_wins=n_wins,
        win_rate=win_rate,
        avg_return=avg_return,
        total_return=total_return,
        expectancy=avg_return,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        avg_p_bullish=avg_p,
        calibration_gap=avg_p - win_rate,
        per_pattern=per_pattern_rate,
        trades=trades,
        bars=len(closes),
    )


# --- I/O wrappers (the only network) ------------------------------------------


def backtest_ticker(
    ticker: str,
    *,
    fetch: Optional[Callable[..., list[float]]] = None,
    period: str = DEFAULT_PERIOD,
    entry_p_threshold: float = DEFAULT_ENTRY_P,
    reward_risk: float = REWARD_RISK,
    max_hold: int = DEFAULT_MAX_HOLD,
) -> BacktestResult:
    """Fetch ``period`` of closes for ``ticker`` and backtest them.

    ``fetch`` defaults to :func:`portfoliomind.signals.technicals.fetch_ohlcv`
    and is injected in tests. A fetch failure (empty list) returns a
    zero-trade result with a ``note`` — never raises.
    """
    if fetch is None:
        fetch = fetch_ohlcv
    try:
        closes = fetch(ticker, period=period)
    except TypeError:
        # An injected fetch may not accept the ``period`` kwarg.
        closes = fetch(ticker)
    except Exception as e:  # noqa: BLE001 — a data failure is a no-result, not a crash
        log.warning("backtest: fetch failed for %s: %s", ticker, type(e).__name__)
        return BacktestResult(ticker=ticker.upper(), note=f"fetch failed: {type(e).__name__}")

    return backtest_closes(
        ticker,
        list(closes or []),
        entry_p_threshold=entry_p_threshold,
        reward_risk=reward_risk,
        max_hold=max_hold,
    )


@dataclass(frozen=True)
class UniverseBacktest:
    """Aggregate of a multi-ticker backtest sweep."""

    results: dict[str, BacktestResult]
    n_trades: int
    win_rate: float
    avg_expectancy: float
    avg_calibration_gap: float

    def to_dict(self) -> dict:
        return {
            "tickers": len(self.results),
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_expectancy": round(self.avg_expectancy, 4),
            "avg_calibration_gap": round(self.avg_calibration_gap, 4),
            "per_ticker": {t: r.to_dict() for t, r in self.results.items()},
        }


def backtest_universe(
    tickers: Iterable[str] = UNIVERSE,
    *,
    fetch: Optional[Callable[..., list[float]]] = None,
    period: str = DEFAULT_PERIOD,
    **kwargs,
) -> UniverseBacktest:
    """Backtest every ticker and pool the trades into one calibration read.

    Never raises: a per-ticker failure contributes a zero-trade result.
    """
    results: dict[str, BacktestResult] = {}
    all_trades: list[BacktestTrade] = []
    for t in tickers:
        res = backtest_ticker(t, fetch=fetch, period=period, **kwargs)
        results[t.upper()] = res
        all_trades.extend(res.trades)

    n = len(all_trades)
    if n == 0:
        return UniverseBacktest(results=results, n_trades=0, win_rate=0.0, avg_expectancy=0.0, avg_calibration_gap=0.0)
    wins = sum(1 for t in all_trades if t.won)
    win_rate = wins / n
    avg_exp = sum(t.ret for t in all_trades) / n
    avg_p = sum(t.p_bullish for t in all_trades) / n
    log.info(
        "backtest_universe: %d tickers, %d pooled trades, win_rate=%.0f%%, calibration_gap=%+.0f%%",
        len(results), n, win_rate * 100, (avg_p - win_rate) * 100,
    )
    return UniverseBacktest(
        results=results,
        n_trades=n,
        win_rate=win_rate,
        avg_expectancy=avg_exp,
        avg_calibration_gap=avg_p - win_rate,
    )


__all__ = [
    "DEFAULT_ENTRY_P",
    "DEFAULT_MAX_HOLD",
    "DEFAULT_PERIOD",
    "BacktestTrade",
    "BacktestResult",
    "UniverseBacktest",
    "backtest_closes",
    "backtest_ticker",
    "backtest_universe",
]
