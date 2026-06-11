#!/usr/bin/env python
"""Backtest the bullish-pattern strategy.

Walk-forward replays each ticker's historical closes through the same
pattern detector and vol-anchored stops the live strategy uses, and
reports the realized win rate, expectancy, and — most importantly —
the calibration gap between the model's claimed ``p_bullish`` and the
actual win rate.

Usage:

    uv run python scripts/backtest.py                 # whole universe, 2y
    uv run python scripts/backtest.py AAPL MSFT NVDA  # specific tickers
    uv run python scripts/backtest.py --period 5y --max-hold 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfoliomind.backtest import backtest_ticker, backtest_universe
from portfoliomind.logging_setup import get_logger, setup_logging
from portfoliomind.universe import UNIVERSE

log = get_logger("scripts.backtest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Tickers to backtest (default: whole universe)")
    parser.add_argument("--period", default="2y", help="yfinance history window (default 2y)")
    parser.add_argument("--max-hold", type=int, default=20, help="Max bars to hold a position")
    parser.add_argument("--entry-p", type=float, default=0.55, help="p_bullish entry threshold")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    tickers = tuple(t.upper() for t in args.tickers) or UNIVERSE

    if len(tickers) == 1:
        res = backtest_ticker(
            tickers[0], period=args.period, max_hold=args.max_hold, entry_p_threshold=args.entry_p
        )
        print(res.summary_line())
        for name, (cnt, rate) in sorted(res.per_pattern.items()):
            print(f"  {name:16s} {cnt:3d} trades  win_rate={rate:.0%}")
        return 0

    sweep = backtest_universe(
        tickers, period=args.period, max_hold=args.max_hold, entry_p_threshold=args.entry_p
    )
    print(
        f"\n=== Universe backtest: {len(sweep.results)} tickers, "
        f"{sweep.n_trades} pooled trades ==="
    )
    print(
        f"win_rate={sweep.win_rate:.1%}  avg_expectancy={sweep.avg_expectancy:+.2%}  "
        f"calibration_gap={sweep.avg_calibration_gap:+.1%}\n"
    )
    for ticker, res in sorted(
        sweep.results.items(), key=lambda kv: kv[1].total_return, reverse=True
    ):
        print(res.summary_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
