"""Demo: top-N combined bullish+news candidates from the strategy engine.

Operator-facing script. Run manually to eyeball the combined signal::

    uv run python scripts/demo_signals.py
    uv run python scripts/demo_signals.py --top-n 5
    uv run python scripts/demo_signals.py --universe SPY,QQQ,AAPL,MSFT
    uv run python scripts/demo_signals.py --no-cache  # force a fresh price pull
    uv run python scripts/demo_signals.py --no-news   # skip the LLM call

Exits non-zero if a fatal error occurs; exits 0 with an empty list
when no tickers pass the AND-of-two gate (e.g. the LLM returned no
positive news, or yfinance is down for the whole universe).

The script does NOT log raw headline text. The technical + sentiment
numbers it prints are the same shape card 7 (sizer + Discord approval)
will consume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from portfoliomind.config import PortfoliomindConfig
from portfoliomind.logging_setup import get_logger, setup_logging
from portfoliomind.signals import (
    DEFAULT_CACHE_PATH as _DEFAULT_PRICE_CACHE_PATH,
    MIN_NEWS_SENTIMENT,
    MIN_TECHNICAL_BULLISH,
    STRATEGY,
    TIMEFRAME,
    WEIGHT_NEWS,
    WEIGHT_TECHNICAL,
    PriceCache,
    score_universe,
)
from portfoliomind.time_utils import iso_now
from portfoliomind.universe import UNIVERSE, UNIVERSE_ETFS, UNIVERSE_STOCKS

log = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="portfoliomind-demo_signals",
        description=(
            "Print the top-N combined bullish+news candidates. Combines "
            "yfinance technical indicators with the LLM-scored news "
            "sentiment from card 5."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top candidates to print (default: 10).",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="",
        help=(
            "Comma-separated tickers to score. Default: full UNIVERSE "
            "(ETFs + stocks). Pass e.g. 'SPY,QQQ,AAPL,MSFT' for a small demo."
        ),
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=24,
        help="News lookback window in hours (default: 24).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Wipe the price cache file at start so yfinance is re-pulled.",
    )
    parser.add_argument(
        "--no-news",
        action="store_true",
        help=(
            "Skip the LLM sentiment call. Every ticker gets 0.0 sentiment, "
            "so the news gate drops every candidate. Useful for testing "
            "the technical path in isolation or when OPENAI_API_KEY is "
            "missing."
        ),
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=str(_DEFAULT_PRICE_CACHE_PATH),
        help=(
            "Override the SQLite price-cache path. Default: "
            f"{_DEFAULT_PRICE_CACHE_PATH}"
        ),
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default="",
        help=(
            "Override the 'as of' date (YYYY-MM-DD). Default: today in "
            "Bogota. Useful for backtesting on a specific trading day."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING). Default: INFO.",
    )
    return parser.parse_args(argv)


def _resolve_universe(arg: str) -> tuple[str, ...]:
    if not arg.strip():
        return UNIVERSE
    parts = [p.strip().upper() for p in arg.split(",") if p.strip()]
    return tuple(parts)


def _print_candidates(candidates, *, top_n: int) -> None:
    """Print the top-N candidates as a human-readable table."""
    if not candidates:
        print("(no candidates passed the AND-of-two gate)")
        return
    print(
        f"{'rank':>4}  {'ticker':<7} {'combined':>9}  "
        f"{'tech':>5}  {'news':>6}  {'close':>9}  reason"
    )
    print("-" * 100)
    for rank, c in enumerate(candidates[:top_n], start=1):
        print(
            f"{rank:>4}  {c.ticker:<7} {c.combined_score:>9.3f}  "
            f"{c.technical_score:>5.2f}  {c.news_score:>+6.3f}  "
            f"{c.entry_price:>9.2f}  {c.top_signal_reason}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(args.log_level.upper())

    cfg = PortfoliomindConfig.from_env()
    if not args.no_news and not cfg.openai_api_key:
        print("ERROR: OPENAI_API_KEY is not set; pass --no-news to skip the LLM call.", file=sys.stderr)
        return 2

    universe = _resolve_universe(args.universe)
    log.info(
        "demo: scoring %d tickers (top_n=%d, since_hours=%d, no_cache=%s, no_news=%s)",
        len(universe), args.top_n, args.since_hours, args.no_cache, args.no_news,
    )

    cache = PriceCache(db_path=args.cache_path)
    if args.no_cache:
        try:
            cache_path = Path(args.cache_path)
            if cache_path.exists():
                cache_path.unlink()
        except OSError as e:
            log.warning("demo: could not wipe price cache: %s", e)
        cache = PriceCache(db_path=args.cache_path)

    as_of = args.as_of_date.strip() or None
    try:
        candidates = score_universe(
            tickers=universe,
            top_n=args.top_n,
            price_cache=cache,
            news_api_key=None if args.no_news else cfg.openai_api_key,
            since_hours=args.since_hours,
            as_of_date=as_of,
        )
    except Exception as e:
        log.warning("demo: scoring failed: %s", type(e).__name__)
        print(f"ERROR: scoring failed: {e}", file=sys.stderr)
        return 1

    print(f"PortfolioMind combined signals — {iso_now()}")
    print(
        f"Universe: {len(universe)} tickers ({len(UNIVERSE_ETFS)} ETFs + "
        f"{len(UNIVERSE_STOCKS)} stocks)"
    )
    print(
        f"Gate: technical >= {MIN_TECHNICAL_BULLISH}/4 bullish AND "
        f"news > +{MIN_NEWS_SENTIMENT:.1f}"
    )
    print(
        f"Weights: technical={WEIGHT_TECHNICAL:.2f}  news={WEIGHT_NEWS:.2f}  "
        f"strategy={STRATEGY}  timeframe={TIMEFRAME}"
    )
    print(f"Cache: {args.cache_path}")
    print()
    _print_candidates(candidates, top_n=args.top_n)
    print()
    print(
        f"Passed gate: {len(candidates)}/{len(universe)} (top {min(args.top_n, len(candidates))} shown)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
