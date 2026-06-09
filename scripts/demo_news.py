"""Demo: top-3 most-positive and top-3 most-negative tickers from news.

Operator-facing script. Run manually to eyeball the sentiment output::

    uv run python scripts/demo_news.py
    uv run python scripts/demo_news.py --since-hours 48
    uv run python scripts/demo_news.py --universe SPY,QQQ,AAPL,MSFT
    uv run python scripts/demo_news.py --no-cache  # force a fresh fetch

Exits non-zero if the LLM call fails irrecoverably; exits 0 with an
empty output if there is simply no news for the supplied window.

This script never logs raw headline text. The score in ``[-1, +1]``
is what the strategy engine will consume in card 6.
"""

from __future__ import annotations

import argparse
import sys

from portfoliomind.config import PortfoliomindConfig
from portfoliomind.logging_setup import get_logger, setup_logging
from portfoliomind.news import NewsCache, score_universe_sentiment
from portfoliomind.time_utils import iso_now
from portfoliomind.universe import UNIVERSE, UNIVERSE_ETFS, UNIVERSE_STOCKS

log = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="portfoliomind-demo_news",
        description=(
            "Print the top-3 most-positive and top-3 most-negative tickers "
            "from the last N hours of news."
        ),
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=24,
        help="How many hours back to look (default: 24).",
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
        "--no-cache",
        action="store_true",
        help="Use a one-shot cache file (cleared at start) so the run "
        "always fetches and re-scores. Useful for testing changes.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=".cache/news_cache.sqlite",
        help="Override the SQLite cache path. Default: .cache/news_cache.sqlite",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="How many top positive / negative tickers to print (default: 3).",
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
    # Preserve user order, but warn on unknowns.
    return tuple(parts)


def _format_score_table(
    scores: dict[str, float],
    *,
    top_n: int,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return (top_positive, top_negative) lists of (ticker, score)."""
    items = [(t, s) for t, s in scores.items()]
    # Sort by score desc, then ticker asc for stable display.
    items.sort(key=lambda x: (-x[1], x[0]))
    top_pos = items[:top_n]
    # Sort by score asc, then ticker asc.
    items.sort(key=lambda x: (x[1], x[0]))
    top_neg = items[:top_n]
    return top_pos, top_neg


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(args.log_level.upper())

    # Validate API key early so the operator sees a clear error before
    # we hit the network.
    cfg = PortfoliomindConfig.from_env()
    if not cfg.openai_api_key:
        print("ERROR: OPENAI_API_KEY is not set; cannot score sentiment.", file=sys.stderr)
        return 2

    universe = _resolve_universe(args.universe)
    log.info(
        "demo: scoring %d tickers (since_hours=%d, no_cache=%s, cache_path=%s)",
        len(universe), args.since_hours, args.no_cache, args.cache_path,
    )

    cache = NewsCache(db_path=args.cache_path)
    if args.no_cache:
        # One-shot path: fresh DB so the fetch + LLM call always run.
        try:
            cache._db_path.unlink()  # type: ignore[attr-defined]
        except FileNotFoundError:
            pass
        cache = NewsCache(db_path=args.cache_path)

    try:
        scores = score_universe_sentiment(
            tickers=universe,
            since_hours=args.since_hours,
            api_key=cfg.openai_api_key,
            cache=cache,
        )
    except Exception as e:
        log.warning("demo: scoring failed: %s", type(e).__name__)
        print(f"ERROR: scoring failed: {e}", file=sys.stderr)
        return 1

    if not scores:
        print("(no scores returned)")
        return 0

    top_pos, top_neg = _format_score_table(scores, top_n=args.top_n)

    print(f"PortfolioMind news sentiment — {iso_now()}")
    print(f"Window: last {args.since_hours}h | Universe: {len(universe)} tickers "
          f"({len(UNIVERSE_ETFS)} ETFs + {len(UNIVERSE_STOCKS)} stocks)")
    print(f"Cache: {args.cache_path}")
    print()
    print(f"Top {args.top_n} MOST POSITIVE:")
    for ticker, score in top_pos:
        bar = "+" * int(round(abs(score) * 20))
        print(f"  {ticker:<6}  {score:+.3f}  {bar}")
    print()
    print(f"Top {args.top_n} MOST NEGATIVE:")
    for ticker, score in top_neg:
        bar = "-" * int(round(abs(score) * 20))
        print(f"  {ticker:<6}  {score:+.3f}  {bar}")
    print()
    print(f"Total scored: {len(scores)} tickers (0.0 means no headlines matched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
