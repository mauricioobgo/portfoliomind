"""Demo: top-5 most bullish + top-5 most bearish tickers from combined signals.

Operator-facing script. Run manually to eyeball the strategy output::

    uv run python scripts/demo_signals.py
    uv run python scripts/demo_signals.py --universe SPY,QQQ,AAPL,MSFT,TSLA
    uv run python scripts/demo_signals.py --no-cache  # force a fresh fetch

Exits non-zero only on operator-side errors (missing API key, bad CLI
args). The signals themselves are best-effort: a network blip on one
ticker does not break the rest.

This script never logs raw OHLCV (per the card 6 spec). The signal
table is what card 7 (Discord approval) will consume.
"""

from __future__ import annotations

import argparse
import os
import sys

from portfoliomind.config import PortfoliomindConfig
from portfoliomind.logging_setup import get_logger, setup_logging
from portfoliomind.signals import (
    Signal,
    TechnicalCache,
    score_universe,
)
from portfoliomind.time_utils import iso_now
from portfoliomind.universe import UNIVERSE, UNIVERSE_ETFS, UNIVERSE_STOCKS

log = get_logger(__name__)


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="portfoliomind-demo_signals",
        description=(
            "Print the top-5 most-bullish and top-5 most-bearish tickers "
            "from the combined technical + news sentiment signal."
        ),
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="",
        help=(
            "Comma-separated tickers to score. Default: full UNIVERSE "
            "(ETFs + stocks). Pass e.g. 'SPY,QQQ,AAPL,MSFT,TSLA' for a small demo."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Use a one-shot cache file (cleared at start) so the run "
            "always re-fetches and re-scores. Useful for testing changes."
        ),
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
        default=5,
        help="How many top bullish / bearish tickers to print (default: 5).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help=(
            "Filter out signals with confidence < this threshold. "
            "Default: 0.0 (no filter). Use 0.3 to keep only agreeing signals."
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
    """Parse a comma-separated ticker list, or default to the full UNIVERSE."""
    if not arg.strip():
        return UNIVERSE
    parts = [p.strip().upper() for p in arg.split(",") if p.strip()]
    return tuple(parts)


# --- Display helpers -------------------------------------------------------


def _format_signal_table(
    signals: list[Signal],
    *,
    top_n: int,
) -> tuple[list[Signal], list[Signal]]:
    """Return (top_bullish, top_bearish) by ``combined`` score."""
    # Sort by combined desc, then ticker asc for stable display.
    by_combined = sorted(signals, key=lambda s: (-s.combined, s.ticker))
    top_bull = by_combined[:top_n]
    by_combined_asc = sorted(signals, key=lambda s: (s.combined, s.ticker))
    top_bear = by_combined_asc[:top_n]
    return top_bull, top_bear


def _bar(value: float, *, width: int = 20) -> str:
    """Render a small ASCII bar of length ~abs(value) * width."""
    n = int(round(abs(value) * width))
    n = max(0, min(width, n))
    char = "+" if value >= 0 else "-"
    return char * n


def _print_signal(s: Signal) -> None:
    """Print a one-line summary of a single signal."""
    err = f"  [ERROR: {s.error}]" if s.error else ""
    print(
        f"  {s.ticker:<6}  combined={s.combined:+.3f}  "
        f"tech={s.technical:+.3f}  sent={s.sentiment:+.3f}  "
        f"conf={s.confidence:.3f}  {_bar(s.combined)}{err}"
    )


def _print_reasons(s: Signal, *, indent: str = "      ") -> None:
    """Print the human-readable reason list for a signal."""
    for r in s.reasons:
        print(f"{indent}- {r}")


# --- Main ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(args.log_level.upper())

    cfg = PortfoliomindConfig.from_env()
    # The technical half works without an OpenAI key — sentiment
    # silently defaults to 0.0 — but we surface that early so the
    # operator knows what they're getting.
    if not cfg.openai_api_key:
        print(
            "WARNING: OPENAI_API_KEY is not set; sentiment will default to 0.0 "
            "(technical signal only).",
            file=sys.stderr,
        )

    universe = _resolve_universe(args.universe)
    log.info(
        "demo_signals: scoring %d tickers (no_cache=%s, cache_path=%s, "
        "min_confidence=%.2f)",
        len(universe), args.no_cache, args.cache_path, args.min_confidence,
    )

    # Honor the operator's --cache-path by setting the env var that
    # TechnicalCache.from_env reads. This way the demo and the production
    # code path use the same resolution logic.
    os.environ["SIGNALS_CACHE_PATH"] = args.cache_path
    cache = TechnicalCache.from_env()

    if args.no_cache:
        # One-shot path: fresh DB so the yfinance fetch always runs.
        try:
            cache.store._db_path.unlink()  # type: ignore[attr-defined]
        except FileNotFoundError:
            pass
        # Re-open so the new DB is initialized.
        cache = TechnicalCache.from_env()

    try:
        signals = score_universe(
            tickers=universe,
            cache=cache,
            openai_api_key=cfg.openai_api_key or None,
        )
    except Exception as e:
        log.warning("demo_signals: scoring failed: %s", type(e).__name__)
        print(f"ERROR: scoring failed: {e}", file=sys.stderr)
        return 1

    if not signals:
        print("(no signals returned)")
        return 0

    # Optional confidence filter.
    if args.min_confidence > 0.0:
        before = len(signals)
        signals = [s for s in signals if s.confidence >= args.min_confidence]
        log.info(
            "demo_signals: confidence filter %.2f removed %d/%d signals",
            args.min_confidence, before - len(signals), before,
        )

    top_bull, top_bear = _format_signal_table(signals, top_n=args.top_n)

    print(f"PortfolioMind combined signals — {iso_now()}")
    print(
        f"Universe: {len(universe)} tickers "
        f"({len(UNIVERSE_ETFS)} ETFs + {len(UNIVERSE_STOCKS)} stocks)"
    )
    print(f"Cache: {args.cache_path}  (technical + sentiment share one file)")
    print(
        f"Weights: technical={0.6:.1f} | sentiment={0.4:.1f} | "
        f"confidence = |combined| * (1 - |tech - sent|)"
    )
    print()

    print(f"Top {args.top_n} MOST BULLISH:")
    for s in top_bull:
        _print_signal(s)
        _print_reasons(s)
    print()

    print(f"Top {args.top_n} MOST BEARISH:")
    for s in top_bear:
        _print_signal(s)
        _print_reasons(s)
    print()

    errored = sum(1 for s in signals if s.error)
    high_conf = sum(1 for s in signals if s.confidence >= 0.5)
    print(
        f"Total scored: {len(signals)} tickers "
        f"({errored} with errors, {high_conf} high-confidence)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
