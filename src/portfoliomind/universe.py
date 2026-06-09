"""Trading universe: ETFs and S&P 500 stocks this agent watches.

Cards 5/6/7/8 all share the same universe — defining it here keeps the
strategy cards honest about what they are scoring.

The ETF list is the spec's literal 15 sector + broad-market + bond ETFs.
The stock list is a reasonable starting S&P 500 top-30 by market cap; the
operator is expected to edit ``UNIVERSE_STOCKS`` later as conviction
changes. We intentionally do NOT make this dynamic (e.g. scraping holdings
from a fund) — the universe is a configuration knob, not data.

The two lists are exposed separately and combined in ``UNIVERSE``. Tests
import the constants directly; scripts iterate over ``UNIVERSE``.

Design notes:

* The lists are tuples (immutable). Mutating them at runtime would be a
  bug — config should be edited at the source, not in-process.
* Order matters: ``UNIVERSE`` is ``UNIVERSE_ETFS + UNIVERSE_STOCKS``.
  Iteration in scoring/budgeting is deterministic.
* ``is_known_ticker`` is provided as a fast membership check for the
  news-match module. It is case-insensitive and also recognizes the dot
  form of Berkshire (BRK.B ↔ BRK-B).
"""

from __future__ import annotations

from typing import Final

# --- ETFs -------------------------------------------------------------------
# Spec: 15 tickers covering broad market, sectors, international, bonds.
# Order: broad-market (3) → sector (10) → international (1) → bonds (1).
UNIVERSE_ETFS: Final[tuple[str, ...]] = (
    # Broad market
    "SPY",   # SPDR S&P 500
    "QQQ",   # Invesco Nasdaq-100
    "IWM",   # iShares Russell 2000
    # Sector SPDRs (alphabetical)
    "XLB",   # Materials
    "XLE",   # Energy
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLK",   # Technology
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
    # International developed
    "EEM",   # Emerging markets
    "VEA",   # Developed markets ex-US
    # Bonds
    "BND",   # Total bond market
)


# --- Stocks -----------------------------------------------------------------
# 30 large-cap S&P 500 names by market cap (approximate, mid-2026 ranking).
# The operator may edit this list at any time; the news and strategy
# modules re-read it on every call. Order is roughly market-cap descending
# so the first entries are the highest-priority scoring targets.
UNIVERSE_STOCKS: Final[tuple[str, ...]] = (
    # Mega-cap tech
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "GOOGL", # Alphabet (Class A)
    "AMZN",  # Amazon
    "META",  # Meta Platforms
    "TSLA",  # Tesla
    # Tech / comms
    "AVGO",  # Broadcom
    "ORCL",  # Oracle
    "CRM",   # Salesforce
    "AMD",   # AMD
    "INTC",  # Intel
    # Consumer
    "WMT",   # Walmart
    "COST",  # Costco
    "HD",    # Home Depot
    "PG",    # Procter & Gamble
    "KO",    # Coca-Cola
    "MCD",   # McDonald's
    "NKE",   # Nike
    # Financials
    "JPM",   # JPMorgan Chase
    "V",     # Visa
    "MA",    # Mastercard
    "BAC",   # Bank of America
    "BRK.B", # Berkshire Hathaway (Class B)
    # Healthcare
    "UNH",   # UnitedHealth
    "LLY",   # Eli Lilly
    "JNJ",   # Johnson & Johnson
    "PFE",   # Pfizer
    # Energy / industrials
    "XOM",   # Exxon Mobil
    "CAT",   # Caterpillar
)


# --- Combined ---------------------------------------------------------------

UNIVERSE: Final[tuple[str, ...]] = UNIVERSE_ETFS + UNIVERSE_STOCKS


# --- Ticker normalization helpers -------------------------------------------

# Berkshire has two common yfinance spellings (BRK.B and BRK-B). The news
# matcher normalizes both to the same internal form so a headline
# containing either form routes to the same ticker.
_BERKSHIRE_ALIASES: Final[frozenset[str]] = frozenset({"BRK.B", "BRK-B", "BRKB"})


def _normalize_ticker(ticker: str) -> str:
    """Return a normalized form for membership comparison.

    Strips whitespace, uppercases, and collapses ``-`` → ``.`` so the
    Berkshire variants fold together. Empty / whitespace-only input
    returns the empty string.
    """
    s = ticker.strip().upper()
    if not s:
        return ""
    return s.replace("-", ".")


def is_known_ticker(ticker: str) -> bool:
    """True when ``ticker`` (case-insensitive, hyphen/dot agnostic) is in the universe."""
    norm = _normalize_ticker(ticker)
    if not norm:
        return False
    if norm in UNIVERSE:
        return True
    # Tickers containing dots or hyphens need an in-tuple scan because
    # we don't pre-normalize the source constants. In practice only
    # BRK.B needs this; the loop is 45 elements either way.
    return any(_normalize_ticker(t) == norm for t in UNIVERSE)


__all__ = [
    "UNIVERSE_ETFS",
    "UNIVERSE_STOCKS",
    "UNIVERSE",
    "is_known_ticker",
    "_normalize_ticker",
    "_BERKSHIRE_ALIASES",
]
