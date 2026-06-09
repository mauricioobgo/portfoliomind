"""Match news headlines to the trading universe.

The matcher is intentionally naive — case-insensitive substring
matching against the ticker list plus a small alias map for the few
companies whose name is more recognizable than their ticker (e.g.
"Apple" → AAPL, "Berkshire" → BRK.B).

Why naive?
* The headline corpus per run is small (3 feeds × ~30 headlines ≈ 100
  items).
* Recursive matching (N-grams, embedding distance, entity linker)
  would add 100× the complexity for negligible accuracy gain on
  financial headlines, where the ticker is usually literally in the
  text.
* The output is consumed by an LLM sentiment scorer, which can
  tolerate a few false positives (a "Google" mention in an AAPL
  story is unlikely; "Apple" in a GOOGL story is rare).

The alias map is hand-curated and small. The operator can extend it
by editing ``_ALIASES`` — there is no auto-discovery, on purpose,
because auto-discovered aliases are a source of false positives
("Bank of America" → BAC, "Bank of New York" → BK).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..logging_setup import get_logger
from ..universe import UNIVERSE, _BERKSHIRE_ALIASES, _normalize_ticker
from .feeds import Headline

log = get_logger(__name__)


# Alias map: lowercase substrings (no spaces around) → canonical ticker.
# Order is irrelevant for substring matching, but keep the dict
# readable. Add new entries by editing this map; the test suite
# (test_news_match.py) covers the obvious ones.
_ALIASES: dict[str, str] = {
    # Tech
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta platforms": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "broadcom": "AVGO",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "intel": "INTC",
    # Consumer
    "walmart": "WMT",
    "costco": "COST",
    "home depot": "HD",
    "procter & gamble": "PG",
    "procter and gamble": "PG",
    "coca-cola": "KO",
    "coca cola": "KO",
    "cocacola": "KO",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "nike": "NKE",
    # Financials
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "visa": "V",
    "mastercard": "MA",
    "bank of america": "BAC",
    "berkshire hathaway": "BRK.B",
    "berkshire": "BRK.B",
    # Healthcare
    "unitedhealth": "UNH",
    "united health": "UNH",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    # Energy / industrials
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "exxon mobil": "XOM",
    "caterpillar": "CAT",
    # ETFs
    "s&p 500": "SPY",
    "sp 500": "SPY",
    "sp500": "SPY",
    "nasdaq 100": "QQQ",
    "nasdaq-100": "QQQ",
    "russell 2000": "IWM",
    "emerging markets": "EEM",
    "developed markets": "VEA",
    "total bond": "BND",
    "bond market": "BND",
    # Sector SPDR common nicknames
    "energy select": "XLE",
    "financial select": "XLF",
    "technology select": "XLK",
    "health care select": "XLV",
    "consumer staples select": "XLP",
    "consumer discretionary select": "XLY",
    "utilities select": "XLU",
    "industrial select": "XLI",
    "materials select": "XLB",
}


def _build_search_terms() -> tuple[dict[str, str], frozenset[str]]:
    """Build the (alias map, direct-ticker set) for the matcher.

    We split the two concerns:

    1. **Alias map** — long-ish substrings (company names, sector
       nicknames) that we match WITHOUT word boundaries because their
       keys are long enough not to collide with common English
       ("apple", "nvidia", "berkshire hathaway", etc.). The exception
       is the bare-ticker entries (``aapl``, ``msft``, ``ma``) which
       we deliberately move to the word-boundary set below.

    2. **Direct-ticker set** — short ticker symbols that we match with
       word boundaries. This prevents ``"ma"`` inside ``"estimates"``
       from spuriously matching Mastercard.

    The matcher scans the alias map as substring (no boundary) and
    the direct-ticker set as word-bounded substring. Both are
    combined into the final set of matched tickers.

    Berkshire dot/hyphen variants (``BRK.B``, ``BRK-B``, ``BRKB``)
    all go through the direct-ticker set so a single boundary-check
    covers them.
    """
    # Alias map: long substrings only. We skip any key that is a bare
    # ticker — those are handled by the word-bounded set below.
    alias_terms: dict[str, str] = {}
    for term, target in _ALIASES.items():
        if term.upper() in UNIVERSE:
            # Bare ticker in the alias map — defer to the word-bounded
            # pass. This keeps "ma" out of the substring matcher.
            continue
        alias_terms[term] = target

    # Direct-ticker set: the full universe + Berkshire variants, all
    # word-bounded.
    direct_tickers: set[str] = set(UNIVERSE)
    direct_tickers.update(_BERKSHIRE_ALIASES)
    return alias_terms, frozenset(direct_tickers)


# Module-level so the dicts are built once per process. Hot path is
# iterated for every headline; rebuilding the dicts per call would be
# wasteful.
_ALIAS_TERMS: dict[str, str]
_DIRECT_TICKERS: frozenset[str]
_ALIAS_TERMS, _DIRECT_TICKERS = _build_search_terms()


def _tickers_in_text(text: str) -> set[str]:
    """Return the set of universe tickers mentioned in ``text``.

    Two passes:

    1. **Alias map** — long substrings (company names, sector nicknames)
       are matched without word boundaries because their keys are long
       enough not to collide with common English. Example: "Apple" in
       "Apple reports earnings".

    2. **Direct-ticker set** — short ticker symbols are matched with
       word boundaries. This prevents "MA" inside "estimates" from
       spuriously matching Mastercard, or "IWM" inside "swimming"
       from matching the Russell 2000 ETF.

    The two passes are combined (deduped) into a single set. Berkshire
    variants (``BRK-B``, ``BRKB``) are normalized to the canonical
    ``BRK.B`` form so downstream code sees a single ticker.
    """
    if not text:
        return set()
    out: set[str] = set()

    # 1) Alias pass: substring, case-insensitive.
    haystack = text.lower()
    for term, ticker in _ALIAS_TERMS.items():
        if term in haystack:
            out.add(ticker)

    # 2) Direct-ticker pass: word-bounded substring on the original
    #    text (uppercased once).
    upper = text.upper()
    for t in _DIRECT_TICKERS:
        if _word_boundary_match(upper, t):
            # Normalize Berkshire variants to the canonical BRK.B.
            normalized = t.replace("-", ".") if t in _BERKSHIRE_ALIASES else t
            out.add(normalized)
    return out


def _word_boundary_match(upper_text: str, upper_term: str) -> bool:
    """Return True if ``upper_term`` appears in ``upper_text`` with non-alnum boundaries."""
    if not upper_term:
        return False
    start = 0
    while True:
        idx = upper_text.find(upper_term, start)
        if idx < 0:
            return False
        # Left boundary: start of string or non-alnum char before.
        left_ok = idx == 0 or not upper_text[idx - 1].isalnum()
        # Right boundary: end of string or non-alnum char after.
        end = idx + len(upper_term)
        right_ok = end >= len(upper_text) or not upper_text[end].isalnum()
        if left_ok and right_ok:
            return True
        start = idx + 1


def match_headlines_to_universe(
    headlines: Iterable[Headline],
    *,
    universe: tuple[str, ...] = UNIVERSE,
) -> dict[str, list[Headline]]:
    """Group ``headlines`` by their matched ticker.

    Returns a dict ``{ticker: [headline, ...]}`` for tickers in
    ``universe`` that have at least one match. Tickers with zero matches
    are omitted (the caller can fill in a 0.0 score for them).
    """
    # We allow the caller to override the universe (handy for tests)
    # but we still index the search terms globally to keep the matching
    # logic deterministic.
    grouped: dict[str, list[Headline]] = defaultdict(list)
    allowed = {t.upper() for t in universe}
    # Pre-normalize the allowed set to handle BRK-B vs BRK.B in the
    # caller's universe.
    allowed_norm = {_normalize_ticker(t) for t in allowed}
    n_seen = 0
    for h in headlines:
        n_seen += 1
        tickers = _tickers_in_text(h.title)
        for t in tickers:
            if t in allowed or _normalize_ticker(t) in allowed_norm:
                grouped[t].append(h)
    log.debug("match: %d headlines -> %d tickers matched", n_seen, len(grouped))
    return dict(grouped)


def match_pairs(
    headlines: Iterable[Headline],
    *,
    universe: tuple[str, ...] = UNIVERSE,
) -> list[tuple[str, Headline]]:
    """Flat list of ``(ticker, headline)`` pairs. Convenience wrapper."""
    grouped = match_headlines_to_universe(headlines, universe=universe)
    out: list[tuple[str, Headline]] = []
    for t, hs in grouped.items():
        for h in hs:
            out.append((t, h))
    return out


__all__ = [
    "match_headlines_to_universe",
    "match_pairs",
    "_tickers_in_text",
    "_ALIASES",
    "_ALIAS_TERMS",
    "_DIRECT_TICKERS",
]
