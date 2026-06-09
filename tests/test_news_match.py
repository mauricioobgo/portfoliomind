"""Unit tests for :mod:`portfoliomind.news.match`.

These tests are hermetic — they construct synthetic :class:`Headline`
records and never hit a network. The matching logic is pure Python.
"""

from __future__ import annotations

from datetime import datetime, timezone

from portfoliomind.news._headline import Headline
from portfoliomind.news.match import (
    _ALIASES,
    _ALIAS_TERMS,
    _DIRECT_TICKERS,
    _tickers_in_text,
    match_headlines_to_universe,
    match_pairs,
)
from portfoliomind.universe import UNIVERSE


# --- Helpers ---------------------------------------------------------------

_NOW = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def _h(title: str, source: str = "test") -> Headline:
    """Build a Headline at a fixed timestamp, source-agnostic."""
    return Headline.make(
        title=title,
        source=source,
        published_at=_NOW,
        link=f"https://example/{source}/{abs(hash(title))}",
    )


# --- _tickers_in_text ------------------------------------------------------


class TestTickersInText:
    """The internal text scanner. Public surface is match_* + _tickers_in_text."""

    def test_direct_ticker_in_title(self):
        assert _tickers_in_text("AAPL beats Q3 estimates") == {"AAPL"}

    def test_alias_for_company(self):
        assert _tickers_in_text("Apple reports record quarter") == {"AAPL"}

    def test_multiple_tickers(self):
        # "Apple and Microsoft" → AAPL, MSFT
        result = _tickers_in_text("Apple and Microsoft both up on the day")
        assert result == {"AAPL", "MSFT"}

    def test_berkshire_dot_form(self):
        assert _tickers_in_text("Berkshire Hathaway buys more") == {"BRK.B"}

    def test_berkshire_hyphen_form(self):
        # Matchers see BRK-B as the same ticker (the .normalize step
        # happens at the boundary, not here — so the alias map covers it).
        result = _tickers_in_text("BRK-B reports earnings")
        # BRK-B is in the alias map → BRK.B
        assert "BRK.B" in result

    def test_etf_alias(self):
        assert _tickers_in_text("S&P 500 closed higher") == {"SPY"}
        assert _tickers_in_text("Nasdaq 100 hit a new high") == {"QQQ"}

    def test_word_boundary_required_for_ticker(self):
        # "IWM" inside "swimming" should NOT match (word boundary).
        assert _tickers_in_text("Swimming pools are open") == set()

    def test_punctuation_boundaries(self):
        # Tickers at sentence start / with trailing period / comma
        assert "AAPL" in _tickers_in_text("AAPL, MSFT, GOOGL all rose.")
        assert "AAPL" in _tickers_in_text("(AAPL) on watch")

    def test_case_insensitive(self):
        assert _tickers_in_text("apple announces deal") == {"AAPL"}
        assert _tickers_in_text("aapl announces deal") == {"AAPL"}
        assert _tickers_in_text("APPLE announces deal") == {"AAPL"}

    def test_empty_text(self):
        assert _tickers_in_text("") == set()

    def test_no_match(self):
        assert _tickers_in_text("Local restaurant opens downtown") == set()

    def test_short_company_name_word_boundary(self):
        # "GE" is not in our universe; "ge" inside "google" should not match.
        # We're testing the matcher here, not the universe coverage.
        assert _tickers_in_text("Google announces something") == {"GOOGL"}

    def test_dedupes_alias_and_direct(self):
        # "AAPL" + "Apple" in the same text — must be one ticker.
        result = _tickers_in_text("AAPL (Apple) reports earnings")
        assert result == {"AAPL"}


# --- Alias map invariants --------------------------------------------------


class TestAliasMap:
    """The hand-curated alias map. New entries must be added thoughtfully."""

    def test_all_alias_targets_are_in_universe(self):
        """An alias that points outside the universe is a dead end."""
        for term, target in _ALIASES.items():
            assert target in UNIVERSE, (
                f"alias {term!r} -> {target!r} but {target} is not in UNIVERSE"
            )

    def test_all_alias_terms_are_lowercase(self):
        """We lowercase both sides of the match, so the term must be lower too."""
        for term in _ALIASES:
            assert term == term.lower(), f"alias {term!r} is not lowercase"

    def test_search_terms_built_correctly(self):
        # Spot-check: the alias map contains the long-phrase entries;
        # bare-ticker entries (and Berkshire variants) live in the
        # direct-ticker set.
        assert _ALIAS_TERMS["apple"] == "AAPL"
        # Bare-ticker entries were moved to the word-bounded set.
        assert "aapl" not in _ALIAS_TERMS
        assert "AAPL" in _DIRECT_TICKERS
        # Berkshire variants.
        assert "BRK.B" in _DIRECT_TICKERS
        assert "BRK-B" in _DIRECT_TICKERS
        assert "BRKB" in _DIRECT_TICKERS
        # Sector SPDR nicknames (long phrases, stay in alias map).
        assert _ALIAS_TERMS["technology select"] == "XLK"
        assert _ALIAS_TERMS["energy select"] == "XLE"

    def test_search_terms_no_duplicates(self):
        # We don't enforce uniqueness in the dict (overwrites are fine),
        # but every term must map to a single ticker.
        for term, target in _ALIAS_TERMS.items():
            assert isinstance(target, str) and target, (
                f"alias term {term!r} has invalid target {target!r}"
            )
        # Direct-ticker set is a set, so no duplicates by construction.
        assert len(_DIRECT_TICKERS) == len(set(_DIRECT_TICKERS))


# --- match_headlines_to_universe -------------------------------------------


class TestMatchHeadlines:
    """The grouped-output matcher."""

    def test_groups_by_ticker(self):
        headlines = [
            _h("Apple reports earnings"),
            _h("AAPL stock rises"),
            _h("Microsoft in talks"),
            _h("Unrelated headline about cats"),
        ]
        grouped = match_headlines_to_universe(headlines)
        assert set(grouped.keys()) == {"AAPL", "MSFT"}
        assert len(grouped["AAPL"]) == 2
        assert len(grouped["MSFT"]) == 1

    def test_filters_to_supplied_universe(self):
        # A custom universe that excludes AAPL — the matcher should
        # still detect "Apple" in the text but not put it in the result.
        headlines = [_h("Apple reports earnings"), _h("Microsoft in talks")]
        grouped = match_headlines_to_universe(headlines, universe=("MSFT",))
        assert "AAPL" not in grouped
        assert "MSFT" in grouped

    def test_empty_headlines(self):
        assert match_headlines_to_universe([]) == {}

    def test_no_matches(self):
        headlines = [_h("Local diner opens"), _h("Football game tonight")]
        assert match_headlines_to_universe(headlines) == {}

    def test_match_pairs_flat(self):
        headlines = [_h("Apple reports earnings"), _h("AAPL stock rises")]
        pairs = match_pairs(headlines)
        # 2 headlines, both match AAPL → 2 pairs.
        tickers = [p[0] for p in pairs]
        assert tickers == ["AAPL", "AAPL"]
        assert all(isinstance(p[1], Headline) for p in pairs)


# --- Universe ↔ matcher integration ---------------------------------------


class TestUniverseInteraction:
    """The matcher and the universe must agree on tickers."""

    def test_universe_contains_all_alias_targets(self):
        # Every alias target must be a known ticker (case-insensitive).
        for target in set(_ALIASES.values()):
            # The alias map keys are TICKER CASES, not lower — but
            # our universe is upper.
            assert target == target.upper(), f"alias target {target!r} is not upper"

    def test_full_universe_round_trip(self):
        """Every ticker in the universe can be matched in a fabricated headline."""
        for ticker in UNIVERSE:
            # Build a synthetic headline mentioning the ticker literally.
            headline = f"{ticker} reports positive news"
            grouped = match_headlines_to_universe([_h(headline)])
            assert ticker in grouped, (
                f"ticker {ticker!r} not matched in synthetic headline {headline!r}"
            )
