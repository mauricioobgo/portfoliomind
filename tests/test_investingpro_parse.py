"""Unit tests for :mod:`portfoliomind.investingpro.parse`.

These tests are browser-free: they exercise the pure-parse layer using
synthetic row data. The Playwright path is verified end-to-end by the
real CLI run (see card 2 acceptance criteria).
"""

from __future__ import annotations

import pytest

from portfoliomind.investingpro.parse import (
    RAW_PICKS_COLUMNS,
    RAW_PICKS_WIDTH,
    DeepDiveFacts,
    RawPick,
    _clean_cell,
    _coerce_pct,
    _coerce_pro_score,
    _looks_like_ticker,
    filter_new_rows,
    make_dedup_key,
    normalize_row_cells,
    parse_ai_picks_table,
    parse_deepdive_payload,
)


# --- _clean_cell ------------------------------------------------------------


class TestCleanCell:
    def test_strips_currency_glyphs(self):
        assert _clean_cell("$1,234.56") == "1,234.56"
        assert _clean_cell("€2.50") == "2.50"
        assert _clean_cell("£99.99") == "99.99"

    def test_normalises_thin_spaces(self):
        # \xa0 (non-breaking), \u2009 (thin), \u202f (narrow nbsp)
        assert _clean_cell("Apple\u00a0Inc.") == "Apple Inc."
        assert _clean_cell("1\u2009234") == "1 234"
        assert _clean_cell("5\u202f000") == "5 000"

    def test_collapses_whitespace(self):
        assert _clean_cell("  Apple   Inc.  ") == "Apple Inc."
        assert _clean_cell("\n\tAAPL\t") == "AAPL"

    def test_empty_and_none(self):
        assert _clean_cell("") == ""
        assert _clean_cell(None) == ""


# --- _looks_like_ticker -----------------------------------------------------


class TestLooksLikeTicker:
    def test_accepts_classic_ticker(self):
        assert _looks_like_ticker("AAPL")
        assert _looks_like_ticker("MSFT")
        assert _looks_like_ticker("BRK.B")
        assert _looks_like_ticker("BRK-B")

    def test_lowercased_is_ok(self):
        assert _looks_like_ticker("aapl")

    def test_rejects_company_name(self):
        assert not _looks_like_ticker("Apple Inc.")
        assert not _looks_like_ticker("Microsoft Corporation")

    def test_rejects_empty(self):
        assert not _looks_like_ticker("")
        assert not _looks_like_ticker(None)

    def test_rejects_too_long(self):
        # Max 10 chars in our regex (1 letter + 9 alphanum/dot/dash)
        assert not _looks_like_ticker("TOOLONGTICKER")


# --- _coerce_pro_score / _coerce_pct ---------------------------------------


class TestCoercions:
    def test_pro_score_strips_label(self):
        assert _coerce_pro_score("Pro Score 92.5") == "92.5"
        assert _coerce_pro_score("92.5") == "92.5"
        assert _coerce_pro_score("Score 88") == "88"
        # Conservative: only strip the exact label strings. "AI Score"
        # becomes "AI  75" (label removed, double space kept as-is for
        # the downstream Sheet to format).
        assert _coerce_pro_score("AI Score 75") == "AI  75"

    def test_pct_preserves_sign(self):
        assert _coerce_pct("+12.4%") == "+12.4%"
        assert _coerce_pct("-2.5%") == "-2.5%"

    def test_pct_parens_to_minus(self):
        assert _coerce_pct("(2.30%)") == "-2.30%"

    def test_pct_no_unit_passes_through(self):
        # No % — we keep the text as-is so the sheet gets a raw number
        # to be parsed later.
        assert _coerce_pct("12.4") == "12.4"


# --- normalize_row_cells ----------------------------------------------------


class TestNormalizeRowCells:
    def test_pads_short_row(self):
        out = normalize_row_cells(["AAPL", "Apple"], expected_width=9)
        assert len(out) == 9
        assert out[0] == "AAPL"
        assert out[1] == "Apple"
        assert out[8] == ""

    def test_truncates_long_row(self):
        out = normalize_row_cells(
            ["AAPL", "Apple", "92", "200", "180", "+11%",
             "Tech", "Buy", "2026-06-08T10:00:00-05:00", "EXTRA"],
            expected_width=9,
        )
        assert len(out) == 9
        assert "EXTRA" not in out

    def test_cleans_each_cell(self):
        out = normalize_row_cells(["$AAPL\u00a0", "  Apple  "], expected_width=9)
        assert out[0] == "AAPL"  # currency stripped, but ticker-shaped
        assert out[1] == "Apple"

    def test_handles_none(self):
        out = normalize_row_cells([None, "AAPL", None, None, None, None, None, None, None])
        assert out[0] == ""
        assert out[1] == "AAPL"

    def test_exact_width_passes_through(self):
        row = ["AAPL", "Apple Inc.", "92", "200", "180", "+11%",
               "Tech", "Buy", "2026-06-08T10:00:00-05:00"]
        out = normalize_row_cells(row)
        assert out == row


# --- parse_ai_picks_table --------------------------------------------------


class TestParseAiPicksTable:
    def test_parses_happy_path(self):
        rows = [
            ["AAPL", "Apple Inc.", "92.5", "220.00", "180.50",
             "+21.88%", "Technology", "Strong Buy"],
        ]
        picks = parse_ai_picks_table(rows, scraped_at="2026-06-08T10:00:00-05:00")
        assert len(picks) == 1
        p = picks[0]
        assert p.ticker == "AAPL"
        assert p.company_name == "Apple Inc."
        assert p.pro_score == "92.5"
        assert p.fair_value == "220.00"
        assert p.current_price == "180.50"
        assert p.upside_pct == "+21.88%"
        assert p.sector == "Technology"
        assert p.recommendation == "Strong Buy"
        assert p.scraped_at == "2026-06-08T10:00:00-05:00"

    def test_drops_blank_rows(self):
        rows = [
            ["AAPL", "Apple Inc.", "92.5", "220", "180",
             "+21.88%", "Technology", "Strong Buy"],
            ["", "", "", "", "", "", "", "", ""],
        ]
        assert len(parse_ai_picks_table(rows)) == 1

    def test_drops_header_row(self):
        rows = [
            ["Ticker", "Company Name", "Pro Score", "Fair Value",
             "Current Price", "Upside %", "Sector", "Recommendation"],
        ]
        assert parse_ai_picks_table(rows) == []

    def test_drops_company_name_only_row(self):
        # A "Top Movers" divider row in InvestingPro sometimes renders
        # as a single cell with no ticker. We drop it.
        rows = [["Top Movers Today", "", "", "", "", "", "", ""]]
        assert parse_ai_picks_table(rows) == []

    def test_uppercases_lowercase_ticker(self):
        rows = [
            ["aapl", "Apple Inc.", "92", "200", "180",
             "+11%", "Technology", "Buy"],
        ]
        picks = parse_ai_picks_table(rows)
        assert picks[0].ticker == "AAPL"

    def test_unknown_recommendation_kept_verbatim(self):
        rows = [
            ["NEWCO", "New Co", "70", "10", "8",
             "+25%", "Tech", "Hold"],  # "Hold" not in VALID_RECS
        ]
        picks = parse_ai_picks_table(rows)
        assert picks[0].recommendation == "Hold"

    def test_scraped_at_default_is_set_by_to_row(self):
        # If scraped_at is empty in the parse call, to_row fills it
        # with iso_now(). We verify the field is non-empty and is a
        # well-formed ISO timestamp.
        rows = [
            ["AAPL", "Apple", "92", "200", "180", "+11%", "Tech", "Buy"],
        ]
        picks = parse_ai_picks_table(rows)
        row = picks[0].to_row()
        assert row[8]  # non-empty
        assert "T" in row[8]  # ISO 8601 contains "T" between date and time
        # The Scraped At must not equal a Ticker shape (regression
        # guard against accidentally using cells[0] as the timestamp).
        assert row[8] != "AAPL"

    def test_to_row_matches_sheet_shape(self):
        pick = RawPick(
            ticker="AAPL",
            company_name="Apple Inc.",
            pro_score="92",
            fair_value="200",
            current_price="180",
            upside_pct="+11%",
            sector="Technology",
            recommendation="Strong Buy",
        )
        row = pick.to_row(scraped_at="2026-06-08T10:00:00-05:00")
        assert len(row) == RAW_PICKS_WIDTH
        assert row == [
            "AAPL",
            "Apple Inc.",
            "92",
            "200",
            "180",
            "+11%",
            "Technology",
            "Strong Buy",
            "2026-06-08T10:00:00-05:00",
        ]
        assert row[0] == "AAPL"  # Ticker
        assert row[1] == "Apple Inc."  # Company Name
        assert row[8] == "2026-06-08T10:00:00-05:00"  # Scraped At

    def test_pads_truncates_during_parse(self):
        # Real InvestingPro rows are sometimes 7 or 8 cells (the
        # Recommendation is missing on some rows). We pad with "".
        rows = [
            ["AAPL", "Apple Inc.", "92", "200", "180", "+11%", "Technology"],
        ]
        picks = parse_ai_picks_table(rows)
        assert len(picks) == 1
        assert picks[0].recommendation == ""


# --- parse_deepdive_payload -------------------------------------------------


class TestParseDeepDivePayload:
    def test_full_payload(self):
        payload = {
            "Market Cap": "2.94T",
            "P/E": "27.4",
            "EPS (TTM)": "6.42",
            "Dividend Yield": "0.52%",
            "Beta": "1.24",
            "Analyst Consensus": "Strong Buy",
        }
        facts = parse_deepdive_payload(
            "AAPL", payload, fetched_at="2026-06-08T10:00:00-05:00"
        )
        assert facts.ticker == "AAPL"
        assert facts.market_cap == "2.94T"
        assert facts.pe_ratio == "27.4"
        assert facts.eps_ttm == "6.42"
        assert facts.dividend_yield == "0.52%"
        assert facts.beta == "1.24"
        assert facts.analyst_consensus == "Strong Buy"
        assert facts.fetched_at == "2026-06-08T10:00:00-05:00"

    def test_aliases_accepted(self):
        facts = parse_deepdive_payload(
            "MSFT",
            {
                "market_cap": "2.8T",
                "pe_ratio": "32.1",
                "eps_ttm": "11.6",
                "dividend_yield": "0.81%",
                "beta": "0.94",
                "analyst_consensus": "Buy",
            },
        )
        assert facts.market_cap == "2.8T"
        assert facts.pe_ratio == "32.1"

    def test_empty_payload_keeps_empty_strings(self):
        facts = parse_deepdive_payload("NEWCO", {})
        assert facts.ticker == "NEWCO"
        assert facts.market_cap == ""
        assert facts.pe_ratio == ""
        assert facts.eps_ttm == ""
        assert facts.dividend_yield == ""
        assert facts.beta == ""
        assert facts.analyst_consensus == ""

    def test_missing_ticker_raises(self):
        with pytest.raises(ValueError, match="ticker is required"):
            parse_deepdive_payload("", {"Market Cap": "1T"})

    def test_ticker_uppercased(self):
        facts = parse_deepdive_payload("aapl", {"Market Cap": "2.94T"})
        assert facts.ticker == "AAPL"

    def test_currency_glyphs_stripped(self):
        facts = parse_deepdive_payload(
            "AAPL",
            {
                "Market Cap": "$2,940B",
                "EPS (TTM)": "$6.42",
            },
        )
        assert facts.market_cap == "2,940B"
        assert facts.eps_ttm == "6.42"

    def test_to_row_shape(self):
        facts = DeepDiveFacts(
            ticker="AAPL",
            market_cap="2.94T",
            pe_ratio="27.4",
            eps_ttm="6.42",
            dividend_yield="0.52%",
            beta="1.24",
            analyst_consensus="Strong Buy",
        )
        row = facts.to_row(fetched_at="2026-06-08T10:00:00-05:00")
        assert len(row) == 8  # DeepDiveFacts has 8 fields


# --- make_dedup_key + filter_new_rows --------------------------------------


class TestDedup:
    def _row(self, ticker: str, ts: str) -> list[str]:
        return [
            ticker, "Apple Inc.", "92", "200", "180", "+11%",
            "Technology", "Buy", ts,
        ]

    def test_dedup_key_shape(self):
        key = make_dedup_key(self._row("AAPL", "2026-06-08T10:00:00-05:00"))
        assert key == "AAPL|2026-06-08T10:00:00-05:00"

    def test_dedup_key_raises_on_short_row(self):
        with pytest.raises(ValueError, match=r"RAW_PICKS row has 1 cells, expected 9"):
            make_dedup_key(["AAPL"])

    def test_filter_new_rows_drops_duplicates(self):
        new = [
            self._row("AAPL", "2026-06-08T10:00:00-05:00"),
            self._row("MSFT", "2026-06-08T10:00:00-05:00"),
        ]
        existing = [
            self._row("AAPL", "2026-06-08T10:00:00-05:00"),
        ]
        fresh = filter_new_rows(new, existing)
        assert len(fresh) == 1
        assert fresh[0][0] == "MSFT"

    def test_filter_new_rows_keeps_all_when_no_overlap(self):
        new = [self._row("AAPL", "2026-06-08T10:00:00-05:00")]
        existing = [self._row("MSFT", "2026-06-08T09:00:00-05:00")]
        assert filter_new_rows(new, existing) == new

    def test_filter_new_rows_idempotent_against_self(self):
        # Re-running with the same timestamp must produce zero new rows.
        rows = [
            self._row("AAPL", "2026-06-08T10:00:00-05:00"),
            self._row("MSFT", "2026-06-08T10:00:00-05:00"),
        ]
        assert filter_new_rows(rows, rows) == []

    def test_filter_new_rows_different_timestamps_pass(self):
        # Same ticker, different timestamps: both rows are fresh.
        a1 = self._row("AAPL", "2026-06-08T09:00:00-05:00")
        a2 = self._row("AAPL", "2026-06-08T10:00:00-05:00")
        assert filter_new_rows([a2], [a1]) == [a2]

    def test_filter_new_rows_handles_empty_existing(self):
        rows = [
            self._row("AAPL", "2026-06-08T10:00:00-05:00"),
        ]
        assert filter_new_rows(rows, []) == rows


# --- Contract guards --------------------------------------------------------


class TestContract:
    def test_raw_picks_columns_match_schema(self):
        # If this fails, the parser has drifted from the spec.
        assert RAW_PICKS_COLUMNS == (
            "Ticker",
            "Company Name",
            "Pro Score",
            "Fair Value",
            "Current Price",
            "Upside %",
            "Sector",
            "Recommendation",
            "Scraped At",
        )

    def test_raw_picks_width(self):
        assert RAW_PICKS_WIDTH == 9

    def test_dedup_key_only_depends_on_ticker_and_timestamp(self):
        # The dedup contract is "Ticker + Scraped At". Two rows with
        # the same Ticker and Scraped At but different Pro Score are
        # considered duplicates. That's the spec.
        r1 = ["AAPL", "Apple", "92", "200", "180", "+11%", "Tech", "Buy",
              "2026-06-08T10:00:00-05:00"]
        r2 = ["AAPL", "Apple Inc.", "88", "210", "180", "+16%", "Tech",
              "Strong Buy", "2026-06-08T10:00:00-05:00"]
        assert make_dedup_key(r1) == make_dedup_key(r2)
