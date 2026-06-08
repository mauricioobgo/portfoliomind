"""Pure-parse layer for InvestingPro AI Picks + deep-dive data.

This module is deliberately browser-free so unit tests can exercise every
edge case in isolation. The Playwright-driven code in :mod:`scrape` and
:mod:`deepdive` produces raw page fragments (HTML strings or lists of cell
text); ``parse_ai_picks_table`` and ``parse_deepdive_payload`` turn those
fragments into the dataclass shapes that flow into the Google Sheet.

The dataclasses match the column definitions in
:mod:`portfoliomind.sheets.schema.TAB_HEADERS` exactly:

    RAW_PICKS (9 columns):
        Ticker, Company Name, Pro Score, Fair Value, Current Price,
        Upside %, Sector, Recommendation, Scraped At

Keeping the shape explicit means a bug in the parser surfaces as a
``ValueError`` at the boundary, not as a silent column shift inside the
sheet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ..time_utils import iso_now

# --- Constants pulled from the spec -----------------------------------------

#: Column headers for the RAW_PICKS tab, in sheet order. Used to validate
#: the contract at the boundary so we never accidentally shift a column.
RAW_PICKS_COLUMNS: tuple[str, ...] = (
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

#: A row in RAW_PICKS is 9 cells. Used by the dedup logic to assert shape
#: before appending to the sheet.
RAW_PICKS_WIDTH: int = len(RAW_PICKS_COLUMNS)

# Regexes for cleaning InvestingPro's noisy text. We see things like
# "AAPL  Apple Inc." or "1,234.56" with currency symbols or thin spaces.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
# Header labels that we KNOW are not tickers, even though they match the
# ticker regex. InvestingPro's table header row is "Ticker", "Company
# Name", ... — we want to drop those without false-positiving on a
# legitimate 1-6 char ticker.
_HEADER_LABELS = frozenset(
    {
        "TICKER",
        "SYMBOL",
        "STOCK",
        "NAME",
        "COMPANY",
        "SCORE",
        "FAIR",
        "PRICE",
        "UPSIDE",
        "SECTOR",
        "RECOMMENDATION",
        "RANK",
    }
)
# InvestingPro recommendations are typically these values, but the site
# sometimes shows "Strong Buy" / "Strong Sell" too. Accept a small set.
_VALID_RECS = {
    "Strong Buy",
    "Buy",
    "Neutral",
    "Sell",
    "Strong Sell",
    "Outperform",
    "Underperform",
}


# --- Data shapes ------------------------------------------------------------


@dataclass(frozen=True)
class RawPick:
    """One row from the InvestingPro AI Picks table.

    The 9 fields map 1:1 onto the columns in
    :data:`RAW_PICKS_COLUMNS`. ``scraped_at`` is filled by :meth:`to_row`
    at the moment we materialise the row for the sheet.
    """

    ticker: str
    company_name: str
    pro_score: str  # string so we don't lose "92.5" vs "92"; sheet will parse
    fair_value: str
    current_price: str
    upside_pct: str
    sector: str
    recommendation: str
    scraped_at: str = ""

    def to_row(self, *, scraped_at: Optional[str] = None) -> list[str]:
        """Materialise the row in sheet order.

        ``scraped_at`` defaults to ``iso_now()`` so the caller doesn't need
        to manage timestamps, but it can be overridden for tests and for
        back-fills.
        """
        ts = scraped_at if scraped_at is not None else iso_now()
        return [
            self.ticker,
            self.company_name,
            self.pro_score,
            self.fair_value,
            self.current_price,
            self.upside_pct,
            self.sector,
            self.recommendation,
            ts,
        ]


@dataclass(frozen=True)
class DeepDiveFacts:
    """Fundamentals block from a ticker's deep-dive page.

    Card 2 captures the headline metrics only — card 3/4 may extend this.
    All fields are strings because the values come from a heterogeneous
    layout (some are money, some %, some text) and the sheet needs them as
    plain text. The downstream forecast engine (future card) is the place
    to coerce these to numeric.
    """

    ticker: str
    market_cap: str = ""
    pe_ratio: str = ""
    eps_ttm: str = ""
    dividend_yield: str = ""
    beta: str = ""
    analyst_consensus: str = ""
    fetched_at: str = ""

    def to_row(self, *, fetched_at: Optional[str] = None) -> list[str]:
        """Row in the order the caller asked for (e.g. AGENT_LOG payload).

        Note: deep-dive facts are not destined for RAW_PICKS (that tab is
        the AI Picks table only). They're emitted as a dict-friendly
        payload for AGENT_LOG and future forecast inputs.
        """
        ts = fetched_at if fetched_at is not None else iso_now()
        return [
            self.ticker,
            self.market_cap,
            self.pe_ratio,
            self.eps_ttm,
            self.dividend_yield,
            self.beta,
            self.analyst_consensus,
            ts,
        ]


# --- Helpers ----------------------------------------------------------------


def _clean_cell(text: str) -> str:
    """Collapse whitespace, strip currency glyphs and stray thin spaces.

    InvestingPro serves numbers with a mix of regular spaces, non-breaking
    spaces (\\xa0), and currency symbols. We normalise aggressively because
    the dedup key (Ticker + Scraped At) is only as good as the cell values
    it compares against.
    """
    if text is None:
        return ""
    s = text.replace("\xa0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_ticker(text: str) -> bool:
    """``AAPL`` is a ticker; ``Apple Inc.`` is not. Conservative heuristic.

    Also rejects known table-header labels (TICKER, SYMBOL, etc.) that
    happen to match the ticker regex. A real InvestingPro header row
    starts with ``Ticker`` and must not be picked up as a pick.
    """
    if not text:
        return False
    candidate = _clean_cell(text).upper()
    if not candidate:
        return False
    if candidate in _HEADER_LABELS:
        return False
    # Some InvestingPro rows show the ticker first in mixed case ("aapl")
    # so we upcase before matching.
    return bool(_TICKER_RE.match(candidate))


def _coerce_pro_score(text: str) -> str:
    """Strip a trailing "Pro Score" or "Score" label if InvestingPro renders it."""
    s = _clean_cell(text)
    # Remove common labels
    for label in ("Pro Score", "Score", "AI Score"):
        s = s.replace(label, "").strip()
    return s


def _coerce_pct(text: str) -> str:
    """Extract a percentage value, returning the cleaned text with the % intact.

    Examples:
        "+12.4%"  -> "+12.4%"
        "(2.30%)" -> "-2.30%"  (parens -> negative, common finance convention)
        "12.4"    -> "12.4"
    """
    s = _clean_cell(text)
    if not s:
        return s
    # Negative-in-parens (accounting style)
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()
    # Drop a trailing unit if it's not a %
    if s.endswith("%"):
        return s
    return s


def _is_blank_row(cells: Iterable[str]) -> bool:
    """True when every cell is empty after cleaning.

    InvestingPro sometimes pads the table with empty rows; we drop those
    so the dedup key isn't poisoned with garbage.
    """
    return all(not _clean_cell(c) for c in cells)


# --- Public parsing API -----------------------------------------------------


def normalize_row_cells(
    cells: Iterable[object], *, expected_width: int = RAW_PICKS_WIDTH
) -> list[str]:
    """Take a raw row (e.g. from Playwright's ``text_content()``) and clean it.

    - Coerces each cell to a stripped string.
    - Pads short rows with empty strings (InvestingPro sometimes drops the
      trailing cell on the last column).
    - Truncates rows that are too long (we keep the first ``expected_width``
      cells; any extras are dropped — they're either junk or a duplicate
      column the site sometimes renders in some locales).
    """
    cleaned: list[str] = []
    for c in cells:
        if c is None:
            cleaned.append("")
        else:
            cleaned.append(_clean_cell(str(c)))
    if len(cleaned) < expected_width:
        cleaned.extend([""] * (expected_width - len(cleaned)))
    if len(cleaned) > expected_width:
        cleaned = cleaned[:expected_width]
    return cleaned


def parse_ai_picks_table(
    rows: Iterable[Iterable[object]], *, scraped_at: Optional[str] = None
) -> list[RawPick]:
    """Convert a list of table rows into :class:`RawPick` objects.

    The input is the ``tr.text_content()`` of each ``<tr>`` in the AI Picks
    results table (or, equivalently, the cell text from any equivalent
    source). Blank rows are dropped. Rows that don't have a ticker-shaped
    first cell are dropped too — those are the header row and any section
    dividers the site renders.

    The function never raises on a malformed row: it skips the row and
    keeps going. The caller can compare the input length to the output
    length to detect that the page layout has shifted.
    """
    out: list[RawPick] = []
    ts = scraped_at
    for raw in rows:
        cells = normalize_row_cells(raw)
        if _is_blank_row(cells):
            continue
        if not _looks_like_ticker(cells[0]):
            # Header row or section divider — skip.
            continue
        ticker = cells[0].upper()
        rec = cells[7]
        if rec and rec not in _VALID_RECS:
            # Don't fail — InvestingPro sometimes invents new labels
            # (e.g. "Hold"). We accept whatever string is there but
            # normalise whitespace.
            rec = _clean_cell(rec)
        pick = RawPick(
            ticker=ticker,
            company_name=_clean_cell(cells[1]),
            pro_score=_coerce_pro_score(cells[2]),
            fair_value=_clean_cell(cells[3]),
            current_price=_clean_cell(cells[4]),
            upside_pct=_coerce_pct(cells[5]),
            sector=_clean_cell(cells[6]),
            recommendation=rec,
            scraped_at=ts or "",
        )
        out.append(pick)
    return out


def parse_deepdive_payload(
    ticker: str,
    payload: dict[str, object],
    *,
    fetched_at: Optional[str] = None,
) -> DeepDiveFacts:
    """Coerce a dict of {label: raw_text} from the deep-dive page into a
    :class:`DeepDiveFacts`.

    InvestingPro's deep-dive page renders fundamentals as a label/value
    grid. The scrape layer is responsible for turning the DOM into a
    ``{"Market Cap": "2.94T", "P/E": "27.4", ...}`` dict; this function
    normalises the keys and produces the dataclass.
    """
    if not ticker:
        raise ValueError("ticker is required for deep-dive facts")

    ts = fetched_at or iso_now()

    def lookup(*aliases: str) -> str:
        for alias in aliases:
            if alias in payload and payload[alias] is not None:
                return _clean_cell(str(payload[alias]))
        return ""

    return DeepDiveFacts(
        ticker=ticker.upper(),
        market_cap=lookup("Market Cap", "MarketCap", "market_cap"),
        pe_ratio=lookup("P/E", "P/E Ratio", "PE Ratio", "pe_ratio"),
        eps_ttm=lookup("EPS (TTM)", "EPS", "EPS TTM", "eps_ttm"),
        dividend_yield=lookup("Dividend Yield", "Yield", "dividend_yield"),
        beta=lookup("Beta", "beta"),
        analyst_consensus=lookup(
            "Analyst Consensus",
            "Consensus",
            "Recommendation",
            "analyst_consensus",
        ),
        fetched_at=ts,
    )


# --- Dedup ------------------------------------------------------------------


def make_dedup_key(row: list[str]) -> str:
    """Build the dedup key for one RAW_PICKS row.

    Contract from card 1: Ticker + Scraped At. The Scraped At is the 9th
    column (index 8). The ticker is index 0.
    """
    if len(row) < RAW_PICKS_WIDTH:
        raise ValueError(
            f"RAW_PICKS row has {len(row)} cells, expected {RAW_PICKS_WIDTH}"
        )
    return f"{row[0]}|{row[8]}"


def filter_new_rows(
    new_rows: list[list[str]], existing_rows: list[list[str]]
) -> list[list[str]]:
    """Drop any row from ``new_rows`` whose dedup key is already in
    ``existing_rows``.

    Both inputs are list-of-list with the standard 9-cell shape. The
    function is O(len(new) + len(existing)) by building a set of existing
    keys once.
    """
    existing_keys = {make_dedup_key(r) for r in existing_rows if r}
    return [r for r in new_rows if make_dedup_key(r) not in existing_keys]


# --- Self-test (run with: python -m portfoliomind.investingpro.parse) ------


if __name__ == "__main__":
    # A tiny smoke test that exercises the parser end-to-end without a
    # browser. Useful for `python -m portfoliomind.investingpro.parse`.
    sample_rows = [
        [
            "AAPL",
            "Apple Inc.",
            "92.5",
            "220.00",
            "180.50",
            "+21.88%",
            "Technology",
            "Strong Buy",
        ],
        ["", "", "", "", "", "", "", "", ""],  # blank
        ["MSFT", "Microsoft Corp", "88", "430", "402.1", "+6.95%", "Technology", "Buy"],
        [
            "GOOGL",
            "Alphabet Inc Class A",
            "85.4",
            "180.00",
            "165.40",
            "+8.83%",
            "Communication Services",
            "Strong Buy",
        ],
    ]
    picks = parse_ai_picks_table(sample_rows)
    print(f"parsed {len(picks)} picks")
    for p in picks:
        print("  ", p)
    # Round-trip via to_row + dedup
    rows = [p.to_row() for p in picks]
    new = filter_new_rows(rows, [])
    print(f"first write would append {len(new)} rows")
    new2 = filter_new_rows(rows, rows)
    print(f"re-run would append {len(new2)} rows (should be 0)")
    # Manual dedup key check
    for r in rows:
        print("  key=", make_dedup_key(r))
    # Timestamp formatting sanity
    print("iso_now:", iso_now())
