"""Integration tests for :mod:`portfoliomind.investingpro.scrape`.

We don't use a real Playwright ``Page`` here (it would need a real
InvestingPro login). Instead we patch ``_read_table_rows`` to return
synthetic row data, and patch ``SheetsClient`` to an in-memory fake.
This gives us end-to-end coverage of:

* ``scrape_ai_picks`` reading, parsing, and dedup-appending to RAW_PICKS
* Re-running the same command producing 0 new rows

The card 2 acceptance criterion is that the first run appends 5 rows and
the second run appends 0. These tests assert exactly that.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from portfoliomind.config import PortfoliomindConfig
from portfoliomind.investingpro.scrape import scrape_ai_picks
from portfoliomind.sheets.client import SheetsClient
from portfoliomind.sheets.schema import RAW_PICKS, TAB_HEADERS
from portfoliomind.time_utils import iso_now

from .conftest import full_env


# --- In-memory fake of SheetsClient -----------------------------------------


class _FakeWorksheet:
    """An in-memory worksheet. ``values`` is a 2D list indexed [row][col]."""

    def __init__(self, headers: list[str]) -> None:
        self.headers = headers
        self.values: list[list[str]] = [list(headers)]


class _FakeSheetsClient:
    """In-memory substitute for :class:`SheetsClient`.

    Mirrors the public surface that ``scrape_ai_picks`` actually uses:
    ``ensure_worksheet``, ``read_range``, ``append_rows``. We track the
    call list so tests can assert on what the scrape attempted to do.
    """

    def __init__(self, existing_rows: list[list[str]] | None = None) -> None:
        self.worksheets: dict[str, _FakeWorksheet] = {}
        if existing_rows:
            self.worksheets[RAW_PICKS] = _FakeWorksheet(TAB_HEADERS[RAW_PICKS])
            self.worksheets[RAW_PICKS].values.extend(existing_rows)
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def ensure_worksheet(self, sheet_id: str, title: str, headers: list[str]) -> dict:
        self.calls.append(("ensure_worksheet", (sheet_id, title, headers), {}))
        if title not in self.worksheets:
            self.worksheets[title] = _FakeWorksheet(headers)
        return {"sheetId": 0, "title": title}

    def read_range(self, sheet_id: str, tab_name: str, range_a1: str) -> list[list[str]]:
        self.calls.append(("read_range", (sheet_id, tab_name, range_a1), {}))
        ws = self.worksheets.get(tab_name)
        if not ws:
            return []
        return [list(row) for row in ws.values]

    def append_rows(
        self, sheet_id: str, tab_name: str, values: list[list[str]]
    ) -> int:
        self.calls.append(("append_rows", (sheet_id, tab_name, values), {}))
        ws = self.worksheets.setdefault(tab_name, _FakeWorksheet(TAB_HEADERS.get(tab_name, [])))
        start = len(ws.values) + 1  # 1-indexed
        ws.values.extend(values)
        return start

    def row_count(self, sheet_id: str, tab_name: str) -> int:
        return len(self.worksheets.get(tab_name, _FakeWorksheet([])).values)


# --- Helpers ----------------------------------------------------------------


def _make_page(rows: list[list[str]]) -> MagicMock:
    """Build a MagicMock that satisfies the ``_read_table_rows`` contract.

    ``_read_table_rows`` calls ``page.query_selector_all(sel)`` to find
    ``<tr>`` elements, then ``tr.query_selector_all('th, td')`` for the
    cells, then ``td.text_content()`` to get the text. We mock just
    enough to return our synthetic rows.
    """
    page = MagicMock()

    def fake_query_all(selector: str, *args, **kwargs):
        # The first selector that yields >0 rows wins in _read_table_rows.
        # Always return one row per synthetic row, so the first selector
        # that matches will be the one we hand back data for.
        if "tr" not in selector:
            return []
        return [_FakeTr(r) for r in rows]

    page.query_selector_all.side_effect = fake_query_all
    page.url = "https://www.investing.com/pro/propicks"
    return page


class _FakeTr:
    def __init__(self, cells: list[str]) -> None:
        self._cells = cells

    def query_selector_all(self, selector: str) -> list:
        if "th" in selector or "td" in selector:
            return [_FakeTd(c) for c in self._cells]
        return []


class _FakeTd:
    def __init__(self, text: str) -> None:
        self._text = text

    def text_content(self) -> str:
        return self._text


# --- Tests ------------------------------------------------------------------


def _config() -> PortfoliomindConfig:
    return PortfoliomindConfig.from_env(env=full_env("test-sheet-id-001"))


SAMPLE_ROWS = [
    ["AAPL", "Apple Inc.", "92.5", "220.00", "180.50",
     "+21.88%", "Technology", "Strong Buy"],
    ["MSFT", "Microsoft Corp", "88.0", "430.00", "402.10",
     "+6.95%", "Technology", "Buy"],
    ["GOOGL", "Alphabet Inc Class A", "85.4", "180.00", "165.40",
     "+8.83%", "Communication Services", "Strong Buy"],
    ["AMZN", "Amazon.com Inc", "82.1", "210.00", "185.20",
     "+13.39%", "Consumer Cyclical", "Strong Buy"],
    ["NVDA", "NVIDIA Corp", "90.0", "1100.00", "950.00",
     "+15.79%", "Technology", "Strong Buy"],
]


def test_first_run_appends_all_rows():
    client = _FakeSheetsClient()
    page = _make_page(SAMPLE_ROWS)
    ts = "2026-06-08T10:00:00-05:00"
    result = scrape_ai_picks(
        page, cast(SheetsClient, client), _config(), limit=5, scraped_at=ts
    )
    assert len(result.picks) == 5
    assert len(result.new_rows) == 5
    assert result.skipped_duplicates == 0
    assert result.sheet_first_row == 2  # 1-indexed, header is row 1
    # All rows are now on the (fake) sheet.
    assert client.row_count("test-sheet-id-001", RAW_PICKS) == 6  # header + 5


def test_second_run_is_idempotent():
    """The card 2 acceptance criterion: re-running must append 0 new rows."""
    client = _FakeSheetsClient()
    page = _make_page(SAMPLE_ROWS)
    ts = "2026-06-08T10:00:00-05:00"
    # First run
    first = scrape_ai_picks(
        page, cast(SheetsClient, client), _config(), limit=5, scraped_at=ts
    )
    assert len(first.new_rows) == 5

    # Second run, same timestamp, same data
    page2 = _make_page(SAMPLE_ROWS)
    second = scrape_ai_picks(
        page2, cast(SheetsClient, client), _config(), limit=5, scraped_at=ts
    )
    assert len(second.picks) == 5
    assert len(second.new_rows) == 0
    assert second.skipped_duplicates == 5
    assert second.sheet_first_row == 0
    # Sheet still has exactly 6 rows (1 header + 5 data).
    assert client.row_count("test-sheet-id-001", RAW_PICKS) == 6


def test_new_timestamp_writes_again():
    """A different ``scraped_at`` is a different dedup key — second run
    DOES append. This is the "tomorrow's run" case."""
    client = _FakeSheetsClient()
    page = _make_page(SAMPLE_ROWS)
    scrape_ai_picks(
        page, cast(SheetsClient, client), _config(), limit=5, scraped_at="2026-06-08T10:00:00-05:00"
    )
    page2 = _make_page(SAMPLE_ROWS)
    second = scrape_ai_picks(
        page2,
        client,
        _config(),
        limit=5,
        scraped_at="2026-06-09T10:00:00-05:00",
    )
    assert len(second.new_rows) == 5
    assert second.skipped_duplicates == 0
    # Sheet now has header + 5 + 5 = 11 rows.
    assert client.row_count("test-sheet-id-001", RAW_PICKS) == 11


def test_limit_caps_rows_appended():
    client = _FakeSheetsClient()
    page = _make_page(SAMPLE_ROWS)  # 5 rows available
    result = scrape_ai_picks(
        page, cast(SheetsClient, client), _config(), limit=3, scraped_at="2026-06-08T10:00:00-05:00"
    )
    assert len(result.picks) == 3
    assert len(result.new_rows) == 3


def test_empty_table_returns_empty_result(monkeypatch):
    """An empty page must short-circuit, not wait 60s for rows to appear."""
    from portfoliomind.investingpro import scrape as scrape_mod

    # Patch _read_table_rows to return [] immediately. The real path
    # would wait 60s for the table to render; we bypass that here.
    monkeypatch.setattr(scrape_mod, "_read_table_rows", lambda page, limit: [])

    client = _FakeSheetsClient()
    page = _make_page([])
    result = scrape_ai_picks(
        page, cast(SheetsClient, client), _config(), limit=5, scraped_at=iso_now()
    )
    assert result.picks == []
    assert result.new_rows == []
    assert result.skipped_duplicates == 0
    assert result.sheet_first_row == 0
    # No append_rows call should have happened.
    append_calls = [c for c in client.calls if c[0] == "append_rows"]
    assert append_calls == []


def test_existing_rows_short_are_padded_before_dedup():
    """A row in the sheet that is shorter than 9 cells must not crash
    the dedup loop. InvestingPro sometimes writes blank cells."""
    short_row = ["AAPL", "Apple Inc."]  # only 2 cells
    client = _FakeSheetsClient(existing_rows=[short_row])
    page = _make_page(SAMPLE_ROWS)
    result = scrape_ai_picks(
        page, cast(SheetsClient, client), _config(), limit=5, scraped_at="2026-06-08T10:00:00-05:00"
    )
    # All 5 picks are fresh (the existing short row has a different
    # timestamp and a different dedup key).
    assert len(result.new_rows) == 5


def test_ensure_worksheet_called_for_raw_picks():
    """Idempotent header guard: even if the sheet was hand-created
    without RAW_PICKS, the scrape will create it with the right headers."""
    client = _FakeSheetsClient()  # no existing RAW_PICKS
    page = _make_page(SAMPLE_ROWS)
    scrape_ai_picks(
        page, cast(SheetsClient, client), _config(), limit=5, scraped_at="2026-06-08T10:00:00-05:00"
    )
    # ensure_worksheet should have been called for RAW_PICKS.
    titles = [
        c[1][1]
        for c in client.calls
        if c[0] == "ensure_worksheet"
    ]
    assert RAW_PICKS in titles


def test_sheet_shape_is_preserved_per_row():
    """Every appended row must be exactly 9 cells in the canonical order."""
    client = _FakeSheetsClient()
    page = _make_page(SAMPLE_ROWS)
    result = scrape_ai_picks(
        page, cast(SheetsClient, client), _config(), limit=5, scraped_at="2026-06-08T10:00:00-05:00"
    )
    for r in result.new_rows:
        assert len(r) == 9
        # Ticker is first, Scraped At is last.
        assert r[0].isalpha() and r[0].isupper()
        assert "T" in r[8]  # ISO 8601


def test_zero_limit_raises():
    client = _FakeSheetsClient()
    page = _make_page(SAMPLE_ROWS)
    with pytest.raises(ValueError, match="limit must be > 0"):
        scrape_ai_picks(page, cast(SheetsClient, client), _config(), limit=0)


def test_missing_sheet_id_raises():
    """A blank GOOGLE_SHEET_ID is rejected — the bootstrap step belongs
    to the caller, not the scrape module."""
    cfg = _config()
    from dataclasses import replace

    cfg = replace(cfg, google_sheet_id="")
    client = _FakeSheetsClient()
    page = _make_page(SAMPLE_ROWS)
    with pytest.raises(Exception, match="GOOGLE_SHEET_ID"):
        scrape_ai_picks(page, client, cfg, limit=5)
