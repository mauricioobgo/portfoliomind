"""Unit tests for :mod:`portfoliomind.investingpro.deepdive`.

The deep-dive module is the thinnest of the three — it delegates parsing
to :mod:`portfoliomind.investingpro.parse` and orchestrates AGENT_LOG
emission. We test the orchestration with a fake ``Page`` and a fake
``SheetsClient``.
"""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock


from portfoliomind.config import PortfoliomindConfig
from portfoliomind.investingpro.deepdive import deepdive_top_n
from portfoliomind.sheets.client import SheetsClient
from portfoliomind.sheets.schema import AGENT_LOG
from portfoliomind.time_utils import iso_now

from .conftest import full_env


# --- In-memory fake SheetsClient (minimal) ---------------------------------


class _FakeWorksheet:
    def __init__(self, headers: list[str]) -> None:
        self.headers = headers
        self.values: list[list[str]] = [list(headers)]


class _FakeSheetsClient:
    def __init__(self) -> None:
        self.worksheets: dict[str, _FakeWorksheet] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def ensure_worksheet(self, sheet_id: str, title: str, headers: list[str]) -> dict:
        self.calls.append(("ensure_worksheet", (sheet_id, title)))
        if title not in self.worksheets:
            self.worksheets[title] = _FakeWorksheet(headers)
        return {"sheetId": 0, "title": title}

    def append_rows(
        self, sheet_id: str, tab_name: str, values: list[list[str]]
    ) -> int:
        self.calls.append(("append_rows", (sheet_id, tab_name, len(values))))
        ws = self.worksheets.setdefault(tab_name, _FakeWorksheet([]))
        start = len(ws.values) + 1
        ws.values.extend(values)
        return start


# --- Page mock that returns a static payload --------------------------------


def _make_page_with_payload(payload: dict[str, str]) -> MagicMock:
    """A mock Page that returns the given fundamentals payload for every
    ticker."""
    page = MagicMock()

    def fake_goto(url: str, **kwargs: Any) -> None:
        page.url = url

    page.goto.side_effect = fake_goto
    page.url = "https://www.investing.com/pro/AAPL"

    def fake_query_all(selector: str):
        # dl dt -> labels, dl dd -> values
        if selector == "dl dt":
            return [_FakeDt(k) for k in payload.keys()]
        if selector == "dl dd":
            return [_FakeDd(v) for v in payload.values()]
        if "key-info" in selector or "fundamentals" in selector or "section.fundamentals" in selector:
            return []
        return []

    page.query_selector_all.side_effect = fake_query_all
    return page


class _FakeDt:
    def __init__(self, text: str) -> None:
        self._t = text

    def text_content(self) -> str:
        return self._t


class _FakeDd:
    def __init__(self, text: str) -> None:
        self._t = text

    def text_content(self) -> str:
        return self._t


def _config() -> PortfoliomindConfig:
    return PortfoliomindConfig.from_env(env=full_env("test-sheet-deepdive"))


PAYLOAD = {
    "Market Cap": "2.94T",
    "P/E": "27.4",
    "EPS (TTM)": "6.42",
    "Dividend Yield": "0.52%",
    "Beta": "1.24",
    "Analyst Consensus": "Strong Buy",
}


# --- Tests ------------------------------------------------------------------


def test_deepdive_captures_one_ticker():
    page = _make_page_with_payload(PAYLOAD)
    client = _FakeSheetsClient()
    result = deepdive_top_n(
        page, cast(SheetsClient, client), _config(),
        tickers=["AAPL"], fetched_at="2026-06-08T10:00:00-05:00",
    )
    assert len(result.successes) == 1
    assert len(result.failures) == 0
    facts = result.successes[0]
    assert facts.ticker == "AAPL"
    assert facts.market_cap == "2.94T"
    assert facts.pe_ratio == "27.4"


def test_deepdive_emits_agent_log_rows():
    page = _make_page_with_payload(PAYLOAD)
    client = _FakeSheetsClient()
    deepdive_top_n(
        page, cast(SheetsClient, client), _config(),
        tickers=["AAPL", "MSFT"], fetched_at="2026-06-08T10:00:00-05:00",
    )
    # Should have appended 2 rows (one per ticker) to AGENT_LOG.
    append_calls = [c for c in client.calls if c[0] == "append_rows"]
    assert len(append_calls) == 1
    sheet_id, tab, count = append_calls[0][1]
    assert tab == AGENT_LOG
    assert count == 2
    # Verify the JSON payload of one of the rows.
    ws = client.worksheets[AGENT_LOG]
    assert ws.values[1][0] == "2026-06-08T10:00:00-05:00"  # Timestamp
    msg = ws.values[1][3]
    data = json.loads(msg)
    assert data["event"] == "investingpro.deepdive.success"
    assert data["ticker"] in {"AAPL", "MSFT"}


def test_deepdive_continues_on_per_ticker_failure():
    """A failure on one ticker must not abort the rest of the batch."""

    # The fake page that errors out on the second ticker.
    page = MagicMock()
    tickers_seen: list[str] = []

    def fake_goto(url: str, **kwargs: Any) -> None:
        tickers_seen.append(url)
        page.url = url
        if "MSFT" in url:
            raise Exception("network error")

    page.goto.side_effect = fake_goto
    page.query_selector_all.return_value = []
    page.url = ""

    client = _FakeSheetsClient()
    result = deepdive_top_n(
        page, cast(SheetsClient, client), _config(),
        tickers=["AAPL", "MSFT"], fetched_at=iso_now(),
    )
    # First ticker should be a failure (no payload), second fails by goto.
    # Both should be in the failures list.
    assert len(result.successes) == 0
    assert len(result.failures) == 2
    assert {t for (t, _) in result.failures} == {"AAPL", "MSFT"}


def test_deepdive_skips_blank_tickers():
    page = _make_page_with_payload(PAYLOAD)
    client = _FakeSheetsClient()
    result = deepdive_top_n(
        page, cast(SheetsClient, client), _config(),
        tickers=["AAPL", "", "  "], fetched_at=iso_now(),
    )
    # Only the non-blank ticker is processed.
    assert len(result.successes) == 1
    assert result.successes[0].ticker == "AAPL"


def test_deepdive_handles_empty_ticker_list():
    page = _make_page_with_payload(PAYLOAD)
    client = _FakeSheetsClient()
    result = deepdive_top_n(
        page, cast(SheetsClient, client), _config(),
        tickers=[], fetched_at=iso_now(),
    )
    assert result.successes == []
    assert result.failures == []
    # No AGENT_LOG writes when there's nothing to report.
    append_calls = [c for c in client.calls if c[0] == "append_rows"]
    assert append_calls == []
