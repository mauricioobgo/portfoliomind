"""Idempotency tests for the Sheets bootstrap path.

These tests do NOT hit a real Google Sheet. They build a fake
``SheetsClient`` that records every call and simulates the Sheets API
responses in-memory, then verify the bootstrap is idempotent.
"""

from __future__ import annotations

from typing import Any

import pytest

from portfoliomind.config import PortfoliomindConfig
from portfoliomind.sheets.bootstrap import bootstrap_sheet, sheet_title_for_today
from portfoliomind.sheets.client import SheetsClient, SheetsClientError
from portfoliomind.sheets.schema import TAB_HEADERS, TAB_NAMES

from .conftest import full_env


# --- In-memory fake of googleapiclient --------------------------------------


class _FakeValuesAPI:
    def __init__(self, store: "_FakeStore", sheet_id: str, tab: str):
        self._store = store
        self._sheet_id = sheet_id
        self._tab = tab

    def get(self, *, spreadsheetId, range, valueRenderOption=None):
        # Return a minimal response-like object.
        from unittest.mock import MagicMock

        cells = self._store.snapshot_tab(spreadsheetId, self._tab)
        # Parse "A1:D10" or "'Tab'!A1:D10"
        cells_filtered = self._filter_range(cells, range)
        resp = MagicMock()
        resp.execute.return_value = {"values": cells_filtered}
        return resp

    def update(self, *, spreadsheetId, range, body, valueInputOption=None):
        from unittest.mock import MagicMock

        # Overwrite the range with the body.
        self._store.write_range(spreadsheetId, self._tab, range, body.get("values", []))
        resp = MagicMock()
        resp.execute.return_value = {"updatedRange": range, "updatedRows": len(body.get("values", []))}
        return resp

    @staticmethod
    def _filter_range(cells: list[list[str]], range_str: str) -> list[list[str]]:
        # Strip the tab prefix.
        if "!" in range_str:
            range_str = range_str.split("!", 1)[1]
        # Handle A:A (full column) and A1:D10 (rect).
        import re

        m = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", range_str)
        if m:
            col1, row1, col2, row2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            c1 = _FakeValuesAPI._col_to_index(col1)
            c2 = _FakeValuesAPI._col_to_index(col2)
            return [r[c1 - 1 : c2] for r in cells[row1 - 1 : row2] if r]
        m = re.match(r"([A-Z]+):([A-Z]+)", range_str)
        if m:
            c1 = _FakeValuesAPI._col_to_index(m.group(1))
            c2 = _FakeValuesAPI._col_to_index(m.group(2))
            return [r[c1 - 1 : c2] for r in cells if r]
        m = re.match(r"([A-Z]+)(\d+):([A-Z]+)", range_str)
        if m:
            c1 = _FakeValuesAPI._col_to_index(m.group(1))
            row1 = int(m.group(2))
            c2 = _FakeValuesAPI._col_to_index(m.group(3))
            return [r[c1 - 1 : c2] for r in cells[row1 - 1 :] if r]
        m = re.match(r"([A-Z]+):([A-Z]+)", range_str)
        return list(cells)

    @staticmethod
    def _col_to_index(letter: str) -> int:
        n = 0
        for c in letter:
            n = n * 26 + (ord(c) - ord("A") + 1)
        return n


class _FakeWorksheetHandle:
    """Mimics the ``spreadsheets()`` resource returned by google-api-python-client."""

    def __init__(self, store: "_FakeStore", sheet_id: str):
        self._store = store
        self._sheet_id = sheet_id

    def get(self, *, spreadsheetId, fields=None):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.execute.return_value = self._store.get_sheet(spreadsheetId)
        return resp

    def create(self, *, body, fields=None):
        from unittest.mock import MagicMock

        resp = MagicMock()
        new_id, url = self._store.create_sheet(body["properties"]["title"])
        resp.execute.return_value = {
            "spreadsheetId": new_id,
            "spreadsheetUrl": url,
        }
        return resp

    def batchUpdate(self, *, spreadsheetId, body):
        from unittest.mock import MagicMock

        for req in body.get("requests", []):
            if "addSheet" in req:
                title = req["addSheet"]["properties"]["title"]
                self._store.add_worksheet(spreadsheetId, title)
            elif "deleteSheet" in req:
                sheetId = int(req["deleteSheet"]["sheetId"])
                self._store.delete_worksheet_by_id(spreadsheetId, sheetId)
            elif "updateSheetProperties" in req:
                # best-effort; ignored in tests
                pass
        resp = MagicMock()
        resp.execute.return_value = {"replies": [{}] * len(body.get("requests", []))}
        return resp

    def values(self):
        # Return a fresh per-tab handle. The fake picks the right one based
        # on the range passed in, so we can return a single handle and
        # dispatch inside it.
        return _FakeValuesAPI(self._store, self._sheet_id, tab="<auto>")


class _FakeStore:
    """In-memory state for the fake Sheets backend."""

    def __init__(self):
        # sheet_id -> {"title": str, "tabs": {tab_name -> [["row1col1", ...], ...]},
        #              "props": [{"sheetId": int, "title": str, "gridProperties": {...}}, ...]}
        self.sheets: dict[str, dict[str, Any]] = {}
        self.next_sheet_id = 100
        self.next_tab_id = 1

    def create_sheet(self, title: str) -> tuple[str, str]:
        sid = f"sheet_{self.next_sheet_id}"
        self.next_sheet_id += 1
        self.sheets[sid] = {
            "title": title,
            "tabs": {"Sheet1": []},
            "props": [{"sheetId": 0, "title": "Sheet1",
                        "gridProperties": {"rowCount": 1000, "columnCount": 26}}],
        }
        self.next_tab_id = 1
        return sid, f"https://docs.google.com/spreadsheets/d/{sid}/edit"

    def get_sheet(self, sheet_id: str) -> dict[str, Any]:
        if sheet_id not in self.sheets:
            raise SheetsClientError(f"fake: sheet {sheet_id} not found")
        sheet = self.sheets[sheet_id]
        return {
            "spreadsheetId": sheet_id,
            "properties": {"title": sheet["title"]},
            "sheets": [{"properties": p} for p in sheet["props"]],
        }

    def add_worksheet(self, sheet_id: str, title: str) -> None:
        sheet = self.sheets[sheet_id]
        if title in sheet["tabs"]:
            return  # already exists; addSheet would 400
        new_id = self.next_tab_id
        self.next_tab_id += 1
        sheet["tabs"][title] = []
        sheet["props"].append({
            "sheetId": new_id,
            "title": title,
            "gridProperties": {"rowCount": 1000, "columnCount": 26, "frozenRowCount": 1},
        })

    def delete_worksheet_by_id(self, sheet_id: str, ws_id: int) -> None:
        sheet = self.sheets[sheet_id]
        for p in list(sheet["props"]):
            if int(p["sheetId"]) == ws_id:
                sheet["tabs"].pop(p["title"], None)
                sheet["props"].remove(p)
                return

    def snapshot_tab(self, sheet_id: str, tab: str) -> list[list[str]]:
        sheet = self.sheets[sheet_id]
        if tab == "<auto>":
            # Not used in tests; we dispatch inside _FakeValuesAPI.
            return []
        return list(sheet["tabs"].get(tab, []))

    def write_range(self, sheet_id: str, tab: str, range_str: str, values: list[list[str]]) -> None:
        sheet = self.sheets[sheet_id]
        # Resolve the tab from the range if tab=="<auto>"
        if tab == "<auto>":
            if "!" in range_str:
                tab = range_str.split("!", 1)[0].strip("'")
            else:
                tab = list(sheet["tabs"].keys())[0]
        if tab not in sheet["tabs"]:
            sheet["tabs"][tab] = []
        cells = sheet["tabs"][tab]
        # Parse A1:Zn
        import re
        m = re.match(r"(?:'[^']+'!)?([A-Z]+)(\d+):([A-Z]+)(\d+)", range_str)
        if not m:
            return
        c1 = _FakeValuesAPI._col_to_index(m.group(1))
        row1 = int(m.group(2))
        c2 = _FakeValuesAPI._col_to_index(m.group(3))
        row2 = int(m.group(4))
        # Pad rows out to row2.
        while len(cells) < row2:
            cells.append([])
        for ri, val_row in enumerate(values):
            row = cells[row1 - 1 + ri]
            while len(row) < c2:
                row.append("")
            for ci, v in enumerate(val_row):
                row[c1 - 1 + ci] = str(v) if v is not None else ""


class _FakeService:
    """Stand-in for the googleapiclient.discovery build() result."""

    def __init__(self, store: _FakeStore):
        self._store = store

    def spreadsheets(self):
        return _FakeWorksheetHandle(self._store, sheet_id="<auto>")


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def fake_store():
    return _FakeStore()


@pytest.fixture
def fake_service(monkeypatch, fake_store):
    """Patch the SheetsClient constructor to use our fake backend."""
    service = _FakeService(fake_store)

    def _fake_build(*args, **kwargs):
        return service

    monkeypatch.setattr("portfoliomind.sheets.client.build", _fake_build)
    return service


@pytest.fixture
def config_blank_sheet():
    return PortfoliomindConfig.from_env(env=full_env(sheet_id=""))


@pytest.fixture
def config_existing_sheet():
    return PortfoliomindConfig.from_env(env=full_env(sheet_id="existing-sheet-id-xyz"))


# --- Tests -------------------------------------------------------------------


def test_bootstrap_creates_sheet_when_id_blank(fake_service, fake_store, config_blank_sheet):
    client = SheetsClient.from_config(config_blank_sheet)
    sid, url = bootstrap_sheet(client, config_blank_sheet)
    assert sid.startswith("sheet_")
    assert url.startswith("https://docs.google.com/")
    assert "PortfolioMind Report" in fake_store.sheets[sid]["title"]
    # All 12 tabs should be present.
    sheet = fake_store.sheets[sid]
    for tab in TAB_NAMES:
        assert tab in sheet["tabs"], f"tab {tab!r} missing after bootstrap"
        # Header row should match.
        assert sheet["tabs"][tab][0] == TAB_HEADERS[tab], f"headers wrong for {tab!r}"
    # Default Sheet1 should be deleted.
    assert "Sheet1" not in sheet["tabs"]


def test_bootstrap_idempotent_re_runs_no_duplicates(fake_service, fake_store, config_blank_sheet):
    client = SheetsClient.from_config(config_blank_sheet)
    sid1, _ = bootstrap_sheet(client, config_blank_sheet)
    sheet = fake_store.sheets[sid1]
    tab_count_after_first = len(sheet["props"])
    # Re-run: the sheet ID is now in config (simulate by re-using the same fake store).
    # Rebuild config with the discovered sheet id.
    from dataclasses import replace
    cfg2 = replace(config_blank_sheet, google_sheet_id=sid1)
    sid2, url2 = bootstrap_sheet(client, cfg2)
    assert sid2 == sid1, "second bootstrap should NOT create a new sheet"
    # No duplicate tabs.
    assert len(sheet["props"]) == tab_count_after_first, (
        f"second bootstrap duplicated tabs: was {tab_count_after_first}, "
        f"now {len(sheet['props'])}"
    )
    # Headers still match (no rewrite).
    for tab in TAB_NAMES:
        assert sheet["tabs"][tab][0] == TAB_HEADERS[tab]


def test_bootstrap_adds_missing_tabs_only(fake_service, fake_store, config_existing_sheet):
    """If the sheet exists but only has 3 of the 12 tabs, bootstrap adds the other 9."""
    sid = config_existing_sheet.google_sheet_id
    # Seed the fake store with a sheet that has 3 tabs.
    fake_store.sheets[sid] = {
        "title": "Pre-existing Sheet",
        "tabs": {t: [TAB_HEADERS[t]] for t in TAB_NAMES[:3]},
        "props": [
            {"sheetId": i + 1, "title": t,
             "gridProperties": {"rowCount": 1000, "columnCount": 26, "frozenRowCount": 1}}
            for i, t in enumerate(TAB_NAMES[:3])
        ],
    }
    fake_store.next_tab_id = 100  # so new tabs get fresh ids
    client = SheetsClient.from_config(config_existing_sheet)
    bootstrap_sheet(client, config_existing_sheet)
    sheet = fake_store.sheets[sid]
    # All 12 tabs should now be present.
    for tab in TAB_NAMES:
        assert tab in sheet["tabs"], f"tab {tab!r} missing"
        assert sheet["tabs"][tab][0] == TAB_HEADERS[tab]
    # Exactly one prop per tab.
    assert len(sheet["props"]) == len(TAB_NAMES)


def test_bootstrap_unreachable_sheet_raises(fake_service, fake_store, config_existing_sheet):
    """If GOOGLE_SHEET_ID points at a sheet that does not exist, raise clearly."""
    bad_id = "definitely-not-a-real-sheet-id"
    from dataclasses import replace
    cfg = replace(config_existing_sheet, google_sheet_id=bad_id)
    client = SheetsClient.from_config(cfg)
    with pytest.raises(SheetsClientError, match="not reachable"):
        bootstrap_sheet(client, cfg)


def test_sheet_title_for_today_format():
    title = sheet_title_for_today()
    assert title.startswith("PortfolioMind Report — ")
    date_part = title.split("— ", 1)[1]
    # YYYY-MM-DD
    assert len(date_part) == 10
    assert date_part[4] == "-" and date_part[7] == "-"
