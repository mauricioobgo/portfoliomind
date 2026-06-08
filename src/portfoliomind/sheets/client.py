"""Google Sheets client built on top of ``google-api-python-client``.

Why raw google-api-python-client and not gspread?
- Keeps the API surface explicit and debuggable (you can log the exact REST
  call when something goes wrong).
- Avoids a thin-wrapper upgrade treadmill.
- Makes the contract for cards 2/3/4 crystal clear: 5 methods, well-defined
  semantics, idempotent where it matters.

API surface (this is the public contract — cards 2/3/4 will depend on it):

    client = SheetsClient.from_config(config)
    sheet = client.get_sheet()
    ws    = client.ensure_worksheet("📥 Raw Picks", ["Ticker", "Price", ...])
    rows  = client.read_range("📥 Raw Picks", "A1:C10")
    client.write_range("📥 Raw Picks", "A1:C10", [["AAPL", 192.40], ...])
    client.append_rows("📥 Raw Picks", [["AAPL", 192.40], ...])
"""

from __future__ import annotations

from typing import Any, Final, Optional

from google.auth import exceptions as google_auth_exceptions
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..config import PortfoliomindConfig
from ..logging_setup import get_logger

log = get_logger(__name__)

# Read-write scope is sufficient: cards 2/3/4 all need to create worksheets,
# append rows, and read ranges back for verification.
_SCOPES: Final[tuple[str, ...]] = (
    "https://www.googleapis.com/auth/spreadsheets",
)


class SheetsClientError(RuntimeError):
    """Raised on unrecoverable Sheets errors (auth failure, missing sheet, etc.).

    The original ``HttpError`` is chained via ``__cause__`` for debugging.
    """


class SheetsClient:
    """Thin, explicit wrapper around the Sheets v4 REST API.

    Construct via :meth:`from_config` (the standard path) or directly with
    pre-built credentials (useful in tests).
    """

    def __init__(self, service_account_info: dict[str, Any]):
        # We hold the parsed service-account dict (not the raw path) so the
        # client is independent of any filesystem state.
        self._sa_info = service_account_info
        creds = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=list(_SCOPES)
        )
        # ``static_discovery=False`` avoids a network call to Google's
        # discovery doc at import time — handy in CI and offline tests.
        self._service = build("sheets", "v4", credentials=creds, static_discovery=False)
        # The spreadsheet resource is cheap to grab repeatedly, but we cache
        # one for convenience.
        self._spreadsheets = self._service.spreadsheets()

    # --- Construction ---

    @classmethod
    def from_config(cls, config: PortfoliomindConfig) -> "SheetsClient":
        """Build a client from a :class:`PortfoliomindConfig`."""
        return cls(service_account_info=config.google_service_account_info)

    # --- Sheet-level ---

    def get_sheet(self, sheet_id: Optional[str] = None) -> dict[str, Any]:
        """Fetch the full spreadsheet metadata.

        Returns the raw ``spreadsheet`` dict (which contains ``spreadsheetId``,
        ``properties.title``, ``sheets[*].properties``, etc). Use this when
        you need to discover which tabs already exist.

        Raises :class:`SheetsClientError` with the sheet ID and the underlying
        error code on 403/404.
        """
        if not sheet_id:
            raise SheetsClientError("get_sheet requires a sheet_id")
        try:
            return (
                self._spreadsheets.get(spreadsheetId=sheet_id).execute()  # type: ignore[union-attr]
            )
        except google_auth_exceptions.RefreshError as e:
            self._log_auth_error(f"get_sheet({sheet_id})", e)
            raise SheetsClientError(
                f"Failed to authenticate to Google Sheets: {e}"
            ) from e
        except HttpError as e:
            self._log_http_error(e, f"get_sheet({sheet_id})")
            raise SheetsClientError(
                f"Failed to fetch sheet {sheet_id}: HTTP {e.resp.status} {e.reason}"
            ) from e

    def list_worksheets(self, sheet_id: str) -> list[dict[str, Any]]:
        """Return the list of worksheet ``properties`` dicts for a sheet."""
        meta = self.get_sheet(sheet_id)
        return [s["properties"] for s in meta.get("sheets", [])]

    def find_worksheet(
        self, sheet_id: str, title: str
    ) -> Optional[dict[str, Any]]:
        """Return the worksheet ``properties`` for ``title`` or ``None``."""
        for props in self.list_worksheets(sheet_id):
            if props.get("title") == title:
                return props
        return None

    def create_spreadsheet(self, title: str) -> tuple[str, str]:
        """Create a brand-new Google Sheet with the given title.

        Returns ``(spreadsheet_id, spreadsheet_url)``. The new sheet has the
        default ``Sheet1`` tab; the caller should call
        :meth:`ensure_worksheet` for each required tab and then optionally
        delete ``Sheet1``.
        """
        try:
            body = {"properties": {"title": title}}
            resp = (
                self._spreadsheets.create(body=body, fields="spreadsheetId,spreadsheetUrl").execute()  # type: ignore[union-attr]
            )
        except google_auth_exceptions.RefreshError as e:
            # The service account is unreachable / disabled / wrong audience.
            # Wrap so the CLI exits cleanly with our error code.
            self._log_auth_error("create_spreadsheet", e)
            raise SheetsClientError(
                f"Failed to authenticate to Google Sheets: {e}"
            ) from e
        except HttpError as e:
            self._log_http_error(e, f"create_spreadsheet({title})")
            raise SheetsClientError(
                f"Failed to create sheet {title!r}: HTTP {e.resp.status} {e.reason}"
            ) from e
        sid = resp["spreadsheetId"]
        url = resp.get("spreadsheetUrl") or f"https://docs.google.com/spreadsheets/d/{sid}/edit"
        return sid, url

    def delete_worksheet(self, sheet_id: str, title: str) -> bool:
        """Delete a worksheet by title. Returns True if deleted, False if not found."""
        props = self.find_worksheet(sheet_id, title)
        if not props:
            return False
        try:
            (
                self._spreadsheets.batchUpdate(  # type: ignore[union-attr]
                    spreadsheetId=sheet_id,
                    body={
                        "requests": [
                            {"deleteSheet": {"sheetId": int(props["sheetId"])}}
                        ]
                    },
                ).execute()
            )
            return True
        except HttpError as e:
            self._log_http_error(e, f"delete_worksheet({sheet_id}, {title!r})")
            raise SheetsClientError(
                f"Failed to delete worksheet {title!r}: HTTP {e.resp.status} {e.reason}"
            ) from e

    # --- Worksheet-level ---

    def ensure_worksheet(
        self, sheet_id: str, title: str, headers: list[str]
    ) -> dict[str, Any]:
        """Idempotent: create the worksheet if missing, verify/update headers.

        Behavior:
        * If the worksheet does not exist -> create it with the given headers
          in row 1, return its ``properties`` dict.
        * If it exists AND row 1 matches ``headers`` -> return existing
          properties, no rewrite.
        * If it exists AND row 1 is empty -> write headers, return updated
          properties.
        * If it exists AND row 1 has different headers -> **update** row 1
          to match (the spec says headers may be revised in future spec
          versions; this keeps the bootstrap honest).

        The update path is logged at WARNING level because it indicates a
        schema drift that the user should know about.
        """
        existing = self.find_worksheet(sheet_id, title)
        if existing is None:
            return self._create_worksheet(sheet_id, title, headers)

        # Worksheet exists. Inspect row 1.
        current = self.read_range(sheet_id, title, "A1:ZZZ1")
        current_headers = current[0] if current else []

        if current_headers == headers:
            return existing

        if not any(h.strip() for h in current_headers):
            # Empty row -> safe to write.
            self._write_headers(sheet_id, title, headers, row_count=existing.get("gridProperties", {}).get("rowCount", 1000))
            return self.find_worksheet(sheet_id, title) or existing

        # Drift: rewrite row 1 to the new headers.
        log.warning(
            "schema_drift sheet_id=%s worksheet=%s current=%r expected=%r; rewriting row 1",
            sheet_id,
            title,
            current_headers,
            headers,
        )
        self._write_headers(sheet_id, title, headers, row_count=existing.get("gridProperties", {}).get("rowCount", 1000))
        return self.find_worksheet(sheet_id, title) or existing

    def _create_worksheet(
        self, sheet_id: str, title: str, headers: list[str]
    ) -> dict[str, Any]:
        """Create a worksheet, then populate row 1 with the given headers."""
        try:
            (
                self._spreadsheets.batchUpdate(  # type: ignore[union-attr]
                    spreadsheetId=sheet_id,
                    body={
                        "requests": [
                            {
                                "addSheet": {
                                    "properties": {
                                        "title": title,
                                        "gridProperties": {
                                            "rowCount": 1000,
                                            "columnCount": max(len(headers), 26),
                                            "frozenRowCount": 1,
                                        },
                                    }
                                }
                            }
                        ]
                    },
                ).execute()
            )
        except HttpError as e:
            self._log_http_error(e, f"create_worksheet({sheet_id}, {title!r})")
            raise SheetsClientError(
                f"Failed to create worksheet {title!r} in sheet {sheet_id}: "
                f"HTTP {e.resp.status} {e.reason}"
            ) from e
        # Now write the headers.
        self._write_headers(sheet_id, title, headers, row_count=1000)
        props = self.find_worksheet(sheet_id, title)
        if not props:
            raise SheetsClientError(
                f"Worksheet {title!r} disappeared immediately after creation"
            )
        return props

    def _write_headers(
        self, sheet_id: str, title: str, headers: list[str], row_count: int
    ) -> None:
        """Write ``headers`` to row 1, expanding the grid if needed."""
        col_count = max(len(headers), 26)
        # Ensure grid is wide enough — no-op if already wider.
        try:
            (
                self._spreadsheets.batchUpdate(  # type: ignore[union-attr]
                    spreadsheetId=sheet_id,
                    body={
                        "requests": [
                            {
                                "updateSheetProperties": {
                                    "properties": {
                                        "sheetId": self._sheet_id_for_title(sheet_id, title),
                                        "gridProperties": {
                                            "rowCount": max(row_count, 1),
                                            "columnCount": col_count,
                                        },
                                    },
                                    "fields": "gridProperties.rowCount,gridProperties.columnCount",
                                }
                            }
                        ]
                    },
                ).execute()
            )
        except HttpError as e:
            # Updating grid props is best-effort; don't fail the whole op.
            log.debug("grid resize failed (non-fatal): %s", e)

        self.write_range(sheet_id, title, f"A1:{self._col_letter(len(headers))}1", [headers])

    def _sheet_id_for_title(self, sheet_id: str, title: str) -> int:
        props = self.find_worksheet(sheet_id, title)
        if not props:
            raise SheetsClientError(f"Worksheet {title!r} not found in sheet {sheet_id}")
        return int(props["sheetId"])

    # --- Cell-level ---

    def read_range(
        self, sheet_id: str, tab_name: str, range_a1: str
    ) -> list[list[str]]:
        """Read a range as a 2D list of strings.

        ``range_a1`` uses A1 notation including the tab name (``'Tab'!A1:D10``)
        OR just the range (``A1:D10``) if ``tab_name`` is also provided —
        :meth:`read_range` always assembles ``'Tab'!Range`` for you.
        """
        full_range = self._build_range(tab_name, range_a1)
        try:
            result = (
                self._service.spreadsheets()  # type: ignore[union-attr]
                .values()
                .get(spreadsheetId=sheet_id, range=full_range, valueRenderOption="UNFORMATTED_VALUE")
                .execute()
            )
        except HttpError as e:
            self._log_http_error(e, f"read_range({sheet_id}, {full_range!r})")
            raise SheetsClientError(
                f"Failed to read range {full_range!r} from sheet {sheet_id}: "
                f"HTTP {e.resp.status} {e.reason}"
            ) from e
        return result.get("values", [])

    def write_range(
        self,
        sheet_id: str,
        tab_name: str,
        range_a1: str,
        values: list[list[Any]],
    ) -> None:
        """Overwrite a range with the given 2D list of values.

        Note: this *overwrites* — it does not insert rows. Use
        :meth:`append_rows` for that.
        """
        full_range = self._build_range(tab_name, range_a1)
        body = {"values": [["" if v is None else v for v in row] for row in values]}
        try:
            (
                self._service.spreadsheets()  # type: ignore[union-attr]
                .values()
                .update(
                    spreadsheetId=sheet_id,
                    range=full_range,
                    body=body,
                    valueInputOption="USER_ENTERED",
                )
                .execute()
            )
        except HttpError as e:
            self._log_http_error(e, f"write_range({sheet_id}, {full_range!r})")
            raise SheetsClientError(
                f"Failed to write range {full_range!r} to sheet {sheet_id}: "
                f"HTTP {e.resp.status} {e.reason}"
            ) from e

    def append_rows(
        self, sheet_id: str, tab_name: str, values: list[list[Any]]
    ) -> int:
        """Smart-append: find the next empty row and write there.

        Returns the 1-indexed row number where the first row of ``values`` was
        written (useful for logging and for the agent log tab).
        """
        if not values:
            return 0
        # Determine next empty row by reading the full column A.
        col_a = self.read_range(sheet_id, tab_name, "A:A")
        next_row = len(col_a) + 1  # 1-indexed; if A is empty, next_row=1
        end_col = self._col_letter(max(len(r) for r in values))
        range_a1 = f"A{next_row}:{end_col}{next_row + len(values) - 1}"
        self.write_range(sheet_id, tab_name, range_a1, values)
        return next_row

    def row_count(self, sheet_id: str, tab_name: str) -> int:
        """How many data rows (including header) the tab currently has."""
        col_a = self.read_range(sheet_id, tab_name, "A:A")
        return len(col_a)

    # --- Internals ---

    def _log_http_error(self, e: HttpError, op: str) -> None:
        """Log a Sheets HTTP error with enough context to debug without leaking data."""
        status = e.resp.status if e.resp is not None else "?"
        reason = e.reason or "(no reason)"
        # Try to extract a useful error message without echoing user data.
        try:
            content = e.content.decode("utf-8", errors="replace") if e.content else ""
        except Exception:
            content = ""
        log.error(
            "sheets_http_error op=%s status=%s reason=%s content_excerpt=%s",
            op,
            status,
            reason,
            content[:200].replace("\n", " "),
        )

    def _log_auth_error(self, op: str, e: google_auth_exceptions.RefreshError) -> None:
        """Log an auth-refresh error without leaking the credential contents."""
        log.error(
            "sheets_auth_error op=%s error_type=%s",
            op,
            type(e).__name__,
        )

    @staticmethod
    def _build_range(tab_name: str, range_a1: str) -> str:
        """Combine tab name + cell range into a single A1 range string.

        If the caller already included the tab name (e.g. ``'📥 Raw Picks'!A1:D10``),
        we pass it through untouched.
        """
        if "!" in range_a1:
            return range_a1
        # Single-quote tab names that contain spaces or special chars.
        safe_tab = tab_name.replace("'", "''")
        return f"'{safe_tab}'!{range_a1}"

    @staticmethod
    def _col_letter(n: int) -> str:
        """1-indexed column number -> spreadsheet column letter (A, B, ..., Z, AA, AB, ...)."""
        if n < 1:
            raise ValueError(f"Column number must be >= 1, got {n}")
        result = ""
        while n > 0:
            n, rem = divmod(n - 1, 26)
            result = chr(ord("A") + rem) + result
        return result


__all__ = ["SheetsClient", "SheetsClientError"]
