"""Bootstrap a Google Sheet for the PortfolioMind report.

This is the one-shot setup that card 1 ships. It creates a sheet if
``GOOGLE_SHEET_ID`` is blank, then ensures all 11 tabs exist with the right
column headers.

The contract for the rest of the agent:

    sid, url = bootstrap_sheet(sheets_client, config)
    # -> sheet is ready; cards 2/3/4 can now read + write to all 11 tabs.

Idempotency: re-running with the same sheet ID will not duplicate tabs, and
will not rewrite headers if they already match. If the sheet is missing
tabs, missing tabs are added; mismatched headers are rewritten with a
WARNING-level log.
"""

from __future__ import annotations

from typing import Final, Tuple

from ..config import PortfoliomindConfig
from ..logging_setup import get_logger
from ..time_utils import iso_now
from .client import SheetsClient, SheetsClientError
from .schema import TAB_HEADERS, TAB_NAMES

log = get_logger(__name__)


SHEET_TITLE_PREFIX: Final[str] = "PortfolioMind Report"


def sheet_title_for_today() -> str:
    """``PortfolioMind Report — 2026-06-08`` style title."""
    return f"{SHEET_TITLE_PREFIX} — {iso_now()[:10]}"


def bootstrap_sheet(
    client: SheetsClient, config: PortfoliomindConfig
) -> Tuple[str, str]:
    """Create-or-verify the Google Sheet + all 11 tabs.

    Returns ``(spreadsheet_id, spreadsheet_url)``.

    Raises :class:`SheetsClientError` if the configured sheet ID is set but
    unreachable (e.g. wrong permissions, sheet deleted).
    """
    if config.has_existing_sheet():
        sheet_id = config.google_sheet_id
        log.info("bootstrap_sheet using existing sheet_id=%s", sheet_id)
        # Verify it's reachable -- a 404/403 here should surface immediately
        # rather than mid-run when card 2/3/4 first tries to write.
        try:
            client.get_sheet(sheet_id)
        except SheetsClientError as e:
            raise SheetsClientError(
                f"Configured GOOGLE_SHEET_ID={sheet_id} is not reachable: {e}"
            ) from e
    else:
        title = sheet_title_for_today()
        log.info("bootstrap_sheet creating new sheet title=%s", title)
        sheet_id, sheet_url = client.create_spreadsheet(title)
        # Remove the default "Sheet1" that Google auto-creates so the
        # workbook opens on a meaningful tab.
        client.delete_worksheet(sheet_id, "Sheet1")
    # Note: in the existing-sheet path we don't have sheet_url; rebuild it.
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"

    # Now ensure all 11 tabs exist with correct headers.
    for tab in TAB_NAMES:
        headers = TAB_HEADERS[tab]
        try:
            client.ensure_worksheet(sheet_id, tab, headers)
        except SheetsClientError:
            # Already wrapped with the tab name by the client; just re-raise.
            raise

    log.info("bootstrap_sheet done sheet_id=%s tabs=%d", sheet_id, len(TAB_NAMES))
    return sheet_id, sheet_url


__all__ = ["SHEET_TITLE_PREFIX", "sheet_title_for_today", "bootstrap_sheet"]
