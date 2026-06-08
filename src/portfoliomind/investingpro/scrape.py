"""InvestingPro AI Picks scrape.

Card 2 of the PortfolioMind v4 build. Orchestrates the post-login flow:

1. Navigate to ``https://www.investing.com/pro/propicks`` (the AI Picks
   page).
2. Wait for the results table to render (60s budget).
3. Read each ``<tr>``'s cells, hand them to
   :func:`portfoliomind.investingpro.parse.parse_ai_picks_table`.
4. Convert each :class:`RawPick` to a 9-cell row and append to
   :data:`portfoliomind.sheets.schema.RAW_PICKS`, applying the
   Ticker + Scraped At dedup contract.

The function takes an already-open :class:`LoginResult` (from
:mod:`portfoliomind.investingpro.login`) so the same context can be
shared with the deep-dive module without re-logging in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from ..config import PortfoliomindConfig
from ..logging_setup import get_logger
from ..sheets.client import SheetsClient, SheetsClientError
from ..sheets.schema import RAW_PICKS, TAB_HEADERS
from .login import _AI_PICKS_URL  # noqa: F401  (re-exported for downstream use)
from .parse import (
    RAW_PICKS_WIDTH,
    RawPick,
    filter_new_rows,
    parse_ai_picks_table,
)

log = get_logger(__name__)

#: How long to wait for the AI Picks results table to render after we
#: navigate to the URL. InvestingPro serves the shell quickly but the
#: async data fetch can take a while on a cold session.
AI_PICKS_RENDER_TIMEOUT_S = 60

# Selector candidates. The ProPicks page renders the table as a series
# of ``<tr>`` inside a known container. We try a list of plausible
# selectors; the first that yields >0 rows wins.
_ROW_SELECTORS = (
    "table tbody tr",
    "[data-test='propicks-table'] tbody tr",
    ".propicks-table tbody tr",
    "div.proPicksTblContainer table tbody tr",
)
_TABLE_SELECTORS = (
    "table.propicks",
    "[data-test='propicks-table']",
    "table.tbl",
    "table",
)


# --- Exceptions -------------------------------------------------------------


class InvestingProScrapeError(RuntimeError):
    """Raised when the AI Picks scrape cannot be completed."""


# --- Result types -----------------------------------------------------------


@dataclass
class ScrapeResult:
    """What the scrape produced and what it did with it."""

    picks: list[RawPick]
    new_rows: list[list[str]]
    skipped_duplicates: int
    sheet_first_row: int  # 0 if no rows appended (nothing to write)


# --- Public API -------------------------------------------------------------


def scrape_ai_picks(
    page: Page,
    sheets: SheetsClient,
    config: PortfoliomindConfig,
    *,
    limit: int = 10,
    scraped_at: Optional[str] = None,
) -> ScrapeResult:
    """Navigate to AI Picks, parse the table, append new rows to the sheet.

    Parameters
    ----------
    page:
        A live Playwright ``Page`` that has already cleared the InvestingPro
        login gate (see :mod:`portfoliomind.investingpro.login`).
    sheets:
        A :class:`SheetsClient` for the target sheet.
    config:
        The PortfolioMind env-driven config. ``google_sheet_id`` is
        required (the bootstrap path is handled by the caller; this
        function assumes the sheet already exists).
    limit:
        Maximum number of picks to scrape. The InvestingPro page may
        render more rows than this; we take the first ``limit``.
    scraped_at:
        Override the timestamp used for the dedup key. Defaults to
        :func:`iso_now`. Tests pass an explicit value so the dedup
        contract is reproducible.

    Returns
    -------
    ScrapeResult

    Raises
    ------
    InvestingProScrapeError
        If the page can't be reached, the table never renders, or the
        sheet write fails.
    """
    if not config.google_sheet_id:
        raise InvestingProScrapeError(
            "scrape_ai_picks requires a non-empty GOOGLE_SHEET_ID; "
            "run bootstrap first"
        )
    if limit <= 0:
        raise ValueError(f"limit must be > 0, got {limit}")

    _navigate_to_ai_picks(page)
    rows = _read_table_rows(page, limit=limit)
    picks = parse_ai_picks_table(rows, scraped_at=scraped_at)
    log.info("investingpro.scrape.parsed count=%d limit=%d", len(picks), limit)

    if not picks:
        return ScrapeResult(
            picks=[],
            new_rows=[],
            skipped_duplicates=0,
            sheet_first_row=0,
        )

    # Ensure the worksheet exists with the canonical headers. The
    # bootstrap usually does this, but a card 1 bootstrap that didn't
    # run (e.g. sheet was hand-created) would otherwise produce a
    # confusing append error.
    sheets.ensure_worksheet(
        config.google_sheet_id, RAW_PICKS, TAB_HEADERS[RAW_PICKS]
    )

    new_rows = [p.to_row(scraped_at=scraped_at) for p in picks]
    # Enforce shape: every row must be exactly RAW_PICKS_WIDTH cells.
    for r in new_rows:
        if len(r) != RAW_PICKS_WIDTH:
            raise InvestingProScrapeError(
                f"Refusing to append malformed row (len={len(r)}, "
                f"expected={RAW_PICKS_WIDTH}): {r!r}"
            )

    # Dedup against what is already on the sheet.
    try:
        existing = sheets.read_range(
            config.google_sheet_id, RAW_PICKS, "A2:I"
        )
    except SheetsClientError as e:
        raise InvestingProScrapeError(
            f"Failed to read existing RAW_PICKS rows: {e}"
        ) from e

    # Pad existing rows to the canonical width so dedup doesn't KeyError
    # on short rows.
    existing_padded = [r + [""] * (RAW_PICKS_WIDTH - len(r)) for r in existing]
    fresh = filter_new_rows(new_rows, existing_padded)
    skipped = len(new_rows) - len(fresh)
    log.info(
        "investingpro.scrape.dedup parsed=%d fresh=%d skipped=%d",
        len(new_rows),
        len(fresh),
        skipped,
    )

    if not fresh:
        return ScrapeResult(
            picks=picks,
            new_rows=[],
            skipped_duplicates=skipped,
            sheet_first_row=0,
        )

    try:
        first_row = sheets.append_rows(config.google_sheet_id, RAW_PICKS, fresh)
    except SheetsClientError as e:
        raise InvestingProScrapeError(
            f"Failed to append {len(fresh)} rows to {RAW_PICKS!r}: {e}"
        ) from e

    log.info(
        "investingpro.scrape.appended count=%d first_row=%s",
        len(fresh),
        first_row,
    )
    return ScrapeResult(
        picks=picks,
        new_rows=fresh,
        skipped_duplicates=skipped,
        sheet_first_row=first_row,
    )


# --- Internals --------------------------------------------------------------


def _navigate_to_ai_picks(page: Page) -> None:
    """Navigate to the AI Picks URL and wait for the URL to settle.

    We do NOT wait for a specific table selector here — the table render
    is async and we want a fresh ``read_table_rows`` to do the waiting
    with the explicit 60s budget.
    """
    log.info("investingpro.scrape.navigate url=%s", _AI_PICKS_URL)
    try:
        page.goto(_AI_PICKS_URL, timeout=30_000, wait_until="domcontentloaded")
    except PlaywrightTimeoutError as e:
        raise InvestingProScrapeError(
            f"Navigation to {_AI_PICKS_URL} timed out"
        ) from e


def _read_table_rows(page: Page, *, limit: int) -> list[list[str]]:
    """Wait for the AI Picks table to render and return up to ``limit`` rows.

    Each row is a list of cell text. We try several table-row selectors
    and return the first non-empty hit. If nothing resolves within
    :data:`AI_PICKS_RENDER_TIMEOUT_S`, we raise.
    """
    deadline_ms = AI_PICKS_RENDER_TIMEOUT_S * 1000
    interval_ms = 500
    elapsed = 0
    last_count = 0
    while elapsed < deadline_ms:
        for sel in _ROW_SELECTORS:
            try:
                elements = page.query_selector_all(sel)
            except Exception:
                continue
            if not elements:
                continue
            # Convert to cell text. We assume the InvestingPro layout is
            # flat: one ``<tr>`` per pick, with cells in column order.
            rows: list[list[str]] = []
            for tr in elements:
                cells = []
                for td in tr.query_selector_all("th, td"):
                    text = td.text_content() or ""
                    cells.append(text)
                rows.append(cells)
            if rows:
                if len(rows) != last_count:
                    log.info(
                        "investingpro.scrape.rows selector=%s count=%d",
                        sel,
                        len(rows),
                    )
                    last_count = len(rows)
                return rows[:limit]
        page.wait_for_timeout(interval_ms)
        elapsed += interval_ms

    # Try a "best-effort" diagnostic: confirm the page is at the right URL
    # and that the table container exists. This makes the failure mode
    # actionable instead of a generic timeout.
    container_hits = []
    for sel in _TABLE_SELECTORS:
        try:
            if page.query_selector(sel) is not None:
                container_hits.append(sel)
        except Exception:
            continue
    raise InvestingProScrapeError(
        f"AI Picks table did not render within {AI_PICKS_RENDER_TIMEOUT_S}s "
        f"(page url={page.url!r}, container hits={container_hits})"
    )


__all__ = [
    "InvestingProScrapeError",
    "ScrapeResult",
    "scrape_ai_picks",
]
