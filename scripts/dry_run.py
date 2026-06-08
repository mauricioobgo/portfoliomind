#!/usr/bin/env python3
"""dry_run.py — exercises the Sheets integration end-to-end with one test row per tab.

Usage:
    uv run python scripts/dry_run.py
    uv run python scripts/dry_run.py --no-bootstrap
    uv run python scripts/dry_run.py --sheet-id <id>
    uv run python scripts/dry_run.py --headless  # for parity with future Playwright scripts

Behavior:
* Loads config, builds a Sheets client, bootstraps/verifies the 11 tabs.
* Writes one timestamped test row to each tab so the user can visually
  confirm structure in the browser.
* Prints a summary table at the end: tab name -> row count after dry-run.

Re-running this script appends another row each time -- intentional. This
proves the ``append_rows`` smart-append works and lets you watch the sheet
fill up over multiple runs.

Exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Make the package importable when running this script standalone.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from portfoliomind.config import ConfigError, PortfoliomindConfig  # noqa: E402
from portfoliomind.logging_setup import get_logger, setup_logging  # noqa: E402
from portfoliomind.sheets.bootstrap import bootstrap_sheet  # noqa: E402
from portfoliomind.sheets.client import SheetsClient, SheetsClientError  # noqa: E402
from portfoliomind.sheets.schema import TAB_HEADERS, TAB_NAMES  # noqa: E402
from portfoliomind.time_utils import iso_now  # noqa: E402

log = get_logger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dry-run: write one test row to each PortfolioMind tab."
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Reserved for future Playwright cards; Sheets work is always server-side.",
    )
    p.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Opposite of --headless (no-op for now).",
    )
    p.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip tab creation/verification; just write the test row.",
    )
    p.add_argument(
        "--sheet-id",
        default=None,
        help="Override GOOGLE_SHEET_ID for this run (must already exist).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args(argv)


def _make_test_row(tab_name: str) -> list[str]:
    """Build a placeholder row of the right length for a given tab.

    The first column is the timestamp (matches the Agent Log convention).
    Remaining columns are short placeholders so the row length matches the
    header count exactly.
    """
    n = len(TAB_HEADERS[tab_name])
    ts = iso_now()
    row = [ts] + [f"dry-run-{i}" for i in range(2, n + 1)]
    # Pad / trim defensively.
    if len(row) < n:
        row += [""] * (n - len(row))
    elif len(row) > n:
        row = row[:n]
    return row


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging(args.log_level)

    # 1) Config
    try:
        config = PortfoliomindConfig.from_env()
    except ConfigError as e:
        log.error("config_error: %s", e)
        return 2
    if args.sheet_id:
        config = replace(config, google_sheet_id=args.sheet_id)

    # 2) Client
    try:
        client = SheetsClient.from_config(config)
    except Exception as e:
        log.error("client_init_failed: %s", e)
        return 3

    # 3) Optionally bootstrap.
    if not args.no_bootstrap:
        try:
            sheet_id, sheet_url = bootstrap_sheet(client, config)
        except SheetsClientError as e:
            log.error("bootstrap_failed: %s", e)
            return 4
    else:
        if not config.has_existing_sheet():
            log.error("--no-bootstrap requires GOOGLE_SHEET_ID (or --sheet-id)")
            return 5
        sheet_id = config.google_sheet_id
        sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
        # Quick reachability check.
        try:
            client.get_sheet(sheet_id)
        except SheetsClientError as e:
            log.error("sheet_unreachable: %s", e)
            return 4

    # 4) Write one test row per tab.
    summary: list[tuple[str, int]] = []
    for tab in TAB_NAMES:
        row = _make_test_row(tab)
        try:
            client.append_rows(sheet_id, tab, [row])
            count = client.row_count(sheet_id, tab)
        except SheetsClientError as e:
            log.error("write_failed tab=%s: %s", tab, e)
            return 6
        summary.append((tab, count))
        log.info("dry_run row written tab=%s row_count=%d", tab, count)

    # 5) Print summary table.
    print(f"sheet_id:  {sheet_id}")
    print(f"sheet_url: {sheet_url}")
    print()
    print(f"{'tab':<30} {'rows after dry-run':>20}")
    print(f"{'-' * 30} {'-' * 20}")
    for tab, count in summary:
        print(f"{tab:<30} {count:>20}")
    print()
    print(f"Total tabs touched: {len(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
