#!/usr/bin/env python3
"""bootstrap_sheet.py — create or verify the Google Sheet + 11 tabs.

Usage:
    uv run python scripts/bootstrap_sheet.py
    uv run python scripts/bootstrap_sheet.py --sheet-id <id>

Behavior:
* If ``GOOGLE_SHEET_ID`` is set, verify the sheet is reachable and ensure all
  11 tabs are present with the correct headers. Does NOT create a new sheet.
* If ``GOOGLE_SHEET_ID`` is empty, create a fresh sheet titled
  ``PortfolioMind Report — YYYY-MM-DD``, populate all 11 tabs, and print
  ``(spreadsheet_id, spreadsheet_url)``.

This script is idempotent. Re-running it on the same sheet will not duplicate
tabs and will not rewrite headers that already match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script as a standalone file (no package install needed
# during early dev). uv adds src/ to PYTHONPATH via the [tool.pytest]
# config, but scripts need it too.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from portfoliomind.config import ConfigError, PortfoliomindConfig  # noqa: E402
from portfoliomind.logging_setup import get_logger, setup_logging  # noqa: E402
from portfoliomind.sheets.bootstrap import bootstrap_sheet  # noqa: E402
from portfoliomind.sheets.client import SheetsClient, SheetsClientError  # noqa: E402
from portfoliomind.sheets.schema import TAB_NAMES  # noqa: E402

log = get_logger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create or verify the PortfolioMind Google Sheet + 11 tabs."
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging(args.log_level)

    # 1) Load config. We do NOT modify GOOGLE_SHEET_ID in the env permanently;
    #    if --sheet-id was passed, we override just for this run.
    try:
        config = PortfoliomindConfig.from_env()
    except ConfigError as e:
        log.error("config_error: %s", e)
        return 2

    if args.sheet_id:
        # Rebuild a config with the override. Easiest: a fresh dataclass.
        from dataclasses import replace

        config = replace(config, google_sheet_id=args.sheet_id)

    # 2) Build the Sheets client.
    try:
        client = SheetsClient.from_config(config)
    except Exception as e:
        log.error("client_init_failed: %s", e)
        return 3

    # 3) Bootstrap.
    try:
        sheet_id, sheet_url = bootstrap_sheet(client, config)
    except SheetsClientError as e:
        log.error("bootstrap_failed: %s", e)
        return 4

    # 4) Report.
    print(f"sheet_id:    {sheet_id}")
    print(f"sheet_url:   {sheet_url}")
    print(f"tabs:        {len(TAB_NAMES)} (verified present)")
    for t in TAB_NAMES:
        print(f"  - {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
