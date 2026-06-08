#!/usr/bin/env python3
"""scrape_investingpro.py — log in to InvestingPro, scrape AI Picks, deep-dive.

Card 2 of the PortfolioMind v4 build. This is the daily entry point the
operator (or card 4's cron job) invokes to refresh the RAW_PICKS tab.

Usage:
    uv run python scripts/scrape_investingpro.py --headless --limit 5
    uv run python scripts/scrape_investingpro.py --headless --limit 10 --no-deepdive
    uv run python scripts/scrape_investingpro.py --no-headless --limit 3  # debug only

Behavior:
1. Loads env from the active Hermes profile (``$HERMES_PROFILE`` or
   ``portfoliomind`` if unset) so the InvestingPro credentials and
   Google SA JSON are available even though the foundation's
   ``load_env_sources()`` defaults to the ``builder`` profile.
2. Builds a Google Sheets client and verifies the configured sheet has
   the RAW_PICKS tab. If ``GOOGLE_SHEET_ID`` is blank, the script
   bootstraps a new sheet (matching card 1's contract) and uses it.
3. Opens a persistent Chromium context to ``$SESSION_DIR`` and signs
   in to InvestingPro. Login failure -> screenshot + exit 4.
4. Navigates to ``/pro/propicks``, waits for the table to render, and
   appends up to ``--limit`` new rows to RAW_PICKS, applying the
   Ticker+Scraped-At dedup. Re-running the same command is a no-op.
5. Unless ``--no-deepdive`` is set, visits the top-N tickers' deep-dive
   pages and emits one AGENT_LOG row per ticker with the fundamentals.

Exit codes (matches the foundation scripts' convention):
    0  success
    1  usage error
    2  config error (missing env)
    3  Sheets client init failed
    4  InvestingPro login failed (screenshot written)
    5  AI Picks scrape failed
    6  Deep-dive partially failed (some tickers OK, some failed; data
       still on the sheet, summary in AGENT_LOG)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Ensure the active Hermes profile's .env is loaded BEFORE the
# foundation's ``load_env_sources()`` runs.
#
# Hermes has TWO profile locations in play:
#   1. The "real" location at ``$HOME/profiles/<name>/`` (where the
#      operator maintains profiles in this image).
#   2. The "legacy" location at ``$HOME/.hermes/profiles/<name>/`` that
#      the foundation's ``load_env_sources()`` defaults to.
#
# We probe both and use whichever is present. The legacy location
# wins on tie (matches the foundation's behaviour for non-portfoliomind
# profiles that the foundation itself was tested with).
_PROFILE = os.environ.get("HERMES_PROFILE", "portfoliomind")
_candidate_envs = (
    Path.home() / "profiles" / _PROFILE / ".env",  # real
    Path.home() / ".hermes" / "profiles" / _PROFILE / ".env",  # legacy
)
_PROFILE_ENV: Optional[Path] = next(
    (p for p in _candidate_envs if p.is_file()),
    None,
)
if _PROFILE_ENV is not None:
    from dotenv import dotenv_values

    for k, v in dotenv_values(_PROFILE_ENV).items():
        if v is None:
            continue
        # The hermes session may have already loaded the env, possibly
        # to an empty value (e.g. the foundation's default profile
        # pointed at ``builder`` which doesn't have these vars). We
        # always overwrite with the value from the active profile, but
        # never stomp a non-empty value already in the process env
        # (that's how a cron-style override is honoured).
        if not os.environ.get(k):
            os.environ[k] = v

from portfoliomind.config import ConfigError, PortfoliomindConfig  # noqa: E402
from portfoliomind.investingpro.deepdive import (  # noqa: E402
    DeepDiveBatchResult,
    deepdive_top_n,
)
from portfoliomind.investingpro.login import (  # noqa: E402
    InvestingProLoginError,
    LoginResult,
    login,
)
from portfoliomind.investingpro.scrape import (  # noqa: E402
    InvestingProScrapeError,
    scrape_ai_picks,
)
from portfoliomind.logging_setup import get_logger, setup_logging  # noqa: E402
from portfoliomind.sheets.bootstrap import bootstrap_sheet  # noqa: E402
from portfoliomind.sheets.client import SheetsClient, SheetsClientError  # noqa: E402
from portfoliomind.time_utils import iso_now  # noqa: E402

log = get_logger(__name__)


# --- CLI plumbing -----------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape InvestingPro AI Picks + deep-dive fundamentals.",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run Chromium headless (default: true).",
    )
    p.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run Chromium with a visible window (debug only).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of AI Picks to scrape (default: 10).",
    )
    p.add_argument(
        "--no-deepdive",
        action="store_true",
        default=False,
        help="Skip the deep-dive pass; only append to RAW_PICKS.",
    )
    p.add_argument(
        "--deepdive-limit",
        type=int,
        default=None,
        help="Override the deep-dive batch size. Defaults to --limit.",
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


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging(args.log_level)

    log.info(
        "scrape.start headless=%s limit=%s deepdive=%s profile_env=%s",
        args.headless,
        args.limit,
        not args.no_deepdive,
        str(_PROFILE_ENV) if _PROFILE_ENV is not None else "<missing>",
    )

    # 1) Config
    try:
        config = PortfoliomindConfig.from_env()
    except ConfigError as e:
        log.error("config_error: %s", e)
        return 2

    if args.sheet_id:
        from dataclasses import replace

        config = replace(config, google_sheet_id=args.sheet_id)

    # 2) Sheets client
    try:
        client = SheetsClient.from_config(config)
    except Exception as e:  # SA parse / google-auth errors land here
        log.error("sheets_client_init_failed: %s", e)
        return 3

    # 3) Bootstrap if needed. If GOOGLE_SHEET_ID is blank, create a
    #    fresh sheet, ensure all 11 tabs, and use it.
    try:
        if not config.has_existing_sheet():
            log.info("bootstrap.start reason=no_existing_sheet")
            sheet_id, sheet_url = bootstrap_sheet(client, config)
            log.info(
                "bootstrap.created sheet_id=%s url=%s",
                sheet_id,
                sheet_url,
            )
            from dataclasses import replace

            config = replace(config, google_sheet_id=sheet_id)
        else:
            # Sheet given; still bootstrap to be idempotent (no-op if
            # all tabs are already present).
            bootstrap_sheet(client, config)
    except SheetsClientError as e:
        log.error("bootstrap_failed: %s", e)
        return 3

    # 4) Login + scrape + deep-dive. The Playwright context is held
    #    for the duration of both passes; we close it on every exit
    #    path via try/finally.
    scrape_ts = iso_now()
    login_result: Optional[LoginResult] = None
    try:
        try:
            login_result = login(config, headless=args.headless)
        except InvestingProLoginError as e:
            log.error("login_failed: %s", e)
            return 4

        try:
            result = scrape_ai_picks(
                login_result.page,
                client,
                config,
                limit=args.limit,
                scraped_at=scrape_ts,
            )
        except InvestingProScrapeError as e:
            log.error("scrape_failed: %s", e)
            return 5

        log.info(
            "scrape.summary picks=%d fresh=%d skipped=%d first_row=%s ts=%s",
            len(result.picks),
            len(result.new_rows),
            result.skipped_duplicates,
            result.sheet_first_row,
            scrape_ts,
        )

        if args.no_deepdive:
            return 0

        # 5) Deep-dive the top N tickers. We feed the picks we just
        #    wrote (in the order they appeared on the AI Picks page).
        dd_limit = args.deepdive_limit if args.deepdive_limit is not None else args.limit
        dd_tickers = [p.ticker for p in result.picks[:dd_limit]]
        if not dd_tickers:
            log.info("deepdive.skip reason=no_picks")
            return 0
        try:
            dd_result: DeepDiveBatchResult = deepdive_top_n(
                login_result.page,
                client,
                config,
                dd_tickers,
                fetched_at=scrape_ts,
            )
        except Exception as e:
            # deepdive_top_n is supposed to swallow per-ticker errors,
            # so any exception here is a structural problem. Log and
            # return partial success (6).
            log.error("deepdive_unexpected: %s", type(e).__name__)
            return 6

        log.info(
            "deepdive.summary ok=%d failed=%d ts=%s",
            len(dd_result.successes),
            len(dd_result.failures),
            scrape_ts,
        )
        if dd_result.failures:
            return 6
        return 0
    finally:
        if login_result is not None:
            try:
                login_result.context.close()
            except Exception as e:  # best-effort cleanup
                log.debug("context_close_failed: %s", type(e).__name__)


if __name__ == "__main__":
    sys.exit(main())
