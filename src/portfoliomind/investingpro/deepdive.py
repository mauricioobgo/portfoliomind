"""InvestingPro ticker deep-dive capture.

Card 2 of the PortfolioMind v4 build. For the top-N picks produced by
:mod:`portfoliomind.investingpro.scrape`, fetch the per-ticker deep-dive
page and pull a small set of fundamentals into
:class:`portfoliomind.investingpro.parse.DeepDiveFacts`.

The card spec says "capture fundamentals" but does not pin a sheet tab
for them; the downstream forecast engine (future card) is the consumer.
For card 2 we therefore:

* Capture the data into the dataclass
* Emit one row to ``AGENT_LOG`` per ticker, with the facts JSON-encoded
  into the Message cell — that gives the operator visibility into what
  we scraped, and gives future cards a stable place to read it back.

The deep-dive fetch is best-effort: a failed ticker does NOT abort the
batch. We log + continue. The acceptance criterion for card 2 is "5 rows
in RAW_PICKS" — the deep-dive output is a secondary artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from ..config import PortfoliomindConfig
from ..logging_setup import get_logger
from ..sheets.client import SheetsClient
from ..sheets.schema import AGENT_LOG, TAB_HEADERS
from .parse import DeepDiveFacts, parse_deepdive_payload

log = get_logger(__name__)

#: How long we wait for a deep-dive page to render. InvestingPro's
#: per-ticker page is usually fast; we leave headroom for 2FA
#: refreshes + cloudflare.
DEEPDIVE_TIMEOUT_S = 30

# Selector candidates for the label/value grid on the deep-dive page.
# InvestingPro renders fundamentals in several flavours; we accept any
# ``<dt>/<dd>`` or label/span pair.
_DEEPDIVE_KEY_VALUE_SELECTORS = (
    "dl dt",
    ".key-info dl dt",
    "div.fundamentals div",
)
_DEEPDIVE_VALUE_SELECTORS = (
    "dl dd",
    ".key-info dl dd",
    "div.fundamentals span",
)


# --- Exceptions -------------------------------------------------------------


class InvestingProDeepDiveError(RuntimeError):
    """Raised when the deep-dive flow cannot be completed.

    Treated as recoverable by the caller: a single ticker failing does
    NOT abort the rest of the batch.
    """


# --- Result -----------------------------------------------------------------


@dataclass
class DeepDiveBatchResult:
    """Per-ticker outcomes for the top-N batch."""

    successes: list[DeepDiveFacts]
    failures: list[tuple[str, str]]  # (ticker, reason)


# --- Public API -------------------------------------------------------------


def deepdive_top_n(
    page: Page,
    sheets: SheetsClient,
    config: PortfoliomindConfig,
    tickers: list[str],
    *,
    fetched_at: Optional[str] = None,
) -> DeepDiveBatchResult:
    """Visit the deep-dive page for each ticker and capture fundamentals.

    Parameters
    ----------
    page:
        A live Playwright ``Page`` that has already cleared InvestingPro
        login.
    sheets:
        A :class:`SheetsClient` for the target sheet.
    config:
        The PortfolioMind env-driven config. ``google_sheet_id`` is
        required for the AGENT_LOG write.
    tickers:
        A list of ticker symbols (e.g. ``["AAPL", "MSFT"]``). The order
        is preserved in the output.
    fetched_at:
        Override the timestamp for the deep-dive facts. Defaults to
        :func:`iso_now`. Tests pass an explicit value.
    """
    successes: list[DeepDiveFacts] = []
    failures: list[tuple[str, str]] = []

    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        try:
            facts = _capture_one(page, ticker, fetched_at=fetched_at)
            successes.append(facts)
        except InvestingProDeepDiveError as e:
            log.warning(
                "investingpro.deepdive.failed ticker=%s err=%s",
                ticker,
                type(e).__name__,
            )
            failures.append((ticker, str(e)))

    # Write a structured AGENT_LOG entry per success. Failures get a
    # single combined entry. We always include the dedup-safe key (the
    # ticker's own ticker) in the Message so the agent log can be
    # queried later.
    if config.google_sheet_id and (successes or failures):
        _emit_agent_log(sheets, config, successes, failures, fetched_at=fetched_at)

    return DeepDiveBatchResult(successes=successes, failures=failures)


# --- Internals --------------------------------------------------------------


def _capture_one(
    page: Page, ticker: str, *, fetched_at: Optional[str] = None
) -> DeepDiveFacts:
    """Navigate to the deep-dive page for one ticker and parse it."""
    url = f"https://www.investing.com/pro/{ticker.lower()}"
    log.info("investingpro.deepdive.navigate ticker=%s url=%s", ticker, url)
    try:
        page.goto(url, timeout=DEEPDIVE_TIMEOUT_S * 1000, wait_until="domcontentloaded")
    except PlaywrightTimeoutError as e:
        raise InvestingProDeepDiveError(
            f"navigation timeout for {ticker}: {type(e).__name__}"
        ) from e

    payload = _read_key_value_pairs(page, ticker)
    if not payload:
        raise InvestingProDeepDiveError(
            f"no fundamentals grid found on {ticker!r} page"
        )
    return parse_deepdive_payload(ticker, payload, fetched_at=fetched_at)


def _read_key_value_pairs(page: Page, ticker: str) -> dict[str, str]:
    """Read the fundamentals grid as a ``{label: value}`` dict.

    InvestingPro's per-ticker page is a moving target; we try a few
    well-known structures and accept the first one that produces any
    pairs. If the page is paywalled or the grid is absent, we return an
    empty dict and let the caller decide what to do.
    """
    # Strategy 1: dl/dt/dd pairs
    try:
        dts = page.query_selector_all("dl dt")
        dds = page.query_selector_all("dl dd")
        if dts and dds and len(dts) == len(dds):
            out: dict[str, str] = {}
            for dt, dd in zip(dts, dds, strict=True):
                k = (dt.text_content() or "").strip()
                v = (dd.text_content() or "").strip()
                if k and v:
                    out[k] = v
            if out:
                return out
    except Exception:
        pass

    # Strategy 2: any element with adjacent label/value siblings.
    # The InvestingPro layout sometimes uses <div>label</div><div>value</div>
    # inside a section. We grab all leaf text and pair them up.
    try:
        nodes = page.query_selector_all("div.key-info, div.fundamentals, section.fundamentals")
        for node in nodes:
            text = node.text_content() or ""
            if not text:
                continue
            out = _parse_inline_label_value_text(text)
            if out:
                return out
    except Exception:
        pass

    log.debug("investingpro.deepdive.empty_grid ticker=%s url=%s", ticker, page.url)
    return {}


def _parse_inline_label_value_text(text: str) -> dict[str, str]:
    """Best-effort extraction of ``Label: value`` pairs from a blob of text.

    InvestingPro's deep-dive page sometimes renders the fundamentals as
    a flat string of ``Label: value`` pairs separated by newlines. We
    split on newlines, strip each line, and accept any that contain a
    ``:``. The result is a dict — collisions are kept (last wins) since
    we never read more than a handful of fields anyway.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


def _emit_agent_log(
    sheets: SheetsClient,
    config: PortfoliomindConfig,
    successes: list[DeepDiveFacts],
    failures: list[tuple[str, str]],
    *,
    fetched_at: Optional[str] = None,
) -> None:
    """Write one AGENT_LOG row per success and a single summary for failures.

    The Message cell carries a compact JSON payload so the downstream
    forecast engine can read the facts back without re-parsing the
    page. We deliberately do NOT log the secrets.
    """
    sheets.ensure_worksheet(
        config.google_sheet_id, AGENT_LOG, TAB_HEADERS[AGENT_LOG]
    )
    from ..time_utils import iso_now

    ts = fetched_at or iso_now()
    rows: list[list[str]] = []
    for f in successes:
        msg = json.dumps(
            {
                "event": "investingpro.deepdive.success",
                "ticker": f.ticker,
                "facts": {
                    "market_cap": f.market_cap,
                    "pe_ratio": f.pe_ratio,
                    "eps_ttm": f.eps_ttm,
                    "dividend_yield": f.dividend_yield,
                    "beta": f.beta,
                    "analyst_consensus": f.analyst_consensus,
                },
                "fetched_at": f.fetched_at or ts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        rows.append([ts, "INFO", "portfoliomind.investingpro.deepdive", msg])

    if failures:
        msg = json.dumps(
            {
                "event": "investingpro.deepdive.batch_summary",
                "success_count": len(successes),
                "failure_count": len(failures),
                "failures": [
                    {"ticker": t, "reason": r} for (t, r) in failures
                ],
                "fetched_at": ts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        rows.append([ts, "WARNING", "portfoliomind.investingpro.deepdive", msg])

    if rows:
        try:
            sheets.append_rows(config.google_sheet_id, AGENT_LOG, rows)
        except Exception as e:  # best-effort — log + continue
            log.warning(
                "investingpro.deepdive.agent_log.append_failed err=%s", type(e).__name__
            )


__all__ = [
    "InvestingProDeepDiveError",
    "DeepDiveBatchResult",
    "deepdive_top_n",
]
