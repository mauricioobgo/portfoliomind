"""InvestingPro morning runner — the card 2 integration seam.

Card 4's ``portfoliomind.scheduler.jobs.morning_run`` lazy-imports this
module and calls :func:`run_morning`. This module is the **glue** — it
composes the existing card 2 pieces (login, scrape, deep-dive) into a
single callable that:

  1. Opens a Playwright session on InvestingPro via
     :func:`portfoliomind.investingpro.login.login`.
  2. Scrapes the AI Picks table via
     :func:`portfoliomind.investingpro.scrape.scrape_ai_picks` and appends
     the fresh rows to ``RAW_PICKS`` in the Google Sheet.
  3. Pulls deep-dive fundamentals for the top-N tickers via
     :func:`portfoliomind.investingpro.deepdive.deepdive_top_n` (the
     deep-dive module emits to ``AGENT_LOG`` because there is no separate
     ``DEEPDIVE`` tab in the v4 schema).
  4. Returns a :class:`portfoliomind.scheduler.jobs.MorningResult` so the
     scheduler can format a single alert line.

Design rules (from the v4 spec and the operator's iron-rules):

* **Never raise.** Every failure mode is converted into a
  :class:`MorningResult` with the ``error`` field set. The scheduler
  depends on this contract — a raise from a runner is a bug, not a
  runtime feature.
* **Idempotent within a Bogota-local day.** The ``scraped_at`` value
  passed to :func:`scrape_ai_picks` is pinned to
  ``today.date().isoformat()`` + the cron trigger's nominal minute, so
  re-running morning_run twice in the same day produces the same dedup
  key and the second call appends zero new rows.
* **Headless.** :func:`login` is called with ``headless=True`` so the
  runner works inside the daemonized scheduler process (the cron job is
  not a TTY).
* **Configurable cap.** The default top-N for deep-dive is 5, matching
  the card 2 acceptance criterion (5 rows in RAW_PICKS).

We also support a test-only injection seam (``_login_factory`` and
``_scrape_factory``) so the test suite never has to touch a real
Playwright browser. The default factories call into the real
``portfoliomind.investingpro.{login,scrape,deepdive}`` modules; tests
monkeypatch them with fakes.

Public surface
--------------

* :func:`run_morning` — the contract callable expected by card 4.

All other functions are private (``_``-prefixed) and not part of the
public contract.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..config import PortfoliomindConfig
from ..logging_setup import get_logger
from ..scheduler.jobs import MorningContext, MorningResult
from .deepdive import deepdive_top_n
from .login import login
from .scrape import scrape_ai_picks

log = get_logger(__name__)


# --- Tunables ---------------------------------------------------------------

#: How many picks to scrape per morning. The InvestingPro page renders
#: more rows than this in some locales; we take the first ``limit``.
DEFAULT_SCRAPE_LIMIT: int = 20

#: How many top-N tickers to deep-dive for fundamentals. The card 2
#: acceptance criterion is "5 rows in RAW_PICKS" and the operator's
#: downstream strategy uses the top of that list for forecasts. 5 is
#: the minimal sensible default.
DEFAULT_DEEPDIVE_TOP_N: int = 5


__all__ = [
    "run_morning",
    "DEFAULT_SCRAPE_LIMIT",
    "DEFAULT_DEEPDIVE_TOP_N",
    "set_factories",
    "reset_factories",
]


# --- Test injection seam ----------------------------------------------------
#
# These factories exist so unit tests can swap in a fake login / scrape /
# deep-dive implementation without monkeypatching the module attribute.
# They default to the real functions; tests call :func:`set_factories` to
# override and :func:`reset_factories` to restore. The public surface
# (``run_morning``) does not change.
#
# Factories are intentionally loose on the parameter types: the
# underlying real functions expect real Playwright / SheetsClient
# objects, but tests can substitute duck-typed fakes.

_LoginFactory = Callable[
    [PortfoliomindConfig],
    Any,  # LoginResult in production; tests return a fake with .context / .page
]
_ScrapeFactory = Callable[
    [Any, Any, PortfoliomindConfig, str],
    Any,  # ScrapeResult in production
]
_DeepDiveFactory = Callable[
    [Any, Any, PortfoliomindConfig, list[str]],
    Any,  # DeepDiveBatchResult in production
]


def _default_scrape_factory(
    page: Any,
    sheets: Any,
    config: PortfoliomindConfig,
    pinned_ts: str,
) -> Any:
    """Default ``_scrape_factory`` — delegates to :func:`scrape_ai_picks`
    with the date-pinned ``scraped_at``.

    The factory is given the pinned timestamp as its 4th argument so
    tests can inspect the value the runner computed. The default
    behavior simply threads it into the real :func:`scrape_ai_picks`
    call.
    """
    return scrape_ai_picks(page, sheets, config, scraped_at=pinned_ts)


_login_factory: _LoginFactory = login
_scrape_factory: _ScrapeFactory = _default_scrape_factory
_deepdive_factory: _DeepDiveFactory = deepdive_top_n


def set_factories(
    *,
    login_factory: Optional[_LoginFactory] = None,
    scrape_factory: Optional[_ScrapeFactory] = None,
    deepdive_factory: Optional[_DeepDiveFactory] = None,
) -> None:
    """Override one or more of the underlying login / scrape / deep-dive
    factories. Intended for tests; production code should leave these
    alone.
    """
    global _login_factory, _scrape_factory, _deepdive_factory
    if login_factory is not None:
        _login_factory = login_factory
    if scrape_factory is not None:
        _scrape_factory = scrape_factory
    if deepdive_factory is not None:
        _deepdive_factory = deepdive_factory


def reset_factories() -> None:
    """Restore the default factories (the real card 2 modules)."""
    global _login_factory, _scrape_factory, _deepdive_factory
    _login_factory = login
    _scrape_factory = _default_scrape_factory
    _deepdive_factory = deepdive_top_n


# --- Helpers ----------------------------------------------------------------


def _date_pinned_scraped_at(today_iso: str) -> str:
    """Build a stable ``scraped_at`` for the run.

    The value is ``today``'s Bogota date + the morning's nominal minute
    (``08:30:00-05:00``). Pinning to a date + a fixed minute ensures a
    second ``morning_run`` call in the same day produces the same dedup
    key, so the second call appends zero new rows. Without the pin, the
    underlying :func:`scrape_ai_picks` would use a per-call timestamp
    and the dedup would never fire.
    """
    # ``today`` is a Bogota-local datetime; we render the date and a
    # fixed morning stamp. We keep the ``-05:00`` offset (Colombia is
    # UTC-5 year-round, no DST) so the value sorts correctly in
    # downstream tools.
    return f"{today_iso[:10]}T08:30:00-05:00"


def _deepdive_tickers_from_picks(
    picks_persisted: list[list[str]], *, top_n: int
) -> list[str]:
    """Extract the first ``top_n`` tickers from the rows we just
    appended to ``RAW_PICKS``.

    Each row is the 9-cell canonical shape; the ticker is column 0. We
    dedup (case-insensitive) so a duplicate ticker in the scrape does
    not produce a duplicate deep-dive call.
    """
    seen: set[str] = set()
    out: list[str] = []
    for r in picks_persisted:
        if not r:
            continue
        ticker = (r[0] or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
        if len(out) >= top_n:
            break
    return out


# --- The contract callable --------------------------------------------------


def run_morning(ctx: MorningContext) -> MorningResult:
    """Run the InvestingPro morning job. Always returns a
    :class:`MorningResult`; never raises.

    Steps
    -----

    1. Open a Playwright session via :func:`login`. Headless by default.
    2. Scrape the AI Picks table. The ``scraped_at`` is pinned to the
       Bogota date so the dedup is stable across re-runs the same day.
    3. Deep-dive the top-N tickers. The deep-dive module writes its
       output to ``AGENT_LOG`` (there is no separate ``DEEPDIVE`` tab in
       the v4 schema).
    4. Tear down the Playwright context. Best-effort — a teardown
       failure is logged but does not fail the run.

    Failures
    --------

    Any exception is converted into a
    :class:`MorningResult(runner="card2", error=str(e))` so the
    scheduler can keep the cron schedule ticking.
    """
    runner_id = "card2"
    try:
        config = ctx.config
        sheets = ctx.sheets
        sheet_id = ctx.sheet_id
        if config is None:
            return MorningResult(
                runner=runner_id,
                error="morning context has no config; refusing to scrape",
            )
        if not sheet_id:
            return MorningResult(
                runner=runner_id,
                error="morning context has empty sheet_id; refuse to scrape",
            )

        ctx.log_to_sheet(
            "INFO",
            f"investingpro.run_morning.start limit={DEFAULT_SCRAPE_LIMIT} "
            f"deepdive_top_n={DEFAULT_DEEPDIVE_TOP_N}",
        )

        # --- 1. Login (browser open) --------------------------------
        try:
            session = _login_factory(config)
        except Exception as e:  # noqa: BLE001
            log.error(
                "investingpro.runner.login_failed err_type=%s err=%r",
                type(e).__name__,
                str(e)[:200],
            )
            return MorningResult(
                runner=runner_id,
                error=f"login failed: {type(e).__name__}: {e}",
            )

        # The session object exposes ``page`` (a Playwright Page) and
        # ``context`` (a BrowserContext). The card 2 login returns a
        # ``LoginResult``; tests may return a duck-typed object with
        # the same attributes.
        page = getattr(session, "page", None)
        context = getattr(session, "context", None)
        if page is None:
            # Make sure we tear down whatever the factory handed us.
            _safe_close_context(context)
            return MorningResult(
                runner=runner_id,
                error="login returned a session with no .page attribute",
            )

        try:
            # --- 2. Scrape + persist ---------------------------------
            pinned_ts = _date_pinned_scraped_at(ctx.today.isoformat())
            scrape_result = _scrape_factory(
                page, sheets, config, pinned_ts,
            )
            # ``scrape_result`` is a :class:`ScrapeResult` in
            # production. It exposes ``picks`` (RawPick list) and
            # ``new_rows`` (the rows actually appended to the sheet).
            # Tests may return an object with a compatible shape.
            new_rows = list(getattr(scrape_result, "new_rows", []) or [])
            picks = list(getattr(scrape_result, "picks", []) or [])
            picks_scraped = len(picks)
            log.info(
                "investingpro.runner.scrape_done picks=%d new_rows=%d",
                picks_scraped,
                len(new_rows),
            )

            # --- 3. Deep-dive the top-N -----------------------------
            deepdive_tickers = _deepdive_tickers_from_picks(
                new_rows, top_n=DEFAULT_DEEPDIVE_TOP_N
            )
            if deepdive_tickers:
                try:
                    _deepdive_factory(
                        page, sheets, config, deepdive_tickers
                    )
                    log.info(
                        "investingpro.runner.deepdive_done tickers=%d",
                        len(deepdive_tickers),
                    )
                except Exception as e:  # noqa: BLE001
                    # Deep-dive failures are recoverable — we already
                    # have the picks, and the operator can re-run the
                    # deep-dive module out of band. Log and move on.
                    log.warning(
                        "investingpro.runner.deepdive_failed err_type=%s "
                        "err=%r",
                        type(e).__name__,
                        str(e)[:200],
                    )
                    ctx.log_to_sheet(
                        "WARNING",
                        f"investingpro.deepdive failed: "
                        f"{type(e).__name__}: {e}",
                    )
            else:
                log.info(
                    "investingpro.runner.no_deepdive_tickers "
                    "(0 new rows this run)"
                )
        finally:
            # --- 4. Tear down the browser context -----------------
            _safe_close_context(context)

        # If we got here without raising, the run succeeded (with
        # whatever caveat the deep-dive failure mode logged).
        ctx.log_to_sheet(
            "INFO",
            f"investingpro.run_morning.ok picks_scraped={picks_scraped}",
        )
        return MorningResult(
            runner=runner_id,
            picks_scraped=picks_scraped,
        )

    except Exception as e:  # noqa: BLE001
        # Catch-all. The scheduler depends on us NEVER raising.
        log.error(
            "investingpro.runner.unexpected err_type=%s err=%r",
            type(e).__name__,
            str(e)[:200],
        )
        return MorningResult(
            runner=runner_id,
            error=f"unexpected: {type(e).__name__}: {e}",
        )


def _safe_close_context(context: Any) -> None:
    """Best-effort teardown of a Playwright context.

    A teardown failure must not mask the real error path. The login
    module's own ``InvestingProLoginError`` path already closes the
    context before re-raising, so this only runs in the success path.
    """
    if context is None:
        return
    close = getattr(context, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "investingpro.runner.context_close_failed err=%s",
            type(e).__name__,
        )
