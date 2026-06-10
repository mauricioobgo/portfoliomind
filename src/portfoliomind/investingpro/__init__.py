"""InvestingPro integration: login, AI Picks scrape, and deep-dive fundamentals.

Card 2 of the PortfolioMind v4 build. Public contract:

    from portfoliomind.investingpro.login import login
    from portfoliomind.investingpro.scrape import scrape_ai_picks
    from portfoliomind.investingpro.deepdive import deepdive_top_n
    from portfoliomind.investingpro.parse import (
        RawPick,
        parse_ai_picks_table,
        parse_deepdive_payload,
    )
    from portfoliomind.investingpro.runner import run_morning

The Playwright flow is split into:

* :mod:`portfoliomind.investingpro.login` — opens a persistent Chromium context
  to ``SESSION_DIR``, fills the InvestingPro login form, and waits for the
  post-auth landing page. Screenshots on failure to ``SCREENSHOT_DIR``.

* :mod:`portfoliomind.investingpro.scrape` — navigates to the AI Picks page,
  waits for the results table to render, and converts each row into a
  :class:`RawPick` (which mirrors the RAW_PICKS tab columns 1:1).

* :mod:`portfoliomind.investingpro.deepdive` — for the top-N ``RawPick``s,
  opens the ticker deep-dive and captures the Fundamentals block.

* :mod:`portfoliomind.investingpro.runner` — the morning-run integration
  seam consumed by the card 4 scheduler. Exposes
  :func:`run_morning(ctx) -> MorningResult` which composes login +
  scrape + deep-dive and returns a result the scheduler can format
  into a Discord alert. The runner is the glue; the underlying modules
  remain the single-purpose, hermetic units they were.

The actual page-to-DataFrame conversion lives in
:mod:`portfoliomind.investingpro.parse` so it can be unit-tested without a
browser. ``scrape`` and ``deepdive`` are thin browser wrappers; ``parse`` is
the deterministic core.
"""

from __future__ import annotations

__all__ = [
    "login",
    "scrape_ai_picks",
    "deepdive_top_n",
    "RawPick",
    "DeepDiveFacts",
    "parse_ai_picks_table",
    "parse_deepdive_payload",
    "run_morning",
]
