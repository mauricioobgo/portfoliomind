"""Wrapper around :class:`portfoliomind.news.store.NewsCache` for technicals.

The technical cache lives in the same SQLite file as the news cache
(card 5 put both tables in one DB on purpose — one file, one lock, one
backup story). The raw store methods
(:meth:`NewsCache.fetch_technical` / :meth:`NewsCache.store_technical`)
are keyed on ``(ticker, asof_date)``; this wrapper:

* Resolves ``asof_date`` from Bogota time (so a re-run in the same
  Bogota day hits the cache; a run after midnight Bogota misses).
* Reconstructs a :class:`~portfoliomind.signals.technicals.TechnicalScore`
  from the cached row, with the raw indicators attached when they
  are available.
* Tolerates an older v1 cache (no ``technical_cache`` table) by
  triggering the in-place migration on first open.
* Tolerates a malformed row (rare corruption from a prior code path)
  by returning ``None`` and logging — the caller falls back to a
  fresh fetch.

Tests use ``tmp_path`` SQLite files; production code goes through
:meth:`TechnicalCache.from_env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..logging_setup import get_logger
from ..news.store import DEFAULT_CACHE_PATH, NewsCache
from ..time_utils import BOGOTA_TZ, now_bogota
from .technicals import TechnicalScore

log = get_logger(__name__)


@dataclass
class TechnicalCache:
    """Thin facade over :class:`NewsCache` for the technicals table.

    The store field is private-ish (we keep a real reference so the
    caller can still fetch headlines + sentiment through the same
    connection). New code should use the ``get`` / ``put`` methods
    here for technical data.
    """

    store: NewsCache

    # --- Construction ---------------------------------------------------

    @classmethod
    def from_env(cls, *, env: Optional[dict] = None) -> "TechnicalCache":
        """Build a cache from env (or the passed mapping, for tests).

        Honors ``SIGNALS_CACHE_PATH`` to override the default
        ``.cache/news_cache.sqlite`` — the same DB file the news
        module uses. We deliberately share the file so a single
        backup captures both tables.
        """
        if env is None:
            env = dict(os.environ)
        path_str = (env.get("SIGNALS_CACHE_PATH") or "").strip()
        if path_str:
            db_path: Path | str = Path(path_str)
        else:
            db_path = DEFAULT_CACHE_PATH
        return cls(store=NewsCache(db_path=db_path))

    @classmethod
    def open(cls, db_path: Path | str) -> "TechnicalCache":
        """Open a specific DB path. Used by tests with ``tmp_path``."""
        return cls(store=NewsCache(db_path=db_path))

    # --- Day key --------------------------------------------------------

    @staticmethod
    def _day_key(today: Optional[datetime] = None) -> str:
        """Return YYYY-MM-DD in Bogota time."""
        dt = today if today is not None else now_bogota()
        return dt.astimezone(BOGOTA_TZ).strftime("%Y-%m-%d")

    # --- Get / put ------------------------------------------------------

    def get(
        self,
        ticker: str,
        *,
        today: Optional[datetime] = None,
    ) -> Optional[TechnicalScore]:
        """Return the cached :class:`TechnicalScore` for ``(ticker, today)`` or None.

        ``today`` defaults to the current Bogota wall time. The cache is
        only valid for the same Bogota day — a re-run after midnight
        Bogota misses and triggers a fresh fetch upstream.
        """
        asof = self._day_key(today)
        try:
            row = self.store.fetch_technical(ticker=ticker, asof_date=asof)
            if row is None:
                return None
            return self._row_to_score(row)
        except Exception as e:  # noqa: BLE001 — never raise from a cache read
            log.warning(
                "technical_cache: fetch failed for %s on %s: %s",
                ticker, asof, type(e).__name__,
            )
            return None

    def put(
        self,
        score: TechnicalScore,
        *,
        today: Optional[datetime] = None,
    ) -> None:
        """Persist ``score`` under ``(ticker, today)``.

        Idempotent: a second call for the same key overwrites the row.
        The store does not raise; a corrupt DB is logged and skipped
        (the next fetch will produce a fresh score from yfinance).
        """
        asof = self._day_key(today)
        # Stamp the asof_date onto the score so the cached row matches
        # what the caller will read back.
        if not score.asof_date:
            score = TechnicalScore(
                ticker=score.ticker,
                trend=score.trend,
                momentum=score.momentum,
                volatility=score.volatility,
                score=score.score,
                reasons=list(score.reasons),
                asof_date=asof,
            )
        try:
            self.store.store_technical(score, asof_date=asof)
        except Exception as e:  # noqa: BLE001 — never raise from a cache write
            log.warning(
                "technical_cache: store failed for %s on %s: %s",
                score.ticker, asof, type(e).__name__,
            )

    @staticmethod
    def _row_to_score(row: dict) -> TechnicalScore:
        """Reconstruct a TechnicalScore from a cached row dict."""
        return TechnicalScore(
            ticker=row["ticker"],
            trend=float(row["trend"]),
            momentum=float(row["momentum"]),
            volatility=float(row["volatility"]),
            score=float(row["score"]),
            reasons=list(row.get("reasons", [])),
            asof_date=row.get("asof_date", ""),
        )

    # --- Introspection --------------------------------------------------

    def stats(self) -> dict:
        """Same shape as :meth:`NewsCache.stats` — convenience pass-through."""
        return self.store.stats()


__all__ = ["TechnicalCache"]
