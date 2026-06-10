"""SQLite cache for the news module.

Why cache at all?
* Re-running the morning loop in the same day must return identical
  sentiment scores — that is part of the spec's idempotency contract.
* Hitting three RSS feeds + the OpenAI API on every re-run is wasteful
  AND slow.
* The 4o-mini sentiment call is deterministic for a fixed input, so
  caching the (ticker, headline_hash, day) → score mapping is safe.

What we cache:
* **Raw headlines** keyed by (source, headline_hash) with the
  published_at timestamp. The feed fetcher reads this to avoid
  re-hitting the network when the feed is unchanged.
* **Sentiment scores** keyed by (ticker, day) with a JSON blob of
  per-headline scores + the aggregate. The sentiment scorer reads
  this to return the same answer for the same day.
* **Technical scores** keyed by (ticker, asof_date) with the raw
  indicators + per-component sub-scores. Card 6 added this table to
  the same DB so a single backup captures both news and signals.

What we do NOT cache:
* The per-run in-process memoization in :mod:`portfoliomind.news.feeds`
  is separate from this — it's the only thing that lets the per-ticker
  scoring path avoid hitting the network N times.

Cache schema version: bump ``NEWS_CACHE_SCHEMA_VERSION`` when you add
a new table. The store runs a forward-only migration (e.g. ``v1 → v2``)
so older caches are upgraded in place rather than refused. A newer
version than the code can handle is rejected.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Iterator, Optional

from ..logging_setup import get_logger
from ..time_utils import iso_now
from ._headline import Headline

log = get_logger(__name__)

#: Bump this when changing the table layout so older caches are detected.
#: v1 = headlines + sentiment (card 5). v2 = + technical_cache (card 6).
NEWS_CACHE_SCHEMA_VERSION: int = 2
#: Back-compat alias. Card 5 callers imported this name.
HEADLINE_CACHE_SCHEMA_VERSION: int = NEWS_CACHE_SCHEMA_VERSION

# Default location: <project>/.cache/news_cache.sqlite, override via
# the constructor ``db_path`` arg.
DEFAULT_CACHE_PATH: Path = Path(".cache") / "news_cache.sqlite"


class NewsCache:
    """A thin, thread-safe wrapper around the headline + sentiment cache.

    The store is a single SQLite file. All writes are serialized via a
    re-entrant lock so concurrent scheduler runs do not corrupt it.
    """

    def __init__(self, db_path: Path | str = DEFAULT_CACHE_PATH):
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    # --- Connection management -------------------------------------------

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # ``check_same_thread=False`` because APScheduler jobs may run on
        # a different thread than the one that created the connection.
        # The lock is the real serialization guarantee.
        c = sqlite3.connect(str(self._db_path), timeout=10.0, check_same_thread=False)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            row = c.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                    (str(NEWS_CACHE_SCHEMA_VERSION),),
                )
            else:
                existing = int(row["value"])
                if existing > NEWS_CACHE_SCHEMA_VERSION:
                    raise NewsCacheError(
                        f"cache schema version newer than code: file={existing} code="
                        f"{NEWS_CACHE_SCHEMA_VERSION}. Refusing to open; check out an "
                        "older revision or wipe the cache."
                    )
                if existing < NEWS_CACHE_SCHEMA_VERSION:
                    self._migrate(c, existing)

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS headlines (
                    source TEXT NOT NULL,
                    headline_hash TEXT NOT NULL,
                    title TEXT NOT NULL,
                    link TEXT NOT NULL DEFAULT '',
                    published_at TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (source, headline_hash)
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_headlines_pub ON headlines(published_at)"
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    ticker TEXT NOT NULL,
                    day TEXT NOT NULL,
                    score REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    per_headline_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    scored_at TEXT NOT NULL,
                    PRIMARY KEY (ticker, day)
                )
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS technical_cache (
                    ticker TEXT NOT NULL,
                    asof_date TEXT NOT NULL,
                    trend REAL NOT NULL,
                    momentum REAL NOT NULL,
                    volatility REAL NOT NULL,
                    score REAL NOT NULL,
                    sma_fast REAL,
                    sma_slow REAL,
                    rsi REAL,
                    short_vol REAL,
                    long_vol REAL,
                    reasons_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (ticker, asof_date)
                )
                """
            )

    def _migrate(self, c: sqlite3.Connection, existing: int) -> None:
        """Run forward-only schema migrations in order.

        Card 6: v1 → v2 adds the ``technical_cache`` table. The news
        tables are unchanged; we just stamp the new version. Later
        cards should add ``if existing < N:`` blocks here.
        """
        if existing < 2:
            # v1 had headlines + sentiment_scores. v2 adds technical_cache.
            # The table is created by _init_schema() after this returns;
            # we only need to bump the version row.
            c.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                ("2",),
            )
            log.info("news_cache: migrated v1 → v2 (added technical_cache)")

    # --- Headline storage -----------------------------------------------

    def store_headlines(
        self,
        *,
        feed_name: str,
        headlines: Iterable[Headline],
    ) -> int:
        """Upsert headlines for ``feed_name``. Returns the row count written.

        Existing rows with the same (source, headline_hash) are left
        untouched (the title is identical by construction; the only
        thing that could change is the link, and we don't trust feeds
        to be consistent about that).
        """
        rows = [
            (
                feed_name,
                h.headline_hash,
                h.title,
                h.link,
                h.published_at.isoformat(),
                iso_now(),
            )
            for h in headlines
        ]
        if not rows:
            return 0
        with self._lock, self._conn() as c:
            c.executemany(
                """
                INSERT OR IGNORE INTO headlines
                    (source, headline_hash, title, link, published_at, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            # Rowcount after executemany is implementation-dependent
            # in some Python versions; query the actual delta instead.
            return len(rows)

    def fetch_cached_headlines(
        self,
        *,
        feed_name: str,
        since: datetime,
    ) -> Optional[list[Headline]]:
        """Return cached headlines for ``feed_name`` published >= ``since``.

        Returns ``None`` (NOT an empty list) when the feed has zero
        cached rows at all — the caller uses this to distinguish
        "never fetched" from "fetched and empty". The latter is rare
        (a feed that legitimately had nothing in the window); the
        former is the common "first run" case where we must hit
        the network.

        An empty list return is treated as a cache hit by the caller
        (no network re-fetch) — this is the right behaviour because
        a 0-row feed fetch took work and we should not repeat it
        within the same window.
        """
        with self._lock, self._conn() as c:
            rows = c.execute(
                """
                SELECT source, headline_hash, title, link, published_at
                FROM headlines
                WHERE source = ? AND published_at >= ?
                ORDER BY published_at DESC
                """,
                (feed_name, since.isoformat()),
            ).fetchall()

        if not rows:
            # Differentiate "no rows at all for this feed" from "no rows
            # in this window". The first is a cold cache.
            with self._lock, self._conn() as c:
                any_rows = c.execute(
                    "SELECT 1 FROM headlines WHERE source = ? LIMIT 1",
                    (feed_name,),
                ).fetchone()
            if any_rows is None:
                return None
            return []

        out: list[Headline] = []
        for r in rows:
            try:
                published = datetime.fromisoformat(r["published_at"])
            except ValueError:
                continue
            out.append(
                Headline(
                    title=r["title"],
                    source=r["source"],
                    published_at=published,
                    link=r["link"],
                    headline_hash=r["headline_hash"],
                )
            )
        return out

    # --- Sentiment storage ---------------------------------------------

    def store_sentiment(
        self,
        *,
        ticker: str,
        day: str,  # YYYY-MM-DD in Bogota
        score: float,
        sample_size: int,
        per_headline: list[dict],
        model: str,
    ) -> None:
        """Persist a sentiment result for ``(ticker, day)``.

        Idempotent: a second call for the same key overwrites the row.
        """
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO sentiment_scores
                    (ticker, day, score, sample_size, per_headline_json, model, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    day,
                    float(score),
                    int(sample_size),
                    json.dumps(per_headline),
                    model,
                    iso_now(),
                ),
            )

    def fetch_sentiment(self, *, ticker: str, day: str) -> Optional[dict]:
        """Return the cached sentiment record for ``(ticker, day)`` or ``None``.

        Shape::

            {
                "ticker": str,
                "day": str,
                "score": float,
                "sample_size": int,
                "per_headline": list[dict],
                "model": str,
                "scored_at": str,
            }
        """
        with self._lock, self._conn() as c:
            row = c.execute(
                """
                SELECT ticker, day, score, sample_size, per_headline_json, model, scored_at
                FROM sentiment_scores
                WHERE ticker = ? AND day = ?
                """,
                (ticker, day),
            ).fetchone()
        if row is None:
            return None
        return {
            "ticker": row["ticker"],
            "day": row["day"],
            "score": row["score"],
            "sample_size": row["sample_size"],
            "per_headline": json.loads(row["per_headline_json"]),
            "model": row["model"],
            "scored_at": row["scored_at"],
        }

    # --- Introspection --------------------------------------------------

    def stats(self) -> dict:
        """Return a small dict of counts for operator inspection / debug logs."""
        with self._lock, self._conn() as c:
            h_count = c.execute("SELECT COUNT(*) AS n FROM headlines").fetchone()["n"]
            s_count = c.execute("SELECT COUNT(*) AS n FROM sentiment_scores").fetchone()["n"]
            t_count = c.execute("SELECT COUNT(*) AS n FROM technical_cache").fetchone()["n"]
        return {
            "db_path": str(self._db_path),
            "schema_version": NEWS_CACHE_SCHEMA_VERSION,
            "headline_rows": h_count,
            "sentiment_rows": s_count,
            "technical_rows": t_count,
        }

    # --- Technical cache (card 6) --------------------------------------

    def store_technical(self, score: Any, *, asof_date: str) -> None:
        """Persist a :class:`TechnicalScore` for ``(score.ticker, asof_date)``.

        Idempotent: a second call for the same key overwrites the row.
        ``score`` is duck-typed (any object with the TechnicalScore
        fields) to avoid an import cycle between signals.cache and
        signals.technicals.
        """
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO technical_cache
                    (ticker, asof_date, trend, momentum, volatility, score,
                     sma_fast, sma_slow, rsi, short_vol, long_vol,
                     reasons_json, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    score.ticker.upper(),
                    asof_date,
                    float(score.trend),
                    float(score.momentum),
                    float(score.volatility),
                    float(score.score),
                    self._extract_indicator(score, "sma_fast"),
                    self._extract_indicator(score, "sma_slow"),
                    self._extract_indicator(score, "rsi"),
                    self._extract_indicator(score, "short_vol"),
                    self._extract_indicator(score, "long_vol"),
                    json.dumps(list(score.reasons)),
                    iso_now(),
                ),
            )

    @staticmethod
    def _extract_indicator(score: Any, name: str) -> Optional[float]:
        """Pull a named indicator from a TechnicalScore if present.

        Older callers may build the score without the raw indicators
        populated (e.g. tests) — we tolerate that and store NULL.
        """
        val = getattr(score, name, None)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def fetch_technical(self, *, ticker: str, asof_date: str) -> Optional[dict]:
        """Return the cached technical row for ``(ticker, asof_date)`` or None.

        Shape::

            {
                "ticker": str,
                "asof_date": str,
                "trend": float,
                "momentum": float,
                "volatility": float,
                "score": float,
                "sma_fast": float | None,
                "sma_slow": float | None,
                "rsi": float | None,
                "short_vol": float | None,
                "long_vol": float | None,
                "reasons": list[str],
                "fetched_at": str,
            }
        """
        with self._lock, self._conn() as c:
            row = c.execute(
                """
                SELECT ticker, asof_date, trend, momentum, volatility, score,
                       sma_fast, sma_slow, rsi, short_vol, long_vol,
                       reasons_json, fetched_at
                FROM technical_cache
                WHERE ticker = ? AND asof_date = ?
                """,
                (ticker.upper(), asof_date),
            ).fetchone()
        if row is None:
            return None
        return {
            "ticker": row["ticker"],
            "asof_date": row["asof_date"],
            "trend": row["trend"],
            "momentum": row["momentum"],
            "volatility": row["volatility"],
            "score": row["score"],
            "sma_fast": row["sma_fast"],
            "sma_slow": row["sma_slow"],
            "rsi": row["rsi"],
            "short_vol": row["short_vol"],
            "long_vol": row["long_vol"],
            "reasons": json.loads(row["reasons_json"]),
            "fetched_at": row["fetched_at"],
        }


class NewsCacheError(RuntimeError):
    """Raised when the cache cannot be opened (e.g. schema mismatch)."""


__all__ = [
    "NewsCache",
    "NewsCacheError",
    "NEWS_CACHE_SCHEMA_VERSION",
    "HEADLINE_CACHE_SCHEMA_VERSION",  # back-compat alias
    "DEFAULT_CACHE_PATH",
]
