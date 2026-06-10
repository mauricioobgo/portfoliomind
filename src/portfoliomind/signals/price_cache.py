"""SQLite cache for daily OHLCV bars pulled from yfinance.

Why a sibling cache (and not extending ``news/store.py``)?

* The news cache is keyed by ``(ticker, day)`` for sentiment + by
  ``(source, headline_hash)`` for raw headlines. Its schema is news-
  shaped and would not gain from a price-bars table bolted on.
* The price cache is keyed by ``(ticker, as_of_date)`` and stores a
  small JSON blob of ``{date: open/high/low/close/volume}`` for one
  ticker. A separate file keeps blast radius small if either table
  gets corrupted.
* The two stores have different freshness needs: news is "same day,
  replay-OK"; prices are "fresh within the trading hour, replay-OK
  if the market is closed". A 1-hour TTL on the price cache is the
  right knob for the morning run (08:30 Bogota).

The store is intentionally a tiny copy of the news cache pattern: same
lock, same ``check_same_thread=False`` SQLite connection, same
schema-version guard, same JSON-as-blob serialization for the OHLCV
rows.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from typing import Iterator, Optional

from ..logging_setup import get_logger
from ..time_utils import iso_now

log = get_logger(__name__)

#: Bump this when changing the table layout so older caches are detected.
PRICE_CACHE_SCHEMA_VERSION: int = 1

#: Default location: <project>/.cache/price_cache.sqlite. Override via
#: the constructor ``db_path`` arg.
DEFAULT_CACHE_PATH: Path = Path(".cache") / "price_cache.sqlite"

#: How long a price row is considered fresh. The morning run (08:30
#: Bogota) re-runs occasionally for a couple of hours — anything
#: within 1h of the pull is fine to use. Once the market is closed
#: (post-16:00 ET) the data does not change intra-day so a hit
#: anywhere in the session is OK.
PRICE_TTL_SECONDS: int = 60 * 60  # 1 hour

#: How many calendar days of history we cache per ticker. We need
#: 200+ to compute SMA(200), and 1y of daily bars is the spec.
PRICE_HISTORY_DAYS: int = 365


class PriceCacheError(RuntimeError):
    """Raised when the cache cannot be opened (e.g. schema mismatch)."""


class PriceCache:
    """Thread-safe SQLite cache for daily OHLCV bars.

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
                    (str(PRICE_CACHE_SCHEMA_VERSION),),
                )
            else:
                existing = int(row["value"])
                if existing != PRICE_CACHE_SCHEMA_VERSION:
                    raise PriceCacheError(
                        f"cache schema version mismatch: file={existing} code="
                        f"{PRICE_CACHE_SCHEMA_VERSION}. Delete the cache or migrate."
                    )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS price_bars (
                    ticker TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    bars_json TEXT NOT NULL,
                    pulled_at TEXT NOT NULL,
                    PRIMARY KEY (ticker, as_of_date)
                )
                """
            )

    # --- Read / write ----------------------------------------------------

    def fetch_bars(
        self,
        *,
        ticker: str,
        as_of_date: str,
        max_age_s: int = PRICE_TTL_SECONDS,
    ) -> Optional[list[dict]]:
        """Return cached bars for ``(ticker, as_of_date)`` or ``None``.

        A row is returned only if it is still within ``max_age_s`` of
        ``iso_now()`` — stale rows are treated as cache misses so a
        long-running scheduler eventually re-pulls when the market
        reopens.

        Each entry in the returned list is a dict with at least::

            {"date": "YYYY-MM-DD", "open": float, "high": float,
             "low": float, "close": float, "volume": int}

        The dict may also include ``adj_close`` (yfinance's
        ``Adj Close`` column) when the source data has it.
        """
        with self._lock, self._conn() as c:
            row = c.execute(
                """
                SELECT bars_json, pulled_at FROM price_bars
                WHERE ticker = ? AND as_of_date = ?
                """,
                (ticker, as_of_date),
            ).fetchone()
        if row is None:
            return None
        try:
            pulled = datetime.fromisoformat(row["pulled_at"])
        except ValueError:
            return None
        age_s = (datetime.now(pulled.tzinfo) - pulled).total_seconds()
        if age_s > max_age_s:
            return None
        try:
            data = json.loads(row["bars_json"])
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list):
            return None
        return data

    def store_bars(
        self,
        *,
        ticker: str,
        as_of_date: str,
        bars: list[dict],
    ) -> None:
        """Persist ``bars`` for ``(ticker, as_of_date)``.

        Idempotent: a second call for the same key overwrites the row
        with the freshest ``pulled_at`` timestamp.
        """
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO price_bars
                    (ticker, as_of_date, bars_json, pulled_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    ticker,
                    as_of_date,
                    json.dumps(bars),
                    iso_now(),
                ),
            )

    # --- Introspection ---------------------------------------------------

    def stats(self) -> dict:
        """Return a small dict of counts for operator inspection / debug logs."""
        with self._lock, self._conn() as c:
            count = c.execute("SELECT COUNT(*) AS n FROM price_bars").fetchone()["n"]
        return {
            "db_path": str(self._db_path),
            "schema_version": PRICE_CACHE_SCHEMA_VERSION,
            "bar_rows": count,
        }

    @staticmethod
    def _day_key(d: date | datetime | None = None) -> str:
        """Return the YYYY-MM-DD key (in Bogota time when ``d`` is None)."""
        if d is None:
            from ..time_utils import now_bogota

            return now_bogota().strftime("%Y-%m-%d")
        if isinstance(d, datetime):
            return d.strftime("%Y-%m-%d")
        return d.strftime("%Y-%m-%d")


__all__ = [
    "PriceCache",
    "PriceCacheError",
    "PRICE_CACHE_SCHEMA_VERSION",
    "PRICE_TTL_SECONDS",
    "PRICE_HISTORY_DAYS",
    "DEFAULT_CACHE_PATH",
]
