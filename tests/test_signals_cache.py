"""Unit tests for :mod:`portfoliomind.signals.cache` + the schema migration
that lives in :mod:`portfoliomind.news.store`.

Hermetic — uses a fresh ``tmp_path`` SQLite file per test. No network.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from portfoliomind.news.store import (
    HEADLINE_CACHE_SCHEMA_VERSION,
    NEWS_CACHE_SCHEMA_VERSION,
    NewsCache,
    NewsCacheError,
)
from portfoliomind.signals.cache import TechnicalCache
from portfoliomind.signals.technicals import TechnicalScore


# --- Helpers ---------------------------------------------------------------


def _ts(
    ticker: str = "AAPL",
    trend: float = 0.4,
    momentum: float = 0.2,
    volatility: float = 0.0,
    score: float | None = None,
    reasons: list[str] | None = None,
    asof_date: str = "",
) -> TechnicalScore:
    """Build a TechnicalScore with default test values."""
    if score is None:
        # Match the default weights so the test math is obvious.
        score = 0.5 * trend + 0.3 * momentum + 0.2 * volatility
    return TechnicalScore(
        ticker=ticker,
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        score=score,
        reasons=reasons if reasons is not None else ["test"],
        asof_date=asof_date,
    )


def _bogota_date(yyyy: int = 2026, mm: int = 6, dd: int = 10) -> datetime:
    """Bogota-local date that survives the cache's day-key extraction."""
    from zoneinfo import ZoneInfo
    return datetime(yyyy, mm, dd, tzinfo=ZoneInfo("America/Bogota"))


# --- Cache round-trip -----------------------------------------------------


class TestTechnicalCacheRoundTrip:
    def test_put_and_get(self, tmp_path: Path):
        db = tmp_path / "test_cache.sqlite"
        cache = TechnicalCache.open(db)
        ts = _ts()
        cache.put(ts, today=_bogota_date())
        got = cache.get("AAPL", today=_bogota_date())
        assert got is not None
        assert got.ticker == "AAPL"
        assert got.trend == pytest.approx(ts.trend)
        assert got.momentum == pytest.approx(ts.momentum)
        assert got.volatility == pytest.approx(ts.volatility)
        assert got.score == pytest.approx(ts.score)
        assert got.reasons == ts.reasons

    def test_get_miss_returns_none(self, tmp_path: Path):
        cache = TechnicalCache.open(tmp_path / "cache.sqlite")
        assert cache.get("ZZZZ", today=_bogota_date()) is None

    def test_get_different_day_misses(self, tmp_path: Path):
        """The same ticker, different Bogota day → fresh fetch (miss)."""
        cache = TechnicalCache.open(tmp_path / "cache.sqlite")
        cache.put(_ts(), today=_bogota_date(2026, 6, 10))
        # Day 11 → no cache row.
        assert cache.get("AAPL", today=_bogota_date(2026, 6, 11)) is None

    def test_ticker_is_uppercased(self, tmp_path: Path):
        cache = TechnicalCache.open(tmp_path / "cache.sqlite")
        cache.put(_ts(ticker="aapl"), today=_bogota_date())
        got = cache.get("AAPL", today=_bogota_date())
        assert got is not None
        assert got.ticker == "AAPL"

    def test_put_is_idempotent(self, tmp_path: Path):
        cache = TechnicalCache.open(tmp_path / "cache.sqlite")
        cache.put(_ts(score=0.10), today=_bogota_date())
        cache.put(_ts(score=0.90), today=_bogota_date())
        # Second put overwrites.
        got = cache.get("AAPL", today=_bogota_date())
        assert got is not None
        assert got.score == pytest.approx(0.90)
        # stats should still show exactly 1 row.
        s = cache.stats()
        assert s["technical_rows"] == 1

    def test_multiple_tickers(self, tmp_path: Path):
        cache = TechnicalCache.open(tmp_path / "cache.sqlite")
        for t, score in [("AAPL", 0.5), ("MSFT", -0.2), ("TSLA", 0.9)]:
            cache.put(_ts(ticker=t, score=score), today=_bogota_date())
        aapl = cache.get("AAPL", today=_bogota_date())
        msft = cache.get("MSFT", today=_bogota_date())
        tsla = cache.get("TSLA", today=_bogota_date())
        assert aapl is not None and msft is not None and tsla is not None
        assert aapl.score == pytest.approx(0.5)
        assert msft.score == pytest.approx(-0.2)
        assert tsla.score == pytest.approx(0.9)
        s = cache.stats()
        assert s["technical_rows"] == 3

    def test_get_without_explicit_today_uses_bogota_now(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The default ``today`` parameter resolves via Bogota time."""
        cache = TechnicalCache.open(tmp_path / "cache.sqlite")
        cache.put(_ts(), today=_bogota_date(2026, 6, 10))
        # Force the cache's clock to read 2026-06-10 in Bogota.
        from portfoliomind.signals import cache as cache_mod

        class _FrozenDateTime:
            @classmethod
            def now(cls, tz=None):
                return _bogota_date(2026, 6, 10).astimezone(tz) if tz else _bogota_date(2026, 6, 10)

        monkeypatch.setattr(cache_mod, "now_bogota", _FrozenDateTime.now)
        got = cache.get("AAPL")
        assert got is not None

    def test_reasons_persist_as_json(self, tmp_path: Path):
        cache = TechnicalCache.open(tmp_path / "cache.sqlite")
        reasons = ["trend: SMA20/SMA50=1.05 → score +0.20", "momentum: RSI=72 (overbought)"]
        cache.put(_ts(reasons=reasons), today=_bogota_date())
        got = cache.get("AAPL", today=_bogota_date())
        assert got is not None
        assert got.reasons == reasons

    def test_stats_includes_technical_rows(self, tmp_path: Path):
        cache = TechnicalCache.open(tmp_path / "cache.sqlite")
        cache.put(_ts(ticker="AAPL"), today=_bogota_date())
        cache.put(_ts(ticker="MSFT"), today=_bogota_date())
        s = cache.stats()
        assert "technical_rows" in s
        assert s["technical_rows"] == 2
        assert s["schema_version"] == NEWS_CACHE_SCHEMA_VERSION


# --- Schema migration v1 → v2 ---------------------------------------------


class TestSchemaMigration:
    def test_fresh_db_is_v2(self, tmp_path: Path):
        cache = TechnicalCache.open(tmp_path / "fresh.sqlite")
        s = cache.stats()
        assert s["schema_version"] == 2

    def test_v1_db_is_upgraded_in_place(self, tmp_path: Path):
        """A hand-built v1 cache (no technical_cache table) gets migrated."""
        db = tmp_path / "v1.sqlite"
        # Build a v1 cache by hand: schema_meta + headlines + sentiment_scores.
        with sqlite3.connect(str(db)) as c:
            c.execute(
                "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            c.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', '1')"
            )
            c.execute(
                """
                CREATE TABLE headlines (
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
                """
                CREATE TABLE sentiment_scores (
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
            c.commit()

        # Open via TechnicalCache — should migrate silently.
        cache = TechnicalCache.open(db)
        s = cache.stats()
        assert s["schema_version"] == 2
        assert s["headline_rows"] == 0  # pre-existing tables untouched
        # And we can write/read the new table.
        cache.put(_ts(ticker="AAPL"), today=_bogota_date())
        got = cache.get("AAPL", today=_bogota_date())
        assert got is not None

    def test_v2_db_does_not_migrate(self, tmp_path: Path):
        """A v2 DB stays at v2 (migration runs but is a no-op)."""
        db = tmp_path / "v2.sqlite"
        cache = TechnicalCache.open(db)
        cache.put(_ts(), today=_bogota_date())
        # Reopen — version is still 2, no spurious migration log.
        cache2 = TechnicalCache.open(db)
        s = cache2.stats()
        assert s["schema_version"] == 2

    def test_newer_version_is_rejected(self, tmp_path: Path):
        """A cache with a newer schema than the code can handle raises."""
        db = tmp_path / "future.sqlite"
        with sqlite3.connect(str(db)) as c:
            c.execute(
                "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            c.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', '999')"
            )
            c.commit()
        with pytest.raises(NewsCacheError) as exc_info:
            NewsCache(db_path=db)
        assert "schema version" in str(exc_info.value).lower()

    def test_v1_data_preserved_after_migration(self, tmp_path: Path):
        """Pre-existing v1 rows survive the v1 → v2 migration."""
        db = tmp_path / "v1_with_data.sqlite"
        with sqlite3.connect(str(db)) as c:
            c.execute(
                "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            c.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', '1')"
            )
            c.execute(
                """
                CREATE TABLE sentiment_scores (
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
                "INSERT INTO sentiment_scores VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("AAPL", "2026-06-09", 0.42, 3, "[]", "gpt-4o-mini", "2026-06-09T08:30:00-05:00"),
            )
            c.commit()

        cache = TechnicalCache.open(db)
        # v1 row still queryable through the underlying NewsCache.
        rec = cache.store.fetch_sentiment(ticker="AAPL", day="2026-06-09")
        assert rec is not None
        assert rec["score"] == 0.42


# --- Constants ------------------------------------------------------------


class TestSchemaVersionConstants:
    def test_news_schema_version_is_v2(self):
        assert NEWS_CACHE_SCHEMA_VERSION == 2

    def test_headline_alias_matches(self):
        # The back-compat alias must point at the same number so
        # any card-5 callers that imported HEADLINE_CACHE_SCHEMA_VERSION
        # see a consistent value.
        assert HEADLINE_CACHE_SCHEMA_VERSION == NEWS_CACHE_SCHEMA_VERSION


# --- Corruption tolerance -------------------------------------------------


class TestCorruptionTolerance:
    def test_malformed_reasons_json_returns_none(self, tmp_path: Path):
        """A hand-corrupted row should NOT crash the cache; it should
        return None and let the caller recompute."""
        db = tmp_path / "corrupt.sqlite"
        cache = TechnicalCache.open(db)
        # Inject a row with a broken JSON blob.
        with sqlite3.connect(str(db)) as c:
            c.execute(
                "INSERT INTO technical_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "AAPL",
                    "2026-06-10",
                    0.1,
                    0.2,
                    0.0,
                    0.1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "not-json{",
                    "2026-06-10T08:30:00-05:00",
                ),
            )
            c.commit()
        # The contract: never raise. A malformed row is reported as
        # a miss so the caller falls through to a fresh fetch.
        got = cache.get("AAPL", today=_bogota_date())
        assert got is None

    def test_cache_db_path_is_created(self, tmp_path: Path):
        """The DB file + parent dir are created if missing."""
        db = tmp_path / "nested" / "deep" / "cache.sqlite"
        assert not db.parent.exists()
        cache = TechnicalCache.open(db)
        cache.put(_ts(), today=_bogota_date())
        assert db.exists()
