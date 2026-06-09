"""RSS feed scraper for the news module.

Three free feeds are wired in:

* **Reuters** business news. Requires a real-ish ``User-Agent`` header
  or the endpoint returns 403.
* **MarketWatch** market pulse. Sometimes 403s behind aggressive
  residential proxies.
* **Investing.com** news. Custom URL path; render is HTML inside the
  feed items, so we strip the HTML before returning.

Each feed is wrapped in its own try/except — a failure in one feed
returns an empty list and logs at WARN, leaving the other feeds
unaffected. This is the "fail-soft" guarantee from the spec.

No raw headline text gets logged at INFO level. The headline body
contains market-moving information; it goes to DEBUG only (operator
can flip the logger when debugging) and never to the agent's
AGENT_LOG tab.

We use :mod:`feedparser` rather than rolling our own XML parser. It
tolerates the messy XML most RSS feeds actually ship in the wild.

``Headline`` is defined in :mod:`portfoliomind.news._headline` (a
leaf module with no project dependencies) to break the circular
import between this module and :mod:`portfoliomind.news.store`. We
re-export it here for backwards compatibility with code that did
``from portfoliomind.news.feeds import Headline``.
"""

from __future__ import annotations

import re
import socket
from datetime import datetime, timedelta, timezone
from typing import Iterable

import feedparser
import requests

from ..logging_setup import get_logger
from ..time_utils import BOGOTA_TZ, now_bogota
from ._headline import Headline
from .store import NewsCache

__all__ = [
    "Headline",
    "REUTERS_FEED_URL",
    "MARKETWATCH_FEED_URL",
    "INVESTING_COM_FEED_URL",
    "fetch_all_feeds",
    "_strip_html",
    "_reset_per_run_cache",
]

log = get_logger(__name__)


# --- Feed URLs --------------------------------------------------------------

# Reuters business-top (free, no auth, requires a desktop User-Agent).
REUTERS_FEED_URL: str = "https://feeds.reuters.com/reuters/businessNews"

# MarketWatch top stories.
MARKETWATCH_FEED_URL: str = "https://www.marketwatch.com/rss/topstories"

# Investing.com news (English, latest). The path is stable.
INVESTING_COM_FEED_URL: str = "https://www.investing.com/rss/news_25.rss"


# --- Configuration ----------------------------------------------------------

# Per-request timeout (connect + read). 10s is generous for an RSS feed
# but bounded so one slow feed doesn't stall the whole morning run.
FEED_TIMEOUT_S: float = 10.0

# User-Agents per feed. Reuters and MarketWatch reject the default
# Python-requests UA; we send a plausible desktop browser.
_REUTERS_UA: str = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_MARKETWATCH_UA: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15"
)
_INVESTING_UA: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# --- HTML stripping ---------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace.

    Some feeds (Investing.com) put the headline content in the
    ``summary`` field as raw HTML. We strip tags defensively even on
    the title, because Reuters occasionally embeds an ``<em>`` or
    ``&amp;`` entity.
    """
    if not text:
        return ""
    # Decode the few HTML entities that survive the strip. We do NOT
    # import html.unescape here because feedparser already decodes
    # most entities — anything left is usually a stray &amp; or &lt;.
    no_tags = _HTML_TAG_RE.sub(" ", text)
    no_amp = no_tags.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    no_nbsp = no_amp.replace("\xa0", " ").replace("\u2009", " ")
    return _WHITESPACE_RE.sub(" ", no_nbsp).strip()


# --- Single-feed fetch ------------------------------------------------------


def _fetch_feed(
    *,
    feed_name: str,
    url: str,
    user_agent: str,
    since: datetime,
) -> list[Headline]:
    """Fetch one RSS feed and return ``Headline`` records newer than ``since``.

    Returns ``[]`` on any failure (timeout, HTTP error, malformed XML).
    The error is logged at WARN with the feed name but no body content.
    """
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/rss+xml, application/xml, */*",
            },
            timeout=FEED_TIMEOUT_S,
        )
        resp.raise_for_status()
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        log.warning("feed %s: HTTP %s (skipping)", feed_name, code)
        return []
    except (requests.RequestException, socket.timeout) as e:
        log.warning("feed %s: request failed: %s (skipping)", feed_name, type(e).__name__)
        return []
    except Exception as e:  # last-ditch — never let a feed break the run
        log.warning("feed %s: unexpected error: %s (skipping)", feed_name, type(e).__name__)
        return []

    try:
        parsed = feedparser.parse(resp.content)
    except Exception as e:  # feedparser rarely raises, but be defensive
        log.warning("feed %s: feedparser failed: %s (skipping)", feed_name, type(e).__name__)
        return []

    if getattr(parsed, "bozo", False) and not parsed.entries:
        # bozo=True with no entries means the feed was actually broken.
        log.warning(
            "feed %s: malformed feed (%s) (skipping)",
            feed_name,
            getattr(parsed, "bozo_exception", "unknown"),
        )
        return []

    out: list[Headline] = []
    for entry in parsed.entries:
        title_raw = entry.get("title", "") or ""
        title = _strip_html(title_raw)
        if not title:
            continue
        published = _parse_entry_datetime(entry, fallback=now_bogota())
        if published < since:
            # Feed entries are usually in reverse chronological order;
            # once we hit one older than `since`, every later one is too.
            break
        link = (entry.get("link") or "").strip()
        out.append(Headline.make(title=title, source=feed_name, published_at=published, link=link))

    log.debug("feed %s: %d headlines since %s", feed_name, len(out), since.isoformat())
    return out


def _parse_entry_datetime(entry, *, fallback: datetime) -> datetime:
    """Best-effort parse of an RSS entry's published date.

    Prefers :func:`feedparser._parse_date` results because they carry
    tzinfo. Falls back to the *published_parsed* tuple, then to the
    *updated_parsed* tuple, then to ``fallback`` (the run's wall time).
    """
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                # ``t`` is a time.struct_time; build a UTC datetime then
                # convert to Bogota.
                dt = datetime(*t[:6], tzinfo=timezone.utc).astimezone(BOGOTA_TZ)
                return dt
            except (TypeError, ValueError):
                continue
    # Last-ditch: try the string form via feedparser's _parse_date (private)
    for attr in ("published", "updated", "created"):
        s = getattr(entry, attr, None)
        if not s:
            continue
        try:
            dt = feedparser._parse_date(s)  # type: ignore[attr-defined]
            if dt:
                return dt.astimezone(BOGOTA_TZ)
        except Exception:
            continue
    return fallback


# --- Public entry point -----------------------------------------------------


# Cached per-process so a run that scores the whole universe doesn't
# hit the network once per ticker. Module-level state, not class-level,
# because the news module is process-wide.
_PER_RUN_FETCH_CACHE: dict[tuple[str, datetime], list[Headline]] = {}


def _cache_key(feed_name: str, since: datetime) -> tuple[str, datetime]:
    # Truncate to the minute so consecutive calls within the same minute
    # hit the same cache slot — useful when the scorer is called per
    # ticker in a tight loop.
    return (feed_name, since.replace(second=0, microsecond=0))


def fetch_all_feeds(
    *,
    since_hours: int = 24,
    cache: NewsCache | None = None,
) -> list[Headline]:
    """Fetch the union of the three feeds, restricted to ``since_hours`` back.

    The optional :class:`NewsCache` is consulted before each feed; cached
    entries are returned without a network call. Failures are
    transparent to the caller — a feed that 403s or times out is simply
    absent from the result.

    The per-run in-process cache short-circuits a second call to
    :func:`fetch_all_feeds` for the same window. The persistent
    :class:`NewsCache` keeps results across runs for the day.
    """
    if since_hours <= 0:
        raise ValueError("since_hours must be positive")
    since = now_bogota() - timedelta(hours=since_hours)
    log.info("news: fetching feeds for last %dh (since %s)", since_hours, since.isoformat())

    feeds: Iterable[tuple[str, str, str]] = (
        ("reuters", REUTERS_FEED_URL, _REUTERS_UA),
        ("marketwatch", MARKETWATCH_FEED_URL, _MARKETWATCH_UA),
        ("investing_com", INVESTING_COM_FEED_URL, _INVESTING_UA),
    )

    all_headlines: list[Headline] = []
    for feed_name, url, ua in feeds:
        key = _cache_key(feed_name, since)
        if key in _PER_RUN_FETCH_CACHE:
            log.debug("feed %s: in-process cache hit (%d)", feed_name, len(_PER_RUN_FETCH_CACHE[key]))
            all_headlines.extend(_PER_RUN_FETCH_CACHE[key])
            continue

        # Consult persistent cache first if provided.
        if cache is not None:
            cached = cache.fetch_cached_headlines(feed_name=feed_name, since=since)
            if cached is not None:
                log.debug("feed %s: persistent cache hit (%d)", feed_name, len(cached))
                _PER_RUN_FETCH_CACHE[key] = cached
                all_headlines.extend(cached)
                continue

        fetched = _fetch_feed(feed_name=feed_name, url=url, user_agent=ua, since=since)
        _PER_RUN_FETCH_CACHE[key] = fetched
        all_headlines.extend(fetched)

        # Persist to disk cache (best-effort; never raise into the caller).
        if cache is not None and fetched:
            try:
                cache.store_headlines(feed_name=feed_name, headlines=fetched)
            except Exception as e:  # pragma: no cover - disk failures are noise
                log.debug("feed %s: cache store failed: %s", feed_name, type(e).__name__)

    # Stable de-duplication by (source, hash) so a republished headline
    # doesn't double-count in the sentiment scorer.
    seen: set[tuple[str, str]] = set()
    deduped: list[Headline] = []
    for h in all_headlines:
        k = (h.source, h.headline_hash)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(h)

    log.info("news: %d unique headlines across feeds", len(deduped))
    return deduped


def _reset_per_run_cache() -> None:
    """Test seam: clear the module-level in-process cache.

    Production code never calls this; tests do, so successive test
    cases don't inherit each other's data.
    """
    _PER_RUN_FETCH_CACHE.clear()
