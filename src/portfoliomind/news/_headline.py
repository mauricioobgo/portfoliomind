"""The :class:`Headline` dataclass.

Lives in its own module to break the circular import between
:mod:`portfoliomind.news.feeds` (which constructs ``Headline``) and
:mod:`portfoliomind.news.store` (which stores them). Either side can
import from here without pulling in the other.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Headline:
    """A single news headline with provenance + timestamp.

    The fields are deliberately minimal — anything richer (summary,
    link popularity, etc.) is dropped here so the news-match and
    sentiment layers stay simple.

    ``headline_hash`` is a stable sha256 of the title; the cache and
    the matcher both key off it to avoid treating re-published
    duplicates as fresh data.
    """

    title: str
    source: str
    published_at: datetime
    link: str
    headline_hash: str

    @staticmethod
    def make(title: str, source: str, published_at: datetime, link: str) -> "Headline":
        return Headline(
            title=title,
            source=source,
            published_at=published_at,
            link=link,
            headline_hash=hashlib.sha256(title.encode("utf-8")).hexdigest(),
        )


__all__ = ["Headline"]
