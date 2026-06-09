"""News ingestion + LLM sentiment scoring (card 5 of PortfolioMind v4).

Public surface:

* :func:`recent_headlines` — fetch the last ``since_hours`` of headlines
  for a single ticker. Used for debugging / operator inspection.
* :func:`score_ticker_sentiment` — same data plus an LLM sentiment score
  in ``[-1, +1]`` for one ticker.
* :func:`score_universe_sentiment` — single batched LLM call for the
  whole universe. This is the one production entry point.

Design constraints (from the card 5 spec):

* One OpenAI call per run, batched across the universe. Cap at 30
  headlines per ticker to bound the per-run cost.
* No raw headline text at INFO log level; DEBUG only.
* LLM prompts are NEVER logged (they contain ticker aliases that look
  like financial advice).
* RSS feeds fail-soft: one feed down does not break the others.
* Idempotent: a re-run in the same day returns identical scores via the
  SQLite cache; a re-run the next day re-scores.

Card 6 (technical + combined signal) imports :func:`score_ticker_sentiment`
and combines the news sentiment with technical indicators. The cache in
:mod:`portfoliomind.news.store` is the right place to add technical-data
caching too (per the card 5 handoff).
"""

from __future__ import annotations

from .feeds import Headline, fetch_all_feeds
from .match import match_headlines_to_universe
from .sentiment import (
    SENTIMENT_MODEL,
    score_ticker_sentiment,
    score_universe_sentiment,
)
from .store import HEADLINE_CACHE_SCHEMA_VERSION, NewsCache

__all__ = [
    "Headline",
    "fetch_all_feeds",
    "match_headlines_to_universe",
    "score_ticker_sentiment",
    "score_universe_sentiment",
    "NewsCache",
    "SENTIMENT_MODEL",
    "HEADLINE_CACHE_SCHEMA_VERSION",
]
