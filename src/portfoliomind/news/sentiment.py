"""LLM sentiment scoring for the news module.

The scorer is the production entry point for card 5. It batches all
tickers into a single OpenAI call (one call per run, regardless of
how many tickers are in the universe) and returns per-ticker
sentiment scores in ``[-1, +1]``.

Design choices (from the spec):

* **Model**: ``gpt-4o-mini``. Cheap, fast, and good enough for the
  short financial-headline classification we ask of it. Spec'd in
  the card 5 description; we expose ``SENTIMENT_MODEL`` for tests.
* **Batching**: all (ticker, headlines) pairs go in a single
  request. The OpenAI Chat Completions API handles 100+ items at
  this size in well under 5s.
* **Per-ticker cap**: ``MAX_HEADLINES_PER_TICKER`` (30). Bounds
  token cost — a 45-ticker universe at 30 headlines each is ~500
  tokens per item, well under the input window.
* **Idempotency**: the per-(ticker, day) result is cached. A re-run
  the same day returns the cached score without a network call.
* **Privacy**: LLM prompts are NEVER logged. The prompts contain
  ticker aliases that, taken in isolation, look like financial
  advice — we keep them out of AGENT_LOG.

Output contract:

* :func:`score_ticker_sentiment` returns a single ``float`` in
  ``[-1, +1]`` for one ticker (or 0.0 when no headlines matched).
* :func:`score_universe_sentiment` returns ``{ticker: float}`` for
  the whole universe, including 0.0 entries for tickers with no news
  (so the strategy engine can iterate over a stable map).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, cast

from ..logging_setup import get_logger
from ..time_utils import BOGOTA_TZ, now_bogota
from ..universe import UNIVERSE
from .feeds import Headline, fetch_all_feeds
from .match import match_headlines_to_universe
from .store import NewsCache

log = get_logger(__name__)


#: The OpenAI model used for sentiment scoring. Spec'd in card 5.
#: Exposed so tests can substitute a fake without touching the network.
SENTIMENT_MODEL: str = "gpt-4o-mini"

#: Cap on the number of headlines we send per ticker. Bounds cost.
MAX_HEADLINES_PER_TICKER: int = 30

#: The OpenAI request timeout. The 4o-mini model is fast but we still
#: set a generous timeout because the morning run is otherwise head-of-line
#: blocked by a stalled scoring call.
SENTIMENT_TIMEOUT_S: float = 30.0

#: Minimum confidence we require for a ticker to be considered "scored"
#: (i.e. we have at least one headline to score against). A ticker with
#: fewer than this many matched headlines returns 0.0 by default — it
#: is the caller's job to decide whether 0.0 is "neutral" or "missing".
MIN_HEADLINES_TO_SCORE: int = 1


# --- LLM response parsing ---------------------------------------------------


@dataclass(frozen=True)
class SentimentRecord:
    """One ticker's sentiment, with provenance + per-headline detail.

    This is what the demo script prints and what card 6 will consume.
    """

    ticker: str
    score: float  # [-1, +1]
    sample_size: int
    per_headline: list[dict]  # [{"title": str, "score": float, "reason": str}]
    model: str
    day: str  # YYYY-MM-DD Bogota

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "score": self.score,
            "sample_size": self.sample_size,
            "per_headline": list(self.per_headline),
            "model": self.model,
            "day": self.day,
        }


class SentimentError(RuntimeError):
    """Raised when the LLM call fails irrecoverably or the response is unparseable."""


def _coerce_score(value: Any) -> Optional[float]:
    """Coerce a parsed-JSON value to a clamped float in [-1, +1].

    Returns None if the value cannot be coerced to a number (so the
    caller can skip the entry rather than treat it as 0.0, which would
    bias the aggregate).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is a subclass of int — exclude it explicitly to avoid
        # True→1.0 sneaking through.
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN
            return None
        return max(-1.0, min(1.0, f))
    if isinstance(value, str):
        # Some models wrap the number in quotes despite the JSON request.
        s = value.strip().strip("\"'")
        try:
            f = float(s)
        except ValueError:
            return None
        return max(-1.0, min(1.0, f))
    return None


def parse_sentiment_response(raw: str) -> dict[str, dict]:
    """Parse the LLM's response into a ``{ticker: {score, reason, per_headline}}`` map.

    The LLM is asked to return a JSON object of the form::

        {
          "AAPL": {"score": 0.42, "reason": "..."},
          ...
        }

    with an optional ``per_headline`` key inside each ticker for the
    per-headline scores used in the aggregate.

    The parser is defensive:

    * Strips ``\u0000`` and other control chars that break ``json.loads``.
    * Falls back to extracting the first JSON object via regex if the
      response is wrapped in prose ("Sure! Here's the JSON: {...}").
    * Per-ticker score coercion is lenient (strings, ints, floats all
      accepted) so a quirky model output still produces a result.
    * Tickers not in the response get a 0.0 default so the output is
      complete.
    """
    if not raw:
        return {}

    # Strip control chars except newlines/tabs/CR.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)

    data: Any = None
    # Try a strict parse first.
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to "first {...} in the string" extraction. The LLM
        # sometimes wraps its answer in a chatty preamble.
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if not isinstance(data, dict):
        return {}

    out: dict[str, dict] = {}
    for ticker, payload in data.items():
        if not isinstance(ticker, str) or not isinstance(payload, dict):
            continue
        score = _coerce_score(payload.get("score"))
        if score is None:
            # No usable score — skip; caller will fill in 0.0 for the
            # missing ticker.
            continue
        reason = payload.get("reason") or ""
        if not isinstance(reason, str):
            reason = str(reason)
        per = payload.get("per_headline") or []
        per_clean: list[dict] = []
        if isinstance(per, list):
            for entry in per:
                if not isinstance(entry, dict):
                    continue
                title = entry.get("title") or ""
                ph_score = _coerce_score(entry.get("score"))
                ph_reason = entry.get("reason") or ""
                if not isinstance(title, str) or not isinstance(ph_reason, str):
                    continue
                per_clean.append(
                    {
                        "title": title,
                        "score": ph_score if ph_score is not None else 0.0,
                        "reason": ph_reason,
                    }
                )
        out[ticker.upper()] = {
            "score": score,
            "reason": reason,
            "per_headline": per_clean,
        }
    return out


# --- OpenAI client (lazy import) -------------------------------------------


def _call_openai_chat(
    *,
    messages: list[dict],
    model: str = SENTIMENT_MODEL,
    api_key: str,
    timeout_s: float = SENTIMENT_TIMEOUT_S,
) -> str:
    """Call the OpenAI Chat Completions API and return the response text.

    The OpenAI client is constructed here (not at module import time)
    so that:

    * Tests can monkeypatch this function and skip the real import.
    * The operator's environment can change the API key between runs
      without restarting the process.
    * The morning job's optional ``--dry-run`` mode never imports
      the openai library at all.
    """
    # Lazy import so importing portfoliomind.news.sentiment doesn't
    # require the openai package to be installed (handy for unit tests
    # that only exercise the parser).
    from openai import OpenAI  # type: ignore[import-not-found]

    client = OpenAI(api_key=api_key, timeout=timeout_s)
    # NOTE: we deliberately do NOT log messages — they contain ticker
    # aliases that look like financial advice. See card 5 spec.
    resp = client.chat.completions.create(
        model=model,
        messages=cast(Any, messages),  # openai's TypedDicts are picky; our shape is correct
        temperature=0.0,  # Deterministic for cacheability.
        response_format={"type": "json_object"},
    )
    if not resp.choices:
        raise SentimentError("OpenAI returned no choices")
    content = resp.choices[0].message.content or ""
    return content


# --- Prompt construction ---------------------------------------------------


def _build_user_prompt(ticker_to_headlines: dict[str, list[str]]) -> str:
    """Build the user-side prompt for the batched LLM call.

    We send JSON so the model has structural guidance. The system
    prompt (separate, fixed) sets the role; this one carries the data.
    """
    payload: dict[str, list[str]] = {}
    for ticker, titles in ticker_to_headlines.items():
        # Drop empties just in case.
        payload[ticker] = [t for t in titles if t]
    return (
        "You are a financial-news sentiment classifier. For each ticker "
        "below, score the OVERALL sentiment of its headlines on a scale "
        "of -1.0 (very negative) to +1.0 (very positive). A score of 0.0 "
        "means neutral or mixed. Use the per-headline scores to inform "
        "your overall call.\n\n"
        "Return STRICT JSON of the form:\n"
        '{\n'
        '  "<TICKER>": {\n'
        '    "score": <float -1.0 to 1.0>,\n'
        '    "reason": "<one short sentence, <120 chars>",\n'
        '    "per_headline": [\n'
        '      {"title": "<verbatim headline>", "score": <float -1.0 to 1.0>, "reason": "<one short phrase>"}\n'
        '    ]\n'
        '  },\n'
        '  ...\n'
        '}\n\n'
        "Tickers and their headlines (JSON, oldest first omitted — these are the latest):\n"
        + json.dumps(payload, ensure_ascii=False)
    )


_SYSTEM_PROMPT: str = (
    "You are a financial-news sentiment analyst. You receive headlines "
    "grouped by ticker and return a strict JSON sentiment score per "
    "ticker on a -1.0 to +1.0 scale. Be conservative: mixed news "
    "warrants a score near 0.0. Do not editorialize. Do not include "
    "markdown fences in your response."
)


# --- Public API -------------------------------------------------------------


def _day_key(today: Optional[datetime] = None) -> str:
    """Return the YYYY-MM-DD day key in Bogota time."""
    dt = today if today is not None else now_bogota()
    return dt.astimezone(BOGOTA_TZ).strftime("%Y-%m-%d")


def _aggregate(scores: list[float]) -> float:
    """Reduce a list of per-headline scores to one aggregate in [-1, +1].

    We use the simple arithmetic mean. An alternative is to weight
    recent headlines more heavily, but the spec calls for "a sentiment
    score" (singular) and a mean is the least surprising aggregate.
    Clamps to [-1, +1] defensively.
    """
    if not scores:
        return 0.0
    m = sum(scores) / len(scores)
    return max(-1.0, min(1.0, m))


def _build_ticker_to_titles(
    grouped: dict[str, list[Headline]],
    *,
    cap: int = MAX_HEADLINES_PER_TICKER,
) -> dict[str, list[str]]:
    """Trim the per-ticker headline list to ``cap`` entries, newest first.

    Empty titles (a malformed feed result) are dropped — the LLM
    doesn't need to know we ever saw a blank string.
    """
    out: dict[str, list[str]] = {}
    for ticker, headlines in grouped.items():
        # Sort newest first so the cap drops the OLDEST, not the newest.
        sorted_hs = sorted(headlines, key=lambda h: h.published_at, reverse=True)
        titles = [h.title for h in sorted_hs[:cap] if h.title]
        out[ticker] = titles
    return out


def _llm_score(
    *,
    ticker_to_headlines: dict[str, list[Headline]],
    api_key: str,
    model: str = SENTIMENT_MODEL,
) -> dict[str, dict]:
    """Call the LLM and parse the response. Returns ``{ticker: parsed_dict}``."""
    if not ticker_to_headlines:
        return {}
    titles_by_ticker = _build_ticker_to_titles(ticker_to_headlines)
    user_prompt = _build_user_prompt(titles_by_ticker)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = _call_openai_chat(messages=messages, model=model, api_key=api_key)
    return parse_sentiment_response(raw)


def _build_record(
    ticker: str,
    *,
    parsed: dict,
    sample_size: int,
    model: str,
    day: str,
) -> SentimentRecord:
    """Build a :class:`SentimentRecord` from the parsed LLM output."""
    score = float(parsed.get("score", 0.0))
    score = max(-1.0, min(1.0, score))
    return SentimentRecord(
        ticker=ticker,
        score=score,
        sample_size=sample_size,
        per_headline=parsed.get("per_headline", []),
        model=model,
        day=day,
    )


def score_ticker_sentiment(
    ticker: str,
    *,
    since_hours: int = 24,
    api_key: Optional[str] = None,
    cache: Optional[NewsCache] = None,
    today: Optional[datetime] = None,
) -> float:
    """Return a sentiment score in [-1, +1] for ``ticker``.

    Public entry point for card 6 / debugging. Internally delegates to
    :func:`score_universe_sentiment` so the per-ticker path uses the
    same single-batched LLM call as the per-universe path.
    """
    ticker = ticker.upper()
    result = score_universe_sentiment(
        tickers=(ticker,),
        since_hours=since_hours,
        api_key=api_key,
        cache=cache,
        today=today,
    )
    return result.get(ticker, 0.0)


def recent_headlines(
    ticker: str,
    *,
    since_hours: int = 24,
    cache: Optional[NewsCache] = None,
) -> list[Headline]:
    """Return raw ``Headline`` records for ``ticker`` over the last ``since_hours``.

    Useful for operator inspection / debugging. Hits the same caches
    as :func:`score_ticker_sentiment`.
    """
    ticker = ticker.upper()
    all_h = fetch_all_feeds(since_hours=since_hours, cache=cache)
    grouped = match_headlines_to_universe(all_h)
    return list(grouped.get(ticker, []))


def score_universe_sentiment(
    tickers: tuple[str, ...] = UNIVERSE,
    *,
    since_hours: int = 24,
    api_key: Optional[str] = None,
    cache: Optional[NewsCache] = None,
    today: Optional[datetime] = None,
    model: str = SENTIMENT_MODEL,
) -> dict[str, float]:
    """Score the supplied ``tickers`` (default: the full universe).

    Returns ``{ticker: score}`` where every input ticker appears in the
    output (tickers with no matched headlines get 0.0). The same-day
    cache is consulted before the LLM call; tickers with a cached
    score for today skip the LLM entirely.

    A re-run on the same day returns identical scores (cache hit).
    A re-run on a new day triggers a fresh LLM call.
    """
    if not tickers:
        return {}
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SentimentError(
            "OPENAI_API_KEY is not set; cannot score sentiment. "
            "Pass api_key=... or export the env var."
        )

    day = _day_key(today)
    upper_tickers = tuple(t.upper() for t in tickers)
    scores: dict[str, float] = {}

    # Group by cache state: hit (skip LLM) vs miss (need scoring).
    to_score_groups: dict[str, list[Headline]] = {}
    if cache is not None:
        for t in upper_tickers:
            cached = cache.fetch_sentiment(ticker=t, day=day)
            if cached is not None:
                scores[t] = float(cached["score"])
            else:
                # We need to know which headlines go with this ticker
                # to know the sample size; gather everything then
                # partition below.
                pass

    # Determine which tickers need an LLM call.
    needs_llm = [t for t in upper_tickers if t not in scores]
    if not needs_llm:
        log.info("sentiment: %d/%d tickers served from cache for %s", len(scores), len(upper_tickers), day)
        return {t: scores.get(t, 0.0) for t in upper_tickers}

    # Fetch + match once for the whole window.
    all_headlines = fetch_all_feeds(since_hours=since_hours, cache=cache)
    grouped_all = match_headlines_to_universe(all_headlines)

    # Build the per-ticker headline input for the LLM: only the tickers
    # that actually need scoring, and only the ones that have at least
    # one matched headline.
    to_score_groups = {t: grouped_all.get(t, []) for t in needs_llm}
    to_score_groups = {t: hs for t, hs in to_score_groups.items() if hs}

    parsed: dict[str, dict] = {}
    if to_score_groups:
        try:
            parsed = _llm_score(
                ticker_to_headlines=to_score_groups,
                api_key=api_key,
                model=model,
            )
        except Exception as e:
            # The LLM call itself is the only thing that can fail
            # irrecoverably. Network blips, schema mismatches, rate
            # limits — all fall here. We log and return what we have
            # (cache hits + 0.0 for the rest). The morning job is
            # never blocked by a sentiment failure.
            log.warning(
                "sentiment: LLM call failed (%s) — %d tickers will use 0.0 default",
                type(e).__name__,
                sum(1 for t in needs_llm if t not in scores),
            )
            parsed = {}

        # Merge parsed LLM output + per-ticker persistence.
        for t in needs_llm:
            if t in parsed:
                rec = _build_record(
                    t,
                    parsed=parsed[t],
                    sample_size=len(to_score_groups.get(t, [])),
                    model=model,
                    day=day,
                )
                scores[t] = rec.score
                if cache is not None:
                    try:
                        cache.store_sentiment(
                            ticker=t,
                            day=day,
                            score=rec.score,
                            sample_size=rec.sample_size,
                            per_headline=rec.per_headline,
                            model=rec.model,
                        )
                    except Exception as e:  # pragma: no cover
                        log.debug("sentiment: cache store failed for %s: %s", t, type(e).__name__)
            else:
                # LLM did not return a score for this ticker (or LLM
                # call failed). Default to 0.0 — the strategy engine
                # treats 0.0 as "no news sentiment" not "negative".
                scores[t] = 0.0

    # Tickers that needed scoring but had no headlines also default to 0.0.
    for t in needs_llm:
        scores.setdefault(t, 0.0)

    log.info(
        "sentiment: %d/%d tickers scored (cache=%d, llm=%d, day=%s)",
        len([t for t in upper_tickers if t in scores]),
        len(upper_tickers),
        len(upper_tickers) - len(needs_llm),
        len(to_score_groups),
        day,
    )
    return {t: scores.get(t, 0.0) for t in upper_tickers}


__all__ = [
    "SENTIMENT_MODEL",
    "MAX_HEADLINES_PER_TICKER",
    "MIN_HEADLINES_TO_SCORE",
    "SentimentRecord",
    "SentimentError",
    "score_ticker_sentiment",
    "score_universe_sentiment",
    "recent_headlines",
    "parse_sentiment_response",
    "_aggregate",
    "_coerce_score",
    "_llm_score",
    "_build_user_prompt",
]
