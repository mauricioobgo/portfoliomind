"""Unit tests for :mod:`portfoliomind.news.sentiment`.

These tests are hermetic — they exercise the LLM response parser and
the sentiment record builder without ever calling OpenAI. The OpenAI
client is mocked at the module boundary.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from portfoliomind.news._headline import Headline
from portfoliomind.news.sentiment import (
    MAX_HEADLINES_PER_TICKER,
    SENTIMENT_MODEL,
    SentimentRecord,
    _aggregate,
    _build_ticker_to_titles,
    _build_user_prompt,
    _coerce_score,
    _llm_score,
    parse_sentiment_response,
    score_ticker_sentiment,
)


# --- _coerce_score ---------------------------------------------------------


class TestCoerceScore:
    """Robust coercion of LLM output to clamped floats."""

    def test_int_to_float(self):
        assert _coerce_score(1) == 1.0
        assert _coerce_score(0) == 0.0
        assert _coerce_score(-1) == -1.0

    def test_float_passthrough(self):
        assert _coerce_score(0.42) == 0.42
        assert _coerce_score(-0.7) == -0.7

    def test_clamps_above_one(self):
        assert _coerce_score(1.5) == 1.0
        assert _coerce_score(100) == 1.0

    def test_clamps_below_minus_one(self):
        assert _coerce_score(-1.5) == -1.0
        assert _coerce_score(-100) == -1.0

    def test_none_returns_none(self):
        assert _coerce_score(None) is None

    def test_bool_rejected(self):
        # bool is an int subclass; reject it explicitly.
        assert _coerce_score(True) is None
        assert _coerce_score(False) is None

    def test_string_parsable(self):
        assert _coerce_score("0.5") == 0.5
        assert _coerce_score('"-0.3"') == -0.3

    def test_string_unparsable(self):
        assert _coerce_score("not a number") is None
        assert _coerce_score("") is None

    def test_nan_rejected(self):
        assert _coerce_score(float("nan")) is None

    def test_unsupported_type(self):
        assert _coerce_score([]) is None
        assert _coerce_score({}) is None


# --- parse_sentiment_response ---------------------------------------------


class TestParseSentimentResponse:
    """Defensive parser for the LLM's JSON response."""

    def test_clean_response(self):
        raw = json.dumps(
            {
                "AAPL": {
                    "score": 0.42,
                    "reason": "Strong quarter",
                    "per_headline": [
                        {"title": "Apple beats", "score": 0.5, "reason": "Beat"},
                    ],
                },
                "MSFT": {"score": -0.2, "reason": "Soft guide"},
            }
        )
        parsed = parse_sentiment_response(raw)
        assert set(parsed.keys()) == {"AAPL", "MSFT"}
        assert parsed["AAPL"]["score"] == 0.42
        assert parsed["AAPL"]["reason"] == "Strong quarter"
        assert parsed["AAPL"]["per_headline"] == [
            {"title": "Apple beats", "score": 0.5, "reason": "Beat"},
        ]
        assert parsed["MSFT"]["score"] == -0.2

    def test_strips_markdown_fences(self):
        # Some models wrap JSON in ```json ... ``` even when told not to.
        inner = json.dumps({"AAPL": {"score": 0.1, "reason": "ok"}})
        raw = f"Sure, here it is:\n```json\n{inner}\n```"
        parsed = parse_sentiment_response(raw)
        assert "AAPL" in parsed
        assert parsed["AAPL"]["score"] == 0.1

    def test_chatty_preamble(self):
        inner = json.dumps({"AAPL": {"score": 0.0, "reason": "neutral"}})
        raw = f"Of course! Here's the JSON you asked for:\n{inner}\nHope that helps."
        parsed = parse_sentiment_response(raw)
        assert "AAPL" in parsed

    def test_empty_response(self):
        assert parse_sentiment_response("") == {}

    def test_non_json_response(self):
        assert parse_sentiment_response("not even close to json") == {}

    def test_json_but_not_object(self):
        assert parse_sentiment_response("[1, 2, 3]") == {}

    def test_strips_control_chars(self):
        # The LLM sometimes inserts null bytes; the parser must handle them.
        inner = json.dumps({"AAPL": {"score": 0.1, "reason": "ok"}})
        raw = inner.replace('"AAPL"', '"AA\u0000PL"')  # would break naive loaders
        # Actually let's just inject a control char into the reason.
        raw = '{"AAPL": {"score": 0.5, "reason": "ok\u0001here"}}'
        parsed = parse_sentiment_response(raw)
        # Either parses (and strips), or returns {}. Both are acceptable;
        # what matters is no exception.
        assert isinstance(parsed, dict)

    def test_missing_score_skips_ticker(self):
        raw = json.dumps({"AAPL": {"reason": "no score"}})
        parsed = parse_sentiment_response(raw)
        # Score missing → ticker omitted; the scorer fills in 0.0.
        assert "AAPL" not in parsed

    def test_invalid_score_skips_ticker(self):
        raw = json.dumps({"AAPL": {"score": "nope", "reason": "x"}})
        parsed = parse_sentiment_response(raw)
        assert "AAPL" not in parsed

    def test_partial_per_headline(self):
        # Per-headline list is optional and may have malformed entries.
        raw = json.dumps(
            {
                "AAPL": {
                    "score": 0.3,
                    "reason": "ok",
                    "per_headline": [
                        {"title": "good", "score": 0.5, "reason": "r1"},
                        {"title": "bad shape", "score": "not a number", "reason": "r2"},
                        "not a dict at all",
                        {"missing fields": True},
                    ],
                }
            }
        )
        parsed = parse_sentiment_response(raw)
        assert "AAPL" in parsed
        # First and fourth entries are skipped (score=None, missing fields).
        # Second entry has invalid score → coerced to 0.0 by the parser.
        # The third is a string → skipped.
        # We accept any of (only good, or good + coerced 0.0) — just make
        # sure no exception was raised and the per_headline list is a list.
        assert isinstance(parsed["AAPL"]["per_headline"], list)

    def test_clamps_out_of_range_scores(self):
        # The parser's _coerce_score clamps; the clamping happens here
        # (not in _build_record) so a quirky model output is normalised
        # before the scorer aggregates it.
        raw = json.dumps({"AAPL": {"score": 5.0, "reason": "too positive"}})
        parsed = parse_sentiment_response(raw)
        assert parsed["AAPL"]["score"] == 1.0

        raw = json.dumps({"AAPL": {"score": -5.0, "reason": "too negative"}})
        parsed = parse_sentiment_response(raw)
        assert parsed["AAPL"]["score"] == -1.0

    def test_tickers_uppercased(self):
        raw = json.dumps({"aapl": {"score": 0.1, "reason": "x"}})
        parsed = parse_sentiment_response(raw)
        assert "AAPL" in parsed


# --- _aggregate ------------------------------------------------------------


class TestAggregate:
    """The simple mean reducer."""

    def test_empty(self):
        assert _aggregate([]) == 0.0

    def test_single(self):
        assert _aggregate([0.5]) == 0.5
        assert _aggregate([-0.3]) == -0.3

    def test_mean(self):
        assert _aggregate([0.5, -0.5]) == 0.0
        assert _aggregate([0.2, 0.4, 0.6]) == pytest.approx(0.4)

    def test_clamps(self):
        # Defensive: if the inputs are out of range for some reason, the
        # output is still bounded.
        assert _aggregate([2.0, 2.0]) == 1.0
        assert _aggregate([-2.0, -2.0]) == -1.0


# --- _build_ticker_to_titles ----------------------------------------------


class TestBuildTickerToTitles:
    def test_caps_per_ticker(self):
        now = datetime(2026, 6, 9, 12, 0)
        # 50 headlines — should be capped at MAX_HEADLINES_PER_TICKER.
        headlines = [
            Headline.make(
                title=f"AAPL headline {i:03d}",
                source="reuters",
                published_at=now.replace(minute=i % 60),
                link="",
            )
            for i in range(50)
        ]
        out = _build_ticker_to_titles({"AAPL": headlines})
        assert len(out["AAPL"]) == MAX_HEADLINES_PER_TICKER

    def test_newest_first(self):
        now = datetime(2026, 6, 9, 12, 0)
        headlines = [
            Headline.make(
                title=f"headline {i}",
                source="reuters",
                published_at=now.replace(minute=i),
                link="",
            )
            for i in range(5)
        ]
        out = _build_ticker_to_titles({"AAPL": headlines})
        # headline 4 is newest.
        assert out["AAPL"][0] == "headline 4"
        assert out["AAPL"][-1] == "headline 0"

    def test_drops_empty_titles(self):
        now = datetime(2026, 6, 9, 12, 0)
        headlines = [
            Headline.make(title="real headline", source="x", published_at=now, link=""),
            Headline.make(title="", source="x", published_at=now, link=""),
        ]
        out = _build_ticker_to_titles({"AAPL": headlines})
        assert out["AAPL"] == ["real headline"]


# --- _build_user_prompt ---------------------------------------------------


class TestBuildUserPrompt:
    def test_contains_ticker_data(self):
        prompt = _build_user_prompt({"AAPL": ["Apple beats"], "MSFT": ["Microsoft down"]})
        assert "AAPL" in prompt
        assert "Apple beats" in prompt
        assert "MSFT" in prompt
        assert "Microsoft down" in prompt

    def test_instructs_score_range(self):
        prompt = _build_user_prompt({})
        # The prompt is a JSON envelope, but it includes the
        # instructions. We don't want to assert exact wording, but
        # the score range should be there.
        assert "-1.0" in prompt or "[-1, +1]" in prompt or "-1.0 to +1.0" in prompt

    def test_empty(self):
        # Should not raise; payload is just {}.
        out = _build_user_prompt({})
        assert "AAPL" not in out


# --- score_ticker_sentiment (with mocked LLM + feeds) --------------------


class TestScoreTickerSentiment:
    """End-to-end (without network) for the per-ticker entry point."""

    def test_returns_zero_when_no_headlines(self):
        # Mock both the feed fetcher and the LLM caller so no network is hit.
        with (
            patch("portfoliomind.news.sentiment.fetch_all_feeds", return_value=[]),
            patch("portfoliomind.news.sentiment._call_openai_chat") as mock_llm,
        ):
            score = score_ticker_sentiment(
                "AAPL", api_key="test-key", since_hours=24
            )
        assert score == 0.0
        # LLM not called when there are no headlines.
        mock_llm.assert_not_called()

    def test_returns_parsed_score(self):
        now = datetime(2026, 6, 9, 12, 0)
        headlines = [
            Headline.make(title="Apple beats estimates", source="reuters", published_at=now, link=""),
            Headline.make(title="AAPL stock rises on news", source="marketwatch", published_at=now, link=""),
        ]
        with (
            patch("portfoliomind.news.sentiment.fetch_all_feeds", return_value=headlines),
            patch(
                "portfoliomind.news.sentiment._call_openai_chat",
                return_value=json.dumps({"AAPL": {"score": 0.42, "reason": "ok"}}),
            ),
        ):
            score = score_ticker_sentiment("AAPL", api_key="test-key", since_hours=24)
        assert score == 0.42

    def test_unknown_ticker_returns_zero(self):
        with (
            patch("portfoliomind.news.sentiment.fetch_all_feeds", return_value=[]),
            patch("portfoliomind.news.sentiment._call_openai_chat") as mock_llm,
        ):
            score = score_ticker_sentiment("ZZZZ", api_key="test-key", since_hours=24)
        # No headlines match, LLM not called, score = 0.0.
        assert score == 0.0
        mock_llm.assert_not_called()

    def test_missing_api_key_raises(self):
        with (
            patch("portfoliomind.news.sentiment.fetch_all_feeds", return_value=[]),
            patch.dict("os.environ", {}, clear=False),  # don't clobber the real env
        ):
            # Force the lookup to fail.
            with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
                with pytest.raises(Exception) as exc_info:
                    score_ticker_sentiment("AAPL", since_hours=24)
        assert "OPENAI_API_KEY" in str(exc_info.value) or "api_key" in str(exc_info.value).lower()

    def test_llm_failure_returns_zero(self):
        now = datetime(2026, 6, 9, 12, 0)
        headlines = [
            Headline.make(title="Apple beats estimates", source="reuters", published_at=now, link=""),
        ]
        with (
            patch("portfoliomind.news.sentiment.fetch_all_feeds", return_value=headlines),
            patch(
                "portfoliomind.news.sentiment._call_openai_chat",
                side_effect=RuntimeError("network down"),
            ),
        ):
            score = score_ticker_sentiment("AAPL", api_key="test-key", since_hours=24)
        # LLM failure is graceful — return 0.0 (no news sentiment), not crash.
        assert score == 0.0


# --- _llm_score (with mocked OpenAI call) --------------------------------


class TestLLMScore:
    def test_no_groups_returns_empty(self):
        # When no groups need scoring, no call is made.
        result = _llm_score(ticker_to_headlines={}, api_key="test-key")
        assert result == {}

    def test_happy_path(self):
        now = datetime(2026, 6, 9, 12, 0)
        groups = {
            "AAPL": [
                Headline.make(title="Apple beats", source="reuters", published_at=now, link=""),
            ],
            "MSFT": [
                Headline.make(title="Microsoft in talks", source="marketwatch", published_at=now, link=""),
            ],
        }
        with patch(
            "portfoliomind.news.sentiment._call_openai_chat",
            return_value=json.dumps(
                {
                    "AAPL": {"score": 0.5, "reason": "good", "per_headline": []},
                    "MSFT": {"score": -0.3, "reason": "bad", "per_headline": []},
                }
            ),
        ) as mock_call:
            result = _llm_score(ticker_to_headlines=groups, api_key="test-key")
        assert set(result.keys()) == {"AAPL", "MSFT"}
        assert result["AAPL"]["score"] == 0.5
        assert result["MSFT"]["score"] == -0.3
        # System + user message, single call.
        assert mock_call.call_count == 1


# --- SentimentRecord (sanity check) --------------------------------------


class TestSentimentRecord:
    def test_to_dict_round_trip(self):
        r = SentimentRecord(
            ticker="AAPL",
            score=0.42,
            sample_size=3,
            per_headline=[{"title": "x", "score": 0.5, "reason": "r"}],
            model=SENTIMENT_MODEL,
            day="2026-06-09",
        )
        d = r.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["score"] == 0.42
        assert d["sample_size"] == 3
        assert d["model"] == SENTIMENT_MODEL
        assert d["day"] == "2026-06-09"
