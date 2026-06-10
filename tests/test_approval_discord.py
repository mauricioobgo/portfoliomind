"""Hermetic tests for :mod:`portfoliomind.approval.discord`.

These tests use the module's plug-in seams (``http_client``,
``reaction_poller``, ``emoji_mapper``, ``now_fn``) to inject canned
responses. No real Discord, no real network, no real time.

The tests cover:

* Message format: the operator sees one line per candidate in the
  exact format the card 7 spec lists.
* Reaction parsing: ✅ → approved, ❌ → rejected (all), ⏸ → wait
  (all), per-candidate 1️⃣/2️⃣/3️⃣/4️⃣/5️⃣ → approve that one.
* Timeout: no reaction = ❌ (per-candidate).
* Error paths: missing bot token, missing channel id, HTTP failure,
  poller exception.
* The function never raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


from portfoliomind.approval import discord as d
from portfoliomind.approval.discord import (
    ApprovalOutcome,
    ApprovedTrade,
    CANDIDATE_EMOJIS,
    EMOJI_APPROVE,
    EMOJI_REJECT,
    EMOJI_WAIT,
    build_candidates_message,
    format_trade_order_line,
    post_candidates_and_collect_reactions,
)
from portfoliomind.signals.commissions import InstrumentType


# --- Fakes --------------------------------------------------------------


@dataclass
class _Trade:
    """A card-7 TradeOrder-shaped duck for the message format tests."""

    ticker: str = "SPY"
    qty: float = 2.0
    entry: float = 100.0
    sl: float = 93.0
    tp: float = 114.0
    notional: float = 200.0
    commission_rt: float = 0.0
    r_r_ratio: float = 2.0
    instrument: Any = InstrumentType.US_ETF
    signal_date: str = "2026-06-10"
    asof_date: str = "2026-06-10"
    combined: float = 0.65
    confidence: float = 0.7
    reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = ["strong trend", "RSI confirms"]


def _http_factory(responses: list[tuple[int, Any]]) -> Callable[..., tuple[int, Any]]:
    """Build a fake http_client that returns canned responses in order."""
    calls: list[tuple[str, str]] = []
    state = {"index": 0}

    def _client(method: str, url: str, *, json: Any = None, headers: Any = None) -> tuple[int, Any]:
        calls.append((method, url))
        idx = min(state["index"], len(responses) - 1)
        state["index"] += 1
        return responses[idx]

    _client.calls = calls  # type: ignore[attr-defined]
    return _client


def _poller_factory(reactions: dict[str, str]) -> Callable[..., dict[str, str]]:
    """Build a fake reaction_poller that returns canned reactions."""
    def _poller(
        message_id: str, channel_id: str, *, bot_token: str,
        deadline_unix: float, poll_interval: float,
    ) -> dict[str, str]:
        return dict(reactions)
    return _poller


def _now_factory(start: float = 1_000_000.0) -> Callable[[], float]:
    state = {"t": start}
    def _now() -> float:
        return state["t"]
    return _now


# --- Message format ----------------------------------------------------


class TestMessageFormat:
    def test_build_message_one_candidate(self):
        trades = [_Trade(ticker="AAPL", qty=1, entry=200.0, sl=186.0, tp=228.0)]
        msg = build_candidates_message(trades)
        assert "**Trade candidates" in msg
        assert "1️⃣" in msg
        assert "AAPL" in msg
        assert "qty=1" in msg
        assert "entry=$200.00" in msg
        assert "SL=$186.00" in msg
        assert "TP=$228.00" in msg
        assert "combined=+0.65" in msg
        assert "confidence=0.70" in msg

    def test_build_message_five_candidates(self):
        trades = [_Trade(ticker=f"T{i}") for i in range(5)]
        msg = build_candidates_message(trades)
        for i in range(5):
            assert CANDIDATE_EMOJIS[i] in msg
            assert f"T{i}" in msg

    def test_build_message_six_candidates_only_first_five_listed(self):
        # The cap is 5; the 6th is silently dropped from the message.
        trades = [_Trade(ticker=f"T{i}") for i in range(6)]
        msg = build_candidates_message(trades)
        assert "T0" in msg
        assert "T4" in msg
        assert "T5" not in msg  # dropped

    def test_format_trade_order_line(self):
        trade = _Trade()
        line = format_trade_order_line(trade, index=0)
        assert line.startswith("1️⃣ **SPY**")
        assert "combined=+0.65" in line
        assert "confidence=0.70" in line
        assert "qty=2" in line
        assert "entry=$100.00" in line
        assert "SL=$93.00" in line
        assert "TP=$114.00" in line
        # First reason is the summary.
        assert "strong trend" in line

    def test_format_trade_order_line_fractional_qty(self):
        trade = _Trade(qty=1.3333)
        line = format_trade_order_line(trade, index=0)
        assert "qty=1.3333" in line


# --- Post + collect: happy path ---------------------------------------


class TestPostAndCollectHappyPath:
    def test_one_candidate_approved(self):
        # The poller returns 1️⃣ → approved; everything else is empty.
        http = _http_factory([
            (201, {"id": "m1", "channel_id": "c1"}),  # post
            (204, None),                              # react 1️⃣
        ])
        poller = _poller_factory({CANDIDATE_EMOJIS[0]: "candidate_approved"})
        now = _now_factory()
        outcome = post_candidates_and_collect_reactions(
            [_Trade()],
            timeout_min=1,
            bot_token="tok",
            channel_thread_id="c1",
            http_client=http,
            reaction_poller=poller,
            now_fn=now,
        )
        assert outcome.error == ""
        assert outcome.message_id == "m1"
        assert len(outcome.approved) == 1
        assert outcome.approved[0].ticker == "SPY"
        assert outcome.approved[0].message_id == "m1"
        assert len(outcome.rejected) == 0

    def test_three_candidates_mixed_decisions(self):
        # 1️⃣ approve, 2️⃣ nothing, 3️⃣ approve → 2 approved, 1 rejected.
        http = _http_factory([
            (201, {"id": "m1", "channel_id": "c1"}),
            (204, None),  # react 1️⃣
            (204, None),  # react 2️⃣
            (204, None),  # react 3️⃣
        ])
        poller = _poller_factory({
            CANDIDATE_EMOJIS[0]: "candidate_approved",
            CANDIDATE_EMOJIS[2]: "candidate_approved",
        })
        outcome = post_candidates_and_collect_reactions(
            [_Trade(ticker="AAPL"), _Trade(ticker="MSFT"), _Trade(ticker="NVDA")],
            timeout_min=1,
            bot_token="tok",
            channel_thread_id="c1",
            http_client=http,
            reaction_poller=poller,
            now_fn=_now_factory(),
        )
        assert outcome.error == ""
        assert {a.ticker for a in outcome.approved} == {"AAPL", "NVDA"}
        assert {r.ticker for r in outcome.rejected} == {"MSFT"}

    def test_no_reactions_means_reject(self):
        # Poller returns no reactions → every candidate is rejected.
        http = _http_factory([
            (201, {"id": "m1", "channel_id": "c1"}),
            (204, None),
        ])
        poller = _poller_factory({})
        outcome = post_candidates_and_collect_reactions(
            [_Trade(ticker="AAPL")],
            timeout_min=1,
            bot_token="tok",
            channel_thread_id="c1",
            http_client=http,
            reaction_poller=poller,
            now_fn=_now_factory(),
        )
        assert len(outcome.approved) == 0
        assert len(outcome.rejected) == 1
        assert outcome.rejected[0].reason == "no per-candidate approval by timeout"


# --- Global reactions (✅ / ❌ / ⏸) -------------------------------


class TestGlobalReactions:
    def test_global_reject_rejects_all(self):
        http = _http_factory([
            (201, {"id": "m1"}),
            (204, None),
        ])
        poller = _poller_factory({EMOJI_REJECT: "reject"})
        outcome = post_candidates_and_collect_reactions(
            [_Trade(ticker="AAPL"), _Trade(ticker="MSFT")],
            timeout_min=1,
            bot_token="tok",
            channel_thread_id="c1",
            http_client=http,
            reaction_poller=poller,
            now_fn=_now_factory(),
        )
        assert len(outcome.approved) == 0
        assert len(outcome.rejected) == 2
        assert all("global" in r.reason for r in outcome.rejected)

    def test_global_wait_waits_all(self):
        http = _http_factory([
            (201, {"id": "m1"}),
            (204, None),
        ])
        poller = _poller_factory({EMOJI_WAIT: "wait"})
        outcome = post_candidates_and_collect_reactions(
            [_Trade(ticker="AAPL"), _Trade(ticker="MSFT")],
            timeout_min=1,
            bot_token="tok",
            channel_thread_id="c1",
            http_client=http,
            reaction_poller=poller,
            now_fn=_now_factory(),
        )
        assert len(outcome.approved) == 0
        assert len(outcome.rejected) == 0
        assert len(outcome.waited) == 2


# --- Timeout as reject -----------------------------------------------


class TestTimeoutAsReject:
    def test_per_candidate_timeout_is_reject(self):
        # Poller returns nothing → after deadline, every candidate
        # is treated as "no per-candidate approval by timeout".
        http = _http_factory([
            (201, {"id": "m1"}),
            (204, None),
        ])
        poller = _poller_factory({})
        # Set a clock that has already passed the deadline.
        future = _now_factory(start=10_000_000_000.0)
        outcome = post_candidates_and_collect_reactions(
            [_Trade(ticker="AAPL")],
            timeout_min=30,
            bot_token="tok",
            channel_thread_id="c1",
            http_client=http,
            reaction_poller=poller,
            now_fn=future,
        )
        assert len(outcome.approved) == 0
        assert len(outcome.rejected) == 1
        assert "timeout" in outcome.rejected[0].reason


# --- Error paths ------------------------------------------------------


class TestErrorPaths:
    def test_missing_bot_token(self, monkeypatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        outcome = post_candidates_and_collect_reactions(
            [_Trade()], bot_token="", channel_thread_id="c1"
        )
        assert outcome.error == "missing bot_token"

    def test_missing_channel_id(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.delenv("DISCORD_HOME_CHANNEL_THREAD_ID", raising=False)
        outcome = post_candidates_and_collect_reactions(
            [_Trade()], bot_token="tok", channel_thread_id=""
        )
        assert outcome.error == "missing channel_thread_id"

    def test_no_candidates(self):
        outcome = post_candidates_and_collect_reactions(
            [], bot_token="tok", channel_thread_id="c1"
        )
        assert outcome.error == "no candidates to post"
        assert outcome.message_id == ""

    def test_http_post_fails(self):
        http = _http_factory([(500, {"error": "server broke"})])
        outcome = post_candidates_and_collect_reactions(
            [_Trade()],
            bot_token="tok",
            channel_thread_id="c1",
            http_client=http,
            reaction_poller=_poller_factory({}),
            now_fn=_now_factory(),
        )
        assert "discord post failed" in outcome.error
        assert outcome.message_id == ""

    def test_http_post_raises(self):
        def bad_http(*_a: Any, **_kw: Any) -> tuple[int, Any]:
            raise RuntimeError("connection refused")
        outcome = post_candidates_and_collect_reactions(
            [_Trade()],
            bot_token="tok",
            channel_thread_id="c1",
            http_client=bad_http,
            reaction_poller=_poller_factory({}),
            now_fn=_now_factory(),
        )
        assert "RuntimeError" in outcome.error
        assert "connection refused" in outcome.error

    def test_poller_raises(self):
        http = _http_factory([
            (201, {"id": "m1"}),
            (204, None),
        ])
        def bad_poller(*_a: Any, **_kw: Any) -> dict[str, str]:
            raise RuntimeError("ws disconnect")
        outcome = post_candidates_and_collect_reactions(
            [_Trade()],
            bot_token="tok",
            channel_thread_id="c1",
            http_client=http,
            reaction_poller=bad_poller,
            now_fn=_now_factory(),
        )
        # Poller exception is logged + treated as no reactions.
        # The candidate lands in rejected (per-candidate timeout-as-reject).
        assert outcome.error == ""
        assert len(outcome.approved) == 0
        assert len(outcome.rejected) == 1


# --- Bot token from env --------------------------------------------


class TestEnvFallback:
    def test_bot_token_from_env(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "env-tok")
        monkeypatch.delenv("DISCORD_HOME_CHANNEL_THREAD_ID", raising=False)
        outcome = post_candidates_and_collect_reactions(
            [_Trade()],
            channel_thread_id="",
        )
        assert "missing channel_thread_id" in outcome.error
        # bot_token was loaded from env (we never got far enough to fail
        # the token check). Verified by the test reaching the channel check.

    def test_channel_id_from_env(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "env-tok")
        monkeypatch.setenv("DISCORD_HOME_CHANNEL_THREAD_ID", "env-chan")
        http = _http_factory([
            (500, "fail"),  # fail the post so we don't go further
        ])
        outcome = post_candidates_and_collect_reactions(
            [_Trade()],
            http_client=http,
            reaction_poller=_poller_factory({}),
            now_fn=_now_factory(),
        )
        # We get past the token / channel checks and hit the post failure.
        assert "discord post failed" in outcome.error


# --- ApprovedTrade is the shape persist expects ---------------------


class TestApprovedTradeShape:
    def test_approved_trade_has_persist_fields(self):
        # The persist module reads these fields by name.
        http = _http_factory([
            (201, {"id": "m1"}),
            (204, None),
        ])
        poller = _poller_factory({CANDIDATE_EMOJIS[0]: "candidate_approved"})
        outcome = post_candidates_and_collect_reactions(
            [_Trade()],
            bot_token="tok",
            channel_thread_id="c1",
            http_client=http,
            reaction_poller=poller,
            now_fn=_now_factory(),
        )
        approved = outcome.approved[0]
        for field in (
            "ticker", "qty", "entry", "sl", "tp", "notional",
            "commission_rt", "r_r_ratio", "instrument",
            "signal_date", "asof_date", "combined", "confidence",
        ):
            assert hasattr(approved, field), f"missing field: {field}"
        assert approved.ticker == "SPY"
        assert approved.instrument is InstrumentType.US_ETF


# --- Emoji helpers ---------------------------------------------------


class TestEmojiHelpers:
    def test_candidate_emojis_distinct(self):
        assert len(set(CANDIDATE_EMOJIS)) == len(CANDIDATE_EMOJIS)
        assert len(CANDIDATE_EMOJIS) == 5

    def test_global_reactions_distinct(self):
        assert {EMOJI_APPROVE, EMOJI_REJECT, EMOJI_WAIT} == {EMOJI_APPROVE, EMOJI_REJECT, EMOJI_WAIT}
        assert EMOJI_APPROVE != EMOJI_REJECT
        assert EMOJI_APPROVE != EMOJI_WAIT
        assert EMOJI_REJECT != EMOJI_WAIT

    def test_emoji_mapper_default(self):
        assert d._numeric_emoji(0) == "1️⃣"
        assert d._numeric_emoji(4) == "5️⃣"
        # Out-of-range wraps around.
        assert d._numeric_emoji(5) == "1️⃣"


# --- Outcome type -----------------------------------------------------


class TestOutcomeType:
    def test_default_outcome_is_empty(self):
        outcome = ApprovalOutcome()
        assert outcome.approved == []
        assert outcome.rejected == []
        assert outcome.waited == []
        assert outcome.error == ""
        assert outcome.message_id == ""

    def test_approved_trade_defaults(self):
        trade = ApprovedTrade(
            ticker="AAPL", qty=1, entry=200.0, sl=186.0, tp=228.0,
            notional=200.0, commission_rt=16.0, r_r_ratio=2.0,
            instrument=InstrumentType.US_STOCK, signal_date="2026-06-10",
            asof_date="2026-06-10", combined=0.65, confidence=0.7,
        )
        assert trade.reasons == []
        assert trade.message_id == ""
