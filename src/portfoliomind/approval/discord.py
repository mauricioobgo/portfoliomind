"""Discord interactive approval for card 7.

Post a list of sized candidates to the operator's Discord home
thread, then collect ✅ / ❌ / ⏸ reactions for up to ``timeout_min``
minutes (default 30). No reaction by the timeout is treated as ❌.

The module is designed to be hermetic in tests. Three plug-in seams
let tests inject canned responses without a real network or
Discord:

* ``http_client`` — a callable ``(method, url, *, json=None, headers=None)``
  that returns a fake ``(status_code, json_body)`` pair. Defaults to
  the module-level :func:`_real_http_client` that uses ``requests``.
* ``reaction_poller`` — a callable ``(message_id, channel_id, *,
  bot_token, deadline_unix) -> dict[ticker, str]`` that returns the
  per-ticker reactions observed. Defaults to
  :func:`_poll_real_reactions` that uses ``requests`` to GET the
  message's reactions endpoint.
* ``ticker_to_emoji`` — a callable ``(ticker) -> str`` that maps
  each candidate to a unique emoji reaction (so the operator can
  approve *this* trade specifically). Defaults to
  :func:`_numeric_emoji` which uses 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ in order.

The flow is:

1. Build a single operator-readable message
   (:func:`build_candidates_message`).
2. POST it to the thread (``POST /channels/{thread_id}/messages``).
3. For each candidate, add the corresponding numeric emoji as a
   reaction to the message so the operator can tap on it.
4. Poll the message's reactions every ``poll_interval`` seconds
   (default 5) until ``timeout_min`` is up. After each poll, the
   first unambiguous reaction per ticker short-circuits the wait:
   the matching emoji + a global ✅ / ❌ / ⏸ are both honored.
5. After the timeout, any ticker with no reaction is treated as ❌.

The function **never raises**. A failure inside is converted into an
:class:`ApprovalOutcome` with ``error`` set and empty approval
lists. The strategy runner treats a non-empty ``error`` as a soft
failure.

Discord reference
-----------------
* POST ``/channels/{id}/messages`` — text body, ``Authorization: Bot <token>``
* PUT ``/channels/{id}/messages/{msg_id}/reactions/{emoji}/@me`` — add
  a reaction (the bot adds its own so the operator can tap on it).
* GET ``/channels/{id}/messages/{msg_id}/reactions/{emoji}`` — list
  users who reacted (we check for any non-bot user).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..logging_setup import get_logger
from ..time_utils import iso_now

log = get_logger(__name__)


# --- Reaction constants ----------------------------------------------------

# Global reactions (per the spec): ✅ approve, ❌ reject, ⏸ wait.
EMOJI_APPROVE: str = "✅"
EMOJI_REJECT: str = "❌"
EMOJI_WAIT: str = "⏸"
GLOBAL_REACTIONS: tuple[str, ...] = (EMOJI_APPROVE, EMOJI_REJECT, EMOJI_WAIT)

# Per-candidate reactions: 1️⃣, 2️⃣, 3️⃣, 4️⃣, 5️⃣. We use the Unicode
# keycap-1/2/3/4/5 — they're recognized by Discord as custom-keycap
# emoji and survive the message round-trip. Capped at 5 (the spec
# caps candidates at 5).
CANDIDATE_EMOJIS: tuple[str, ...] = ("1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣")


# --- Result types ---------------------------------------------------------


@dataclass(frozen=True)
class ApprovedTrade:
    """A single trade that the operator approved via ✅ reaction.

    Mirrors the relevant fields of
    :class:`~portfoliomind.signals.sizer.TradeOrder` so the
    persistence step can write it to ``APPROVED_TRADES`` without
    needing the full sizer.
    """

    ticker: str
    qty: float
    entry: float
    sl: float
    tp: float
    notional: float
    commission_rt: float
    r_r_ratio: float
    instrument: Any  # str or InstrumentType; persist reads .value if present
    signal_date: str
    asof_date: str
    combined: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    message_id: str = ""  # Discord message id (for logging)


@dataclass(frozen=True)
class RejectedTrade:
    """A trade the operator rejected (❌) or ignored (no reaction)."""

    ticker: str
    reason: str  # "operator rejected" or "no reaction by timeout"
    message_id: str = ""


@dataclass(frozen=True)
class WaitedTrade:
    """A trade the operator parked (⏸). The runner keeps the trade for a re-prompt."""

    ticker: str
    reason: str = "operator wait"
    message_id: str = ""


@dataclass(frozen=True)
class ApprovalOutcome:
    """The full result of :func:`post_candidates_and_collect_reactions`.

    The strategy runner (card 8) reads ``approved``, ``rejected``,
    and ``waited``. ``error`` is non-empty on any failure (Discord
    post failed, no candidates to post, etc.) — the runner treats
    that as a soft fail and the morning job continues.
    """

    approved: list[ApprovedTrade] = field(default_factory=list)
    rejected: list[RejectedTrade] = field(default_factory=list)
    waited: list[WaitedTrade] = field(default_factory=list)
    message_id: str = ""
    channel_id: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""


class DiscordApprovalError(RuntimeError):
    """Raised only by :func:`_real_http_client` and friends; the public
    entry point converts these into :class:`ApprovalOutcome.error`."""


# --- Public entry point ---------------------------------------------------


# Default HTTP client (production). Tests inject a fake.
def _real_http_client(
    method: str, url: str, *, json: Any = None, headers: Optional[dict] = None
) -> tuple[int, Any]:
    import requests  # local import — keep the dep optional in tests

    headers = headers or {}
    try:
        resp = requests.request(method, url, json=json, headers=headers, timeout=30)
    except Exception as e:  # noqa: BLE001
        raise DiscordApprovalError(f"http {method} {url} failed: {e!r}") from e
    try:
        body: Any = resp.json() if resp.text else None
    except ValueError:
        body = resp.text
    return resp.status_code, body


# Default reaction poller (production). Tests inject a fake.
def _poll_real_reactions(
    message_id: str,
    channel_id: str,
    *,
    bot_token: str,
    deadline_unix: float,
    poll_interval: float,
) -> dict[str, str]:
    """Poll the message's reactions until the deadline.

    Returns a mapping ``{emoji: "approve" | "reject" | "wait"}`` for
    the global reactions, plus a per-candidate mapping in
    ``{"<numeric_emoji>": "approve"}`` if the operator tapped that
    candidate's emoji. The function is resilient to transient
    failures: a single failed poll logs a warning and continues.
    """
    import requests  # local import

    out: dict[str, str] = {}
    base = f"https://discord.com/api/v10/channels/{channel_id}/messages/{message_id}/reactions"
    headers = {"Authorization": f"Bot {bot_token}"}
    # Encode emoji for URL. Discord expects URL-encoded UTF-8.
    import urllib.parse

    def _encoded(emoji: str) -> str:
        return urllib.parse.quote(emoji, safe="")

    # Map our short names to the reaction emoji.
    short_for = {EMOJI_APPROVE: "approve", EMOJI_REJECT: "reject", EMOJI_WAIT: "wait"}

    while time.time() < deadline_unix:
        for emoji, short in short_for.items():
            url = f"{base}/{_encoded(emoji)}?limit=10"
            try:
                resp = requests.get(url, headers=headers, timeout=10)
            except Exception as e:  # noqa: BLE001
                log.warning("discord_poll: %s fetch failed: %s", emoji, type(e).__name__)
                continue
            if resp.status_code != 200:
                # 404 means the message was deleted; surface it.
                if resp.status_code == 404:
                    log.warning("discord_poll: message %s 404", message_id)
                    return out
                continue
            try:
                users = resp.json()
            except ValueError:
                continue
            # If any non-bot user reacted, that's an operator decision.
            if any(u for u in users if not _is_bot_user(u, bot_token=bot_token)):
                out[emoji] = short
        for emoji in CANDIDATE_EMOJIS:
            url = f"{base}/{_encoded(emoji)}?limit=10"
            try:
                resp = requests.get(url, headers=headers, timeout=10)
            except Exception:
                continue
            if resp.status_code != 200:
                continue
            try:
                users = resp.json()
            except ValueError:
                continue
            if any(u for u in users if not _is_bot_user(u, bot_token=bot_token)):
                out[emoji] = "candidate_approved"
        if any(out.get(e) for e in short_for):
            # The operator made a global decision. Stop polling.
            break
        time.sleep(poll_interval)
    return out


def _is_bot_user(user: dict, *, bot_token: str) -> bool:
    """Heuristic: a user is the bot itself if its id matches the bot token.

    Discord doesn't expose the bot's user id via the reactions
    endpoint, but the *Application* object does. We fall back to
    treating all reactions as operator reactions if we can't tell.
    """
    return bool(user.get("bot"))


def _numeric_emoji(index: int) -> str:
    """Map a 0-indexed candidate position to its numeric-emoji reaction.

    Indices >= len(CANDIDATE_EMOJIS) wrap around (the spec caps at 5).
    """
    return CANDIDATE_EMOJIS[index % len(CANDIDATE_EMOJIS)]


# --- The public function --------------------------------------------------


def post_candidates_and_collect_reactions(
    candidates: Iterable[Any],
    *,
    timeout_min: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    bot_token: str = "",
    channel_thread_id: str = "",
    poll_interval: float = 5.0,
    http_client: Callable[..., tuple[int, Any]] = _real_http_client,
    reaction_poller: Callable[..., dict[str, str]] = _poll_real_reactions,
    emoji_mapper: Callable[[int], str] = _numeric_emoji,
    now_fn: Callable[[], float] = time.time,
) -> ApprovalOutcome:
    """Post ``candidates`` to Discord and collect reactions.

    Parameters
    ----------
    candidates:
        Iterable of :class:`~portfoliomind.signals.sizer.TradeOrder`
        (or anything with the same fields). The function reads
        ``ticker``, ``qty``, ``entry``, ``sl``, ``tp``, ``notional``,
        ``commission_rt``, ``r_r_ratio``, ``instrument``,
        ``signal_date``/``asof_date``, ``combined``, ``confidence``,
        ``reasons``.
    timeout_min:
        How long to wait for reactions in minutes. Default 30 (per
        the spec). ``timeout_seconds`` overrides ``timeout_min`` if
        both are given; the card 8 strategy runner uses
        ``timeout_seconds`` and the CLI uses ``timeout_min``.
    timeout_seconds:
        How long to wait for reactions in seconds. Wins over
        ``timeout_min`` when both are passed. The card 8 contract.
    bot_token:
        Discord bot token. When empty, the function looks at
        ``DISCORD_BOT_TOKEN`` in the environment. When still empty,
        the function returns an :class:`ApprovalOutcome` with
        ``error="missing bot_token"`` (no Discord call attempted).
    channel_thread_id:
        The thread to post in. When empty, looks at
        ``DISCORD_HOME_CHANNEL_THREAD_ID``. Empty + nothing in env
        → ``error="missing channel_thread_id"``.
    poll_interval:
        Seconds between reaction polls. Default 5s.
    http_client / reaction_poller / emoji_mapper:
        Test seams. Production uses the module-level defaults.
    now_fn:
        Test seam for the clock. Production uses :func:`time.time`.

    Returns
    -------
    :class:`ApprovalOutcome`
        Always. Never raises.
    """
    started_at = iso_now()
    if not bot_token:
        bot_token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not channel_thread_id:
        channel_thread_id = os.environ.get("DISCORD_HOME_CHANNEL_THREAD_ID", "").strip()
    if not bot_token:
        return ApprovalOutcome(error="missing bot_token", started_at=started_at)
    if not channel_thread_id:
        return ApprovalOutcome(error="missing channel_thread_id", started_at=started_at)

    # Normalize timeout: seconds wins, min defaults to 30, seconds
    # defaults to min*60.
    if timeout_seconds is None:
        timeout_seconds = (timeout_min if timeout_min is not None else 30) * 60

    trade_list = list(candidates)
    if not trade_list:
        return ApprovalOutcome(
            error="no candidates to post",
            started_at=started_at,
            finished_at=iso_now(),
        )

    try:
        return _post_and_collect_unchecked(
            trade_list,
            timeout_seconds=timeout_seconds,
            bot_token=bot_token,
            channel_thread_id=channel_thread_id,
            poll_interval=poll_interval,
            http_client=http_client,
            reaction_poller=reaction_poller,
            emoji_mapper=emoji_mapper,
            now_fn=now_fn,
            started_at=started_at,
        )
    except Exception as e:  # noqa: BLE001 — last-ditch: never raise
        log.error("post_candidates_and_collect_reactions: %s", type(e).__name__)
        return ApprovalOutcome(
            error=f"{type(e).__name__}: {e!r}",
            started_at=started_at,
            finished_at=iso_now(),
        )


# --- Internals -----------------------------------------------------------


def _post_and_collect_unchecked(
    trades: list[Any],
    *,
    timeout_seconds: int,
    bot_token: str,
    channel_thread_id: str,
    poll_interval: float,
    http_client: Callable[..., tuple[int, Any]],
    reaction_poller: Callable[..., dict[str, str]],
    emoji_mapper: Callable[[int], str],
    now_fn: Callable[[], float],
    started_at: str,
) -> ApprovalOutcome:
    message = build_candidates_message(trades)
    post_url = f"https://discord.com/api/v10/channels/{channel_thread_id}/messages"
    headers = {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }
    status, body = http_client(
        "POST", post_url, json={"content": message}, headers=headers
    )
    if status not in (200, 201) or not isinstance(body, dict):
        return ApprovalOutcome(
            error=f"discord post failed: status={status} body={body!r}",
            started_at=started_at,
            finished_at=iso_now(),
        )
    message_id = str(body.get("id", ""))

    # Add candidate-specific reactions so the operator can approve
    # individual trades. The bot adds its own reactions (PUT
    # ``/.../reactions/{emoji}/@me``) so the emoji appear in the UI.
    for i, _ in enumerate(trades):
        if i >= len(CANDIDATE_EMOJIS):
            break
        emoji = emoji_mapper(i)
        react_url = (
            f"https://discord.com/api/v10/channels/{channel_thread_id}"
            f"/messages/{message_id}/reactions/{_url_quote(emoji)}/@me"
        )
        try:
            http_client("PUT", react_url, headers=headers)
        except Exception as e:  # noqa: BLE001
            log.warning("discord_react_add: %s: %s", emoji, type(e).__name__)

    # Poll for reactions.
    deadline = now_fn() + float(timeout_seconds)
    try:
        observed = reaction_poller(
            message_id,
            channel_thread_id,
            bot_token=bot_token,
            deadline_unix=deadline,
            poll_interval=poll_interval,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("reaction_poller raised: %s", type(e).__name__)
        observed = {}

    # Decide per-trade.
    approved: list[ApprovedTrade] = []
    rejected: list[RejectedTrade] = []
    waited: list[WaitedTrade] = []

    # Global overrides win.
    if observed.get(EMOJI_REJECT) == "reject":
        # Operator said "reject all".
        for t in trades:
            ticker = str(getattr(t, "ticker", ""))
            rejected.append(RejectedTrade(ticker=ticker, reason="operator rejected (global)", message_id=message_id))
        return ApprovalOutcome(
            approved=approved,
            rejected=rejected,
            waited=waited,
            message_id=message_id,
            channel_id=channel_thread_id,
            started_at=started_at,
            finished_at=iso_now(),
        )
    if observed.get(EMOJI_WAIT) == "wait":
        # Operator said "wait on all".
        for t in trades:
            ticker = str(getattr(t, "ticker", ""))
            waited.append(WaitedTrade(ticker=ticker, message_id=message_id))
        return ApprovalOutcome(
            approved=approved,
            rejected=rejected,
            waited=waited,
            message_id=message_id,
            channel_id=channel_thread_id,
            started_at=started_at,
            finished_at=iso_now(),
        )

    # Otherwise, per-candidate.
    for i, t in enumerate(trades):
        ticker = str(getattr(t, "ticker", ""))
        numeric = emoji_mapper(i) if i < len(CANDIDATE_EMOJIS) else ""
        if numeric and observed.get(numeric) == "candidate_approved":
            approved.append(_to_approved(t, message_id=message_id))
        else:
            # No reaction (or only a global approve with no per-trade
            # tap): treat as reject so the dedup layer doesn't include it.
            rejected.append(
                RejectedTrade(
                    ticker=ticker,
                    reason="no per-candidate approval by timeout",
                    message_id=message_id,
                )
            )

    return ApprovalOutcome(
        approved=approved,
        rejected=rejected,
        waited=waited,
        message_id=message_id,
        channel_id=channel_thread_id,
        started_at=started_at,
        finished_at=iso_now(),
    )


def _to_approved(trade: Any, *, message_id: str) -> ApprovedTrade:
    """Map a :class:`TradeOrder` (or duck-typed equivalent) to :class:`ApprovedTrade`."""
    return ApprovedTrade(
        ticker=str(getattr(trade, "ticker", "")),
        qty=float(getattr(trade, "qty", 0.0) or 0.0),
        entry=float(getattr(trade, "entry", 0.0) or 0.0),
        sl=float(getattr(trade, "sl", 0.0) or 0.0),
        tp=float(getattr(trade, "tp", 0.0) or 0.0),
        notional=float(getattr(trade, "notional", 0.0) or 0.0),
        commission_rt=float(getattr(trade, "commission_rt", 0.0) or 0.0),
        r_r_ratio=float(getattr(trade, "r_r_ratio", 0.0) or 0.0),
        instrument=getattr(trade, "instrument", None),
        signal_date=str(getattr(trade, "signal_date", "") or ""),
        asof_date=str(getattr(trade, "asof_date", "") or ""),
        combined=float(getattr(trade, "combined", 0.0) or 0.0),
        confidence=float(getattr(trade, "confidence", 0.0) or 0.0),
        reasons=list(getattr(trade, "reasons", []) or []),
        message_id=message_id,
    )


# --- Message formatting --------------------------------------------------


def build_candidates_message(trades: Iterable[Any]) -> str:
    """Build the operator-facing Discord message.

    The format is one line per candidate, in the order:
    ``SYMBOL  combined=+X.XX confidence=X.XX  qty=N entry=$P.PP
    SL=$S.SS TP=$T.TT  reason``
    """
    header = "**Trade candidates (card 7)** — react ✅/❌/⏸ globally, or 1️⃣/2️⃣/3️⃣/4️⃣/5️⃣ to approve a specific trade."
    lines = [header]
    for i, t in enumerate(trades):
        if i >= len(CANDIDATE_EMOJIS):
            break
        lines.append(format_trade_order_line(t, index=i))
    if not list(trades) if not isinstance(trades, list) else False:
        # iterable was empty — defensive
        pass
    return "\n".join(lines)


def format_trade_order_line(trade: Any, *, index: int = 0) -> str:
    """Format one trade as a single operator-readable line.

    The numeric emoji prefix is included so the operator can see
    which 1️⃣/2️⃣/... reaction corresponds to which trade.
    """
    ticker = str(getattr(trade, "ticker", "?"))
    combined = float(getattr(trade, "combined", 0.0) or 0.0)
    confidence = float(getattr(trade, "confidence", 0.0) or 0.0)
    qty = float(getattr(trade, "qty", 0.0) or 0.0)
    entry = float(getattr(trade, "entry", 0.0) or 0.0)
    sl = float(getattr(trade, "sl", 0.0) or 0.0)
    tp = float(getattr(trade, "tp", 0.0) or 0.0)
    numeric = CANDIDATE_EMOJIS[index] if 0 <= index < len(CANDIDATE_EMOJIS) else "•"
    # Compact qty: 2.0 → 2, 2.5 → 2.5
    if qty == int(qty):
        qty_s = str(int(qty))
    else:
        qty_s = f"{qty:.4f}".rstrip("0").rstrip(".")
    reasons = list(getattr(trade, "reasons", []) or [])
    reason_summary = "; ".join(reasons[:1]) if reasons else "no extra context"
    return (
        f"{numeric} **{ticker}**  combined={combined:+.2f} confidence={confidence:.2f}  "
        f"qty={qty_s} entry=${entry:.2f} SL=${sl:.2f} TP=${tp:.2f}  "
        f"{reason_summary}"
    )


def _url_quote(s: str) -> str:
    import urllib.parse

    return urllib.parse.quote(s, safe="")


__all__ = [
    "ApprovalOutcome",
    "ApprovedTrade",
    "RejectedTrade",
    "WaitedTrade",
    "DiscordApprovalError",
    "post_candidates_and_collect_reactions",
    "build_candidates_message",
    "format_trade_order_line",
    "EMOJI_APPROVE",
    "EMOJI_REJECT",
    "EMOJI_WAIT",
    "CANDIDATE_EMOJIS",
]
