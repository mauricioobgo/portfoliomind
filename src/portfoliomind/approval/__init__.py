"""Operator-approval flow for the card 7 sizer (Discord + Sheets).

Public surface — the names and signatures the card 8 strategy runner
imports (see :func:`portfoliomind.strategy_runner._try_import_approval`):

* :func:`post_candidates_and_collect_reactions` — post a list of
  :class:`~portfoliomind.signals.sizer.TradeOrder` to the operator's
  Discord home thread, wait for ``✅`` / ``❌`` / ``⏸`` reactions,
  return an :class:`ApprovalOutcome`.
* :func:`persist_approved_trades` — append the approved
  :class:`TradeOrder` rows to the ``APPROVED_TRADES`` Google Sheet tab,
  with dedup against the existing rows.

Both functions **never raise**. A failure is converted into an
:class:`ApprovalOutcome` (with ``error`` set) or a
:class:`PersistResult` (with ``error`` set) so the morning job
continues. The exception is :class:`RuntimeError` from clear
programming errors (bad arguments); those are caught by the strategy
runner and logged.

The package is split across three modules:

* :mod:`portfoliomind.approval.discord` — the HTTP/websocket client for
  Discord. Uses ``requests`` for the REST call (post the message) and
  ``websockets`` (already a transitive dep via Discord's gateway
  protocol in :mod:`portfoliomind.discord` adapters — but for our
  minimal use we hit the REST API for everything). The
  ``fake_*`` seams let the tests inject canned responses without a
  real network.
* :mod:`portfoliomind.approval.persist` — the Sheets write step. Reads
  ``APPROVED_TRADES``, filters out already-present
  ``(Ticker, signal_date, Entry Price)`` triples, appends the rest.
  Dedup is keyed on the triple, not just the ticker, so a re-run with
  a different ``entry_price`` is treated as a new trade.
* :mod:`portfoliomind.approval.__init__` (this file) — the
  re-exports that make the package import surface match the card 8
  contract.

The card 7 spec mandates that the Discord message format is
operator-readable:

    SYMBOL  combined=+X.XX confidence=X.XX  qty=N entry=$P.PP
    SL=$S.SS TP=$T.TT  reason

with one line per candidate, in a single message. The Discord client
posts this message, listens for reactions, and stops at the first
non-ambiguous reaction per candidate (✅ → approved, ❌ → rejected,
⏸ → wait, anything else → ignore).
"""

from __future__ import annotations

from .discord import (
    ApprovalOutcome,
    ApprovedTrade,
    RejectedTrade,
    WaitedTrade,
    DiscordApprovalError,
    build_candidates_message,
    format_trade_order_line,
    post_candidates_and_collect_reactions,
)
from .persist import (
    PersistResult,
    persist_approved_trades,
)

__all__ = [
    # Functions (the card-8 contract)
    "post_candidates_and_collect_reactions",
    "persist_approved_trades",
    # Outcome types
    "ApprovalOutcome",
    "ApprovedTrade",
    "RejectedTrade",
    "WaitedTrade",
    "PersistResult",
    "DiscordApprovalError",
    # Formatters (exposed for tests and CLI scripts)
    "build_candidates_message",
    "format_trade_order_line",
]
