"""The independent validation agent (card 10).

This is a *second*, deliberately separate agent from the primary
investing agent in :mod:`portfoliomind.agent.skills`. The primary
agent finds and proposes trades (news + technical + pattern analysis,
sizing, persistence). This agent runs **after** that work is done and
independently double-checks each proposed trade before it ever reaches
execution — then hands its verdict to the user for the final call.

Separation of duties is enforced structurally:

* The validator's skill registry contains **no execution skill**. It
  can read proposed trades, run backtests, re-check news, validate,
  and record — but it cannot place an order. Only the user (or the
  primary pipeline, post-confirmation) can.
* Its checks come from :mod:`portfoliomind.validation`, which
  re-derives the evidence from scratch rather than trusting the
  primary pipeline's scores.

The agent loop (``scripts/run_validator.py``) ends by presenting a
per-trade ``APPROVE`` / ``FLAG`` / ``REJECT`` report to the user and
asking for the go/no-go. Rejected trades are recorded to the
``🚫 Disqualified`` tab; every verdict is written to ``🗒️ Agent Log``.
"""

from __future__ import annotations

import json
from typing import Any

from ..logging_setup import get_logger
from ..time_utils import iso_now
from . import skills as _skills
from .skills import AgentSkill, _safe, analyze_news

log = get_logger(__name__)


VALIDATOR_MODEL: str = "gpt-4o"

VALIDATOR_SYSTEM_PROMPT: str = """\
You are PortfolioMind-Validator, an INDEPENDENT risk reviewer. You are a
separate agent from the primary investing agent. The primary agent has
already done the news and technical analysis and proposed sized trades; your
job is to validate those proposals BEFORE any money moves, and to bring a
clear recommendation to the user for the final decision.

# Your stance
You are skeptical by default and you trust evidence, not the primary agent's
optimism. You re-derive everything yourself: you re-check the news, you run a
historical backtest to confirm the pattern actually has an out-of-sample
edge, and you check reward:risk and position concentration. A pretty score
from the primary agent means nothing to you until the backtest and the news
agree with it.

# Workflow
1. `read_proposed_trades` — pull the trades the primary agent persisted to
   the Approved Trades tab.
2. For each proposed trade:
   - `backtest_ticker` — confirm the bullish pattern has a positive
     historical expectancy and check calibration (claimed probability vs.
     realized win rate).
   - `recheck_news` — independently re-score the ticker's news. Negative
     news vetoes the trade.
   - `validate_trade` — run the full independent gate (iron rules, R:R,
     news, backtest support, calibration, concentration) and get an
     APPROVE / FLAG / REJECT verdict.
3. `record_validation` — write each verdict to the audit log; REJECTs also
   go to the Disqualified tab.
4. Present the user a concise per-trade summary: the verdict, the one or two
   decisive reasons, and the backtest's win rate / expectancy. Group by
   APPROVE / FLAG / REJECT.

# Hard rules
- You CANNOT execute trades. You have no execution skill. Do not claim to
  have placed anything.
- REJECT any trade that fails a hard check (broken SL/TP, negative news,
  negative historical edge, over the concentration cap).
- FLAG (don't silently approve) trades with thin backtest samples, weak
  reward:risk, or an overconfident probability vs. the backtest.
- End by explicitly asking the user which trades they want to proceed with.
  The user makes the final call — you advise.
- Never echo credentials or secrets. Record every verdict to the audit log.
"""


# --- Validator-specific handlers ----------------------------------------------


@_safe
def read_proposed_trades() -> dict:
    """Read the trades the primary agent persisted to Approved Trades."""
    from ..sheets.schema import APPROVED_TRADES

    sheets, sheet_id = _skills._sheets_and_id()
    rows = sheets.read_range(sheet_id, APPROVED_TRADES, "A2:K")
    # Columns: Timestamp, Ticker, Type, Strategy, Timeframe, Allocation ($),
    #          Qty, Entry Price, SL, TP, Approval Note
    trades = []
    for r in rows or []:
        padded = list(r) + [""] * (11 - len(r))
        trades.append(
            {
                "timestamp": padded[0],
                "ticker": padded[1],
                "allocation": padded[5],
                "qty": padded[6],
                "entry_price": padded[7],
                "sl": padded[8],
                "tp": padded[9],
                "note": padded[10],
            }
        )
    return {"status": "ok", "count": len(trades), "proposed_trades": trades}


@_safe
def backtest_ticker_skill(ticker: str) -> dict:
    """Walk-forward backtest one ticker (2y) to confirm the edge."""
    from ..backtest import backtest_ticker as _bt

    result = _bt(ticker)
    return {"status": "ok", **result.to_dict(), "summary": result.summary_line()}


@_safe
def recheck_news(ticker: str) -> dict:
    """Independently re-score the ticker's news sentiment."""
    return analyze_news(ticker=ticker)


def _coerce_order(trade: dict) -> Any:
    """Turn a proposed-trade dict into an order-shaped object for the validator."""

    def _num(v: Any) -> float:
        try:
            return float(str(v).replace("$", "").replace(",", ""))
        except (TypeError, ValueError):
            return 0.0

    class _Order:
        ticker = str(trade.get("ticker", "")).upper()
        entry_price = _num(trade.get("entry_price"))
        sl = _num(trade.get("sl"))
        tp = _num(trade.get("tp"))
        allocation = _num(trade.get("allocation"))
        p_bullish = _num(trade.get("p_bullish"))

    return _Order()


@_safe
def validate_trade_skill(
    ticker: str,
    entry_price: float,
    sl: float,
    tp: float,
    allocation: float = 0.0,
    p_bullish: float = 0.0,
    equity: float = 10_000.0,
) -> dict:
    """Run the full independent validation gate on one trade."""
    from ..validation import validate_trade as _validate

    order = _coerce_order(
        {
            "ticker": ticker,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "allocation": allocation,
            "p_bullish": p_bullish,
        }
    )
    verdict = _validate(order, equity=equity)
    return {"status": "ok", **verdict.to_dict(), "summary": verdict.summary_line()}


@_safe
def record_validation(ticker: str, decision: str, detail: str = "") -> dict:
    """Audit a verdict to Agent Log; REJECTs also land in Disqualified."""
    from ..sheets.schema import AGENT_LOG, DISQUALIFIED

    sheets, sheet_id = _skills._sheets_and_id()
    sheets.append_rows(
        sheet_id, AGENT_LOG, [[iso_now(), "INFO", "validator", f"{decision} {ticker}: {detail}"]]
    )
    if decision.strip().upper() == "REJECT":
        # Disqualified columns: Timestamp, Ticker, Strategy, Reason, Rule Ref
        sheets.append_rows(
            sheet_id,
            DISQUALIFIED,
            [[iso_now(), ticker.upper(), "bullish-patterns", detail, "independent-validation"]],
        )
    return {"status": "ok"}


# --- Registry -----------------------------------------------------------------

_NO_PARAMS: dict = {"type": "object", "properties": {}, "required": []}
_TICKER_PARAM: dict = {
    "type": "object",
    "properties": {"ticker": {"type": "string", "description": "Ticker symbol"}},
    "required": ["ticker"],
}

VALIDATOR_SKILLS: dict[str, AgentSkill] = {
    s.name: s
    for s in (
        AgentSkill(
            name="read_proposed_trades",
            description="Read the trades the primary agent persisted to the Approved Trades tab.",
            parameters=_NO_PARAMS,
            handler=read_proposed_trades,
        ),
        AgentSkill(
            name="backtest_ticker",
            description=(
                "Run a 2-year walk-forward backtest of the bullish-pattern "
                "strategy on one ticker. Returns win rate, expectancy, max "
                "drawdown, and the claimed-vs-realized calibration gap."
            ),
            parameters=_TICKER_PARAM,
            handler=backtest_ticker_skill,
        ),
        AgentSkill(
            name="recheck_news",
            description="Independently re-score the ticker's recent news sentiment in [-1, +1].",
            parameters=_TICKER_PARAM,
            handler=recheck_news,
        ),
        AgentSkill(
            name="validate_trade",
            description=(
                "Run the full independent validation gate on one proposed trade "
                "(iron rules, reward:risk, news re-check, backtest support, "
                "calibration, concentration). Returns APPROVE / FLAG / REJECT."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "entry_price": {"type": "number"},
                    "sl": {"type": "number"},
                    "tp": {"type": "number"},
                    "allocation": {"type": "number"},
                    "p_bullish": {"type": "number"},
                    "equity": {"type": "number"},
                },
                "required": ["ticker", "entry_price", "sl", "tp"],
            },
            handler=validate_trade_skill,
        ),
        AgentSkill(
            name="record_validation",
            description=(
                "Audit a verdict to the Agent Log; a REJECT is also written to "
                "the Disqualified tab."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "decision": {"type": "string", "enum": ["APPROVE", "FLAG", "REJECT"]},
                    "detail": {"type": "string"},
                },
                "required": ["ticker", "decision"],
            },
            handler=record_validation,
        ),
    )
}


def validator_tools() -> list[dict]:
    """Render the validator registry as OpenAI Chat Completions tools."""
    return [
        {
            "type": "function",
            "function": {"name": s.name, "description": s.description, "parameters": s.parameters},
        }
        for s in VALIDATOR_SKILLS.values()
    ]


def invoke_validator_skill(name: str, arguments: str | dict | None = None) -> dict:
    """Dispatch one validator tool call. Never raises."""
    if name not in VALIDATOR_SKILLS:
        return {"status": "error", "error": f"unknown validator skill {name!r}"}
    if arguments is None:
        kwargs: dict = {}
    elif isinstance(arguments, str):
        try:
            kwargs = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"bad arguments JSON: {e.msg}"}
    else:
        kwargs = dict(arguments)
    return VALIDATOR_SKILLS[name].handler(**kwargs)


__all__ = [
    "VALIDATOR_MODEL",
    "VALIDATOR_SYSTEM_PROMPT",
    "VALIDATOR_SKILLS",
    "validator_tools",
    "invoke_validator_skill",
]
