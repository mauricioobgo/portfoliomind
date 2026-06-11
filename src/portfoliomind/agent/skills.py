"""The agent's skill registry — the tools the LLM can call.

Each :class:`AgentSkill` pairs an OpenAI-function-calling-compatible
schema with a handler that wires into the existing PortfolioMind
modules (Sheets client, InvestingPro/XTB Playwright logins, signals,
sizer, approval). The agent loop in ``scripts/run_agent.py`` exposes
:func:`to_openai_tools` to the model and dispatches calls through
:func:`invoke_skill`.

Handlers lazy-import their dependencies so importing this module is
cheap and hermetic (no Playwright/Google/OpenAI at import time), and
every handler returns a JSON-serializable dict — the LLM only ever
sees structured results, never raw exceptions or secrets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..logging_setup import get_logger
from ..time_utils import iso_now

log = get_logger(__name__)


@dataclass(frozen=True)
class AgentSkill:
    """One callable skill: schema for the LLM + handler for the runtime."""

    name: str
    description: str
    parameters: dict
    handler: Callable[..., dict]


# --- Handlers -------------------------------------------------------------------
# Every handler returns {"status": "ok"|"error", ...} and never raises:
# the agent loop feeds the dict straight back to the model.


def _safe(fn: Callable[..., dict]) -> Callable[..., dict]:
    def wrapper(**kwargs: Any) -> dict:
        try:
            return fn(**kwargs)
        except Exception as e:  # noqa: BLE001 — the model gets a structured error, not a traceback
            log.warning("skill %s failed: %s", fn.__name__, type(e).__name__)
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    wrapper.__name__ = fn.__name__
    return wrapper


def _sheets_and_id() -> tuple[Any, str]:
    from ..config import PortfoliomindConfig
    from ..sheets.client import SheetsClient

    config = PortfoliomindConfig.from_env()
    return SheetsClient.from_config(config), config.google_sheet_id


@_safe
def connect_google_sheets() -> dict:
    """Authenticate the Google service account and verify sheet access."""
    sheets, sheet_id = _sheets_and_id()
    if not sheet_id:
        return {"status": "error", "error": "GOOGLE_SHEET_ID is blank — bootstrap the sheet first"}
    tabs = [w.get("title", "") for w in sheets.list_worksheets(sheet_id)]
    return {"status": "ok", "sheet_id": sheet_id, "tabs": tabs}


@_safe
def login_investingpro() -> dict:
    """Log in to InvestingPro via the persistent Playwright session."""
    from ..config import PortfoliomindConfig
    from ..investingpro.login import login

    config = PortfoliomindConfig.from_env()
    result = login(config)
    return {"status": "ok", "logged_in": True, "detail": str(getattr(result, "final_url", ""))}


@_safe
def login_xtb() -> dict:
    """Log in to XTB xStation 5 via the persistent Playwright session."""
    from ..config import PortfoliomindConfig
    from ..xtb.login import XTBSessionPaths, build_context, ensure_logged_in, teardown_context

    config = PortfoliomindConfig.from_env()
    paths = XTBSessionPaths.from_config(config)
    context = build_context(paths)
    try:
        page = context.pages[0] if context.pages else context.new_page()
        ensure_logged_in(page, config, failure_screenshot_dir=paths.login_screenshots_dir)
    finally:
        teardown_context(context)
    return {"status": "ok", "logged_in": True}


@_safe
def read_suggestions() -> dict:
    """Read the operator's standing mandate from the Suggestions tab."""
    from ..approval import read_suggestions as _read

    suggestions = _read()
    return {
        "status": "ok",
        "count": len(suggestions),
        "suggestions": [
            {
                "ticker": s.ticker,
                "action": s.action,
                "max_allocation": s.max_allocation,
                "conviction": s.conviction,
                "status": s.status,
                "active_buy": s.is_active_buy(),
            }
            for s in suggestions
        ],
    }


@_safe
def scan_bullish_patterns(ticker: str) -> dict:
    """Detect bullish chart patterns + posterior P(upside) for one ticker."""
    from ..signals.patterns import detect_bullish_patterns
    from ..signals.technicals import fetch_ohlcv

    closes = fetch_ohlcv(ticker)
    result = detect_bullish_patterns(ticker, closes=closes)
    return {"status": "ok", **result.to_dict()}


@_safe
def analyze_news(ticker: str) -> dict:
    """Score recent news sentiment for one ticker (LLM over RSS headlines)."""
    import os

    from ..news.sentiment import score_ticker_sentiment

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"status": "error", "error": "OPENAI_API_KEY not set"}
    score = score_ticker_sentiment(ticker, api_key=api_key)
    return {"status": "ok", "ticker": ticker.upper(), "sentiment": score}


@_safe
def score_universe(top_n: int = 5) -> dict:
    """Probabilistic bullish scan of the universe — top-N qualified candidates."""
    from ..signals.combined import score_universe as _score

    candidates = _score(top_n=int(top_n))
    return {"status": "ok", "candidates": [c.to_dict() for c in candidates]}


@_safe
def propose_trades(top_n: int = 5) -> dict:
    """Score, size (fractional-Kelly), match against the mandate, persist approved."""
    from ..approval import persist_approved_trades, post_candidates_and_collect_reactions
    from ..signals.combined import score_universe as _score
    from ..signals.sizer import PositionSizer, SizingError

    candidates = _score(top_n=int(top_n))
    sizer = PositionSizer()
    sized = []
    skipped: list[str] = []
    for c in candidates:
        try:
            sized.append(sizer.size(c))
        except SizingError as e:
            skipped.append(str(e))
    outcome = post_candidates_and_collect_reactions(sized)
    persisted = persist_approved_trades(outcome.approved)
    return {
        "status": "ok",
        "scored": len(candidates),
        "sized": len(sized),
        "sizing_skipped": skipped,
        "approved": len(outcome.approved),
        "rejected": len(outcome.rejected),
        "persisted_rows": persisted,
        "decisions": outcome.notes,
    }


@_safe
def execute_approved_trades() -> dict:
    """Run the XTB executor over Approved Trades (dry-run unless live-gated)."""
    from datetime import datetime

    from ..config import PortfoliomindConfig
    from ..scheduler.jobs import MorningContext
    from ..sheets.client import SheetsClient
    from ..sheets.schema import AGENT_LOG
    from ..time_utils import BOGOTA_TZ
    from ..xtb.runner import run_morning

    config = PortfoliomindConfig.from_env()
    sheets = SheetsClient.from_config(config)
    sheet_id = config.google_sheet_id

    def log_to_sheet(level: str, message: str) -> None:
        sheets.append_rows(sheet_id, AGENT_LOG, [[iso_now(), level, "agent", message]])

    ctx = MorningContext(
        config=config,
        sheets=sheets,
        sheet_id=sheet_id,
        today=datetime.now(BOGOTA_TZ),
        log_to_sheet=log_to_sheet,
    )
    result = run_morning(ctx)
    return {
        "status": "ok",
        "dry_run": config.xtb_dry_run,
        "orders_placed": result.orders_placed,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "error": result.error,
    }


@_safe
def log_action(level: str, message: str) -> dict:
    """Append one audit row to the Agent Log tab."""
    from ..sheets.schema import AGENT_LOG

    sheets, sheet_id = _sheets_and_id()
    sheets.append_rows(sheet_id, AGENT_LOG, [[iso_now(), level.upper(), "agent", message]])
    return {"status": "ok"}


# --- Registry -------------------------------------------------------------------

_NO_PARAMS: dict = {"type": "object", "properties": {}, "required": []}
_TICKER_PARAM: dict = {
    "type": "object",
    "properties": {"ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}},
    "required": ["ticker"],
}
_TOP_N_PARAM: dict = {
    "type": "object",
    "properties": {
        "top_n": {
            "type": "integer",
            "description": "How many top candidates to return (default 5)",
        }
    },
    "required": [],
}

SKILLS: dict[str, AgentSkill] = {
    s.name: s
    for s in (
        AgentSkill(
            name="connect_google_sheets",
            description=(
                "Log in to Google with the service account and verify access to "
                "the portfolio Google Sheet. Returns the sheet ID and tab list."
            ),
            parameters=_NO_PARAMS,
            handler=connect_google_sheets,
        ),
        AgentSkill(
            name="login_investingpro",
            description="Log in to Investing.com / InvestingPro with the configured account.",
            parameters=_NO_PARAMS,
            handler=login_investingpro,
        ),
        AgentSkill(
            name="login_xtb",
            description="Log in to XTB xStation 5 with the configured account.",
            parameters=_NO_PARAMS,
            handler=login_xtb,
        ),
        AgentSkill(
            name="read_suggestions",
            description=(
                "Read the operator's standing investment mandate from the "
                "Suggestions tab. Only tickers with an ACTIVE BUY row may be bought."
            ),
            parameters=_NO_PARAMS,
            handler=read_suggestions,
        ),
        AgentSkill(
            name="scan_bullish_patterns",
            description=(
                "Detect bullish chart patterns (golden cross, breakout, RSI "
                "recovery, MACD cross, higher lows, pullback bounce, uptrend "
                "stack) for one ticker and return the posterior probability of "
                "upside p_bullish."
            ),
            parameters=_TICKER_PARAM,
            handler=scan_bullish_patterns,
        ),
        AgentSkill(
            name="analyze_news",
            description="Score recent news sentiment in [-1, +1] for one ticker.",
            parameters=_TICKER_PARAM,
            handler=analyze_news,
        ),
        AgentSkill(
            name="score_universe",
            description=(
                "Probabilistic bullish scan of the whole universe. Returns the "
                "top-N candidates that pass the bullish-tech, pattern, "
                "positive-news, and strength gates."
            ),
            parameters=_TOP_N_PARAM,
            handler=score_universe,
        ),
        AgentSkill(
            name="propose_trades",
            description=(
                "Full pipeline: score the universe, size the top candidates with "
                "fractional-Kelly, match them against the Suggestions mandate, "
                "and persist the approved orders to Approved Trades."
            ),
            parameters=_TOP_N_PARAM,
            handler=propose_trades,
        ),
        AgentSkill(
            name="execute_approved_trades",
            description=(
                "Hand the Approved Trades batch to the XTB executor. DRY-RUN "
                "unless the operator enabled the two-toggle live gate."
            ),
            parameters=_NO_PARAMS,
            handler=execute_approved_trades,
        ),
        AgentSkill(
            name="log_action",
            description="Append one audit entry (level + message) to the Agent Log tab.",
            parameters={
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "message": {"type": "string"},
                },
                "required": ["level", "message"],
            },
            handler=log_action,
        ),
    )
}


def get_skill(name: str) -> AgentSkill:
    """Look up a skill by name. Raises ``KeyError`` for unknown names."""
    return SKILLS[name]


def to_openai_tools() -> list[dict]:
    """Render the registry as OpenAI Chat Completions ``tools``."""
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            },
        }
        for s in SKILLS.values()
    ]


def invoke_skill(name: str, arguments: str | dict | None = None) -> dict:
    """Dispatch one tool call. ``arguments`` may be the raw JSON string
    from the model or an already-parsed dict. Never raises."""
    if name not in SKILLS:
        return {"status": "error", "error": f"unknown skill {name!r}"}
    if arguments is None:
        kwargs: dict = {}
    elif isinstance(arguments, str):
        try:
            kwargs = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"bad arguments JSON: {e.msg}"}
    else:
        kwargs = dict(arguments)
    return SKILLS[name].handler(**kwargs)


__all__ = [
    "AgentSkill",
    "SKILLS",
    "get_skill",
    "to_openai_tools",
    "invoke_skill",
]
