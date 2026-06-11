"""Sheet schema: 12 tabs and their column headers.

These are the single source of truth for tab names and column definitions
across all PortfolioMind cards. The bootstrap and dry-run scripts both
import from here; the upcoming cards (2/3/4) will too.

Column header conventions:
- Match the v4 spec verbatim where the spec gave an explicit list (e.g. the
  Returns Tracker block at the end of the spec).
- For tabs where the spec only described the *purpose* but not exact columns,
  we define a minimum-viable header set that captures every data point the
  spec says the tab must hold. These can be refined in later cards as
  scraping reveals the real shape.

If a column header needs a comma, quote, or newline, use the python string
literal escape — Sheets will treat the cell value as-is.
"""

from __future__ import annotations

from typing import Final

# --- Tab names (verbatim from v4 spec, with their emoji prefixes) -----------

RAW_PICKS: Final[str] = "📥 Raw Picks"
STRATEGY_SELECTION: Final[str] = "🎯 Strategy Selection"
SIGNAL_SCORECARD: Final[str] = "📊 Signal Scorecard"
FORECASTS: Final[str] = "🔮 Forecasts"
DISQUALIFIED: Final[str] = "🚫 Disqualified"
APPROVED_TRADES: Final[str] = "✅ Approved Trades"
EXECUTED_ORDERS: Final[str] = "📈 Executed Orders"
RETURNS_TRACKER: Final[str] = "💰 Returns Tracker"
FORECAST_ACCURACY: Final[str] = "📊 Forecast Accuracy"
MACRO_CONTEXT: Final[str] = "📰 Macro Context"
AGENT_LOG: Final[str] = "🗒️ Agent Log"
SUGGESTIONS: Final[str] = "💡 Suggestions"

TAB_NAMES: Final[tuple[str, ...]] = (
    RAW_PICKS,
    STRATEGY_SELECTION,
    SIGNAL_SCORECARD,
    FORECASTS,
    DISQUALIFIED,
    APPROVED_TRADES,
    EXECUTED_ORDERS,
    RETURNS_TRACKER,
    FORECAST_ACCURACY,
    MACRO_CONTEXT,
    AGENT_LOG,
    SUGGESTIONS,
)

assert len(TAB_NAMES) == 12, f"Expected exactly 12 tabs, got {len(TAB_NAMES)}"

# --- Column headers per tab -------------------------------------------------

TAB_HEADERS: Final[dict[str, list[str]]] = {
    RAW_PICKS: [
        # Per spec lines 316-326: InvestingPro AI pick scrape fields.
        "Ticker",
        "Company Name",
        "Pro Score",
        "Fair Value",
        "Current Price",
        "Upside %",
        "Sector",
        "Recommendation",
        "Scraped At",
    ],
    STRATEGY_SELECTION: [
        # Spec lines 660-725: the strategy decision tree output.
        "Timestamp",
        "Market Regime",
        "VIX",
        "Rate Direction",
        "SPY Trend",
        "Earnings Season",
        "Short Strategy",
        "Medium Strategy",
        "Long Strategy",
        "Rationale",
    ],
    SIGNAL_SCORECARD: [
        # Spec line 1181 + 360: per-instrument signal breakdown.
        "Timestamp",
        "Ticker",
        "Strategy",
        "Timeframe",
        "Technical Score",
        "Fundamental Score",
        "Macro Score",
        "Sentiment Score",
        "Composite Score",
        "Threshold",
        "Pass / Fail",
    ],
    FORECASTS: [
        # Spec line 1469: full FM forecast block per instrument.
        "Timestamp",
        "Ticker",
        "Strategy",
        "Timeframe",
        "Primary Model",
        "Bear PT",
        "Base PT",
        "Bull PT",
        "Confidence",
        "Entry Price",
        "SL",
        "TP",
        "R/R",
    ],
    DISQUALIFIED: [
        # Spec line 607: removed instruments + reason.
        "Timestamp",
        "Ticker",
        "Strategy",
        "Disqualification Reason",
        "Rule Reference",
    ],
    APPROVED_TRADES: [
        # Spec line 626: user-confirmed trades with allocation + SL + TP.
        "Timestamp",
        "Ticker",
        "Type",
        "Strategy",
        "Timeframe",
        "Allocation ($)",
        "Qty",
        "Entry Price",
        "SL",
        "TP",
        "Approval Note",
    ],
    EXECUTED_ORDERS: [
        # Spec line 616-624: order IDs + prices + timestamps + status.
        "Timestamp",
        "Ticker",
        "Order ID",
        "Side",
        "Qty",
        "Entry Price",
        "SL",
        "TP",
        "Status",
        "Screenshot",
    ],
    # Verbatim from spec lines 1558-1562 (the only fully-spec'd tab).
    RETURNS_TRACKER: [
        "Ticker",
        "Type (Stock/ETF)",
        "Strategy",
        "Timeframe",
        "Entry Date",
        "Entry Price",
        "Current Price",
        "Qty",
        "Entry Value",
        "Current Value",
        "Unrealized P&L ($)",
        "Unrealized P&L (%)",
        "Days Held",
        "SL",
        "TP",
        "Dividend Received ($)",
        "Total Return",
        "vs SPY",
        "Status",
    ],
    FORECAST_ACCURACY: [
        # Spec lines 1506-1521.
        "Ticker",
        "Strategy",
        "Timeframe",
        "Primary Model",
        "Entry Price",
        "Bear PT",
        "Base PT",
        "Bull PT",
        "Exit Price",
        "Actual Return %",
        "Model Error %",
        "Within 1σ Band?",
        "Holding Days",
        "Exit Reason",
    ],
    MACRO_CONTEXT: [
        # Spec lines 525-536 + 599-600: VIX, SPY, Fed rate, sector RS, regime.
        "Timestamp",
        "VIX",
        "SPY Price",
        "SPY vs SMA200",
        "Regime",
        "Fed Rate Direction",
        "Top 3 Sectors (1M RS)",
        "Earnings Season Window",
        "Notes",
    ],
    AGENT_LOG: [
        # Spec line 626: full audit trail.
        "Timestamp",
        "Level",
        "Module",
        "Message",
    ],
    SUGGESTIONS: [
        # The operator's standing investment mandate. The approval layer
        # auto-approves a sized order only when its ticker has a row here
        # with Action=BUY and Status=ACTIVE, clamped to the allocation cap.
        "Timestamp",
        "Ticker",
        "Action",
        "Max Allocation ($)",
        "Conviction",
        "Source",
        "Notes",
        "Status",
    ],
}

# Sanity: every tab name has a header list of the same length key.
for _name in TAB_NAMES:
    assert _name in TAB_HEADERS, f"Missing header list for tab: {_name!r}"
    assert len(TAB_HEADERS[_name]) > 0, f"Empty header list for tab: {_name!r}"


__all__ = [
    "TAB_NAMES",
    "TAB_HEADERS",
    "RAW_PICKS",
    "STRATEGY_SELECTION",
    "SIGNAL_SCORECARD",
    "FORECASTS",
    "DISQUALIFIED",
    "APPROVED_TRADES",
    "EXECUTED_ORDERS",
    "RETURNS_TRACKER",
    "FORECAST_ACCURACY",
    "MACRO_CONTEXT",
    "AGENT_LOG",
    "SUGGESTIONS",
]
