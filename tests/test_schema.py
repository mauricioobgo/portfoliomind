"""Unit tests for :mod:`portfoliomind.sheets.schema`."""

from __future__ import annotations

from portfoliomind.sheets.schema import (
    AGENT_LOG,
    APPROVED_TRADES,
    DISQUALIFIED,
    EXECUTED_ORDERS,
    FORECAST_ACCURACY,
    FORECASTS,
    MACRO_CONTEXT,
    RAW_PICKS,
    RETURNS_TRACKER,
    SIGNAL_SCORECARD,
    STRATEGY_SELECTION,
    SUGGESTIONS,
    TAB_HEADERS,
    TAB_NAMES,
)


def test_exactly_twelve_tabs():
    assert len(TAB_NAMES) == 12


def test_tab_names_contain_required_emoji_prefixes():
    expected_substrings = {
        RAW_PICKS: "Raw Picks",
        STRATEGY_SELECTION: "Strategy Selection",
        SIGNAL_SCORECARD: "Signal Scorecard",
        FORECASTS: "Forecasts",
        DISQUALIFIED: "Disqualified",
        APPROVED_TRADES: "Approved Trades",
        EXECUTED_ORDERS: "Executed Orders",
        RETURNS_TRACKER: "Returns Tracker",
        FORECAST_ACCURACY: "Forecast Accuracy",
        MACRO_CONTEXT: "Macro Context",
        AGENT_LOG: "Agent Log",
        SUGGESTIONS: "Suggestions",
    }
    for tab, sub in expected_substrings.items():
        assert sub in tab, f"tab {tab!r} should contain {sub!r}"


def test_every_tab_has_a_header_list():
    for tab in TAB_NAMES:
        assert tab in TAB_HEADERS, f"missing headers for {tab!r}"
        headers = TAB_HEADERS[tab]
        assert len(headers) > 0, f"empty headers for {tab!r}"


def test_returns_tracker_columns_match_v4_spec():
    """The v4 spec is most explicit about Returns Tracker columns; pin them verbatim."""
    expected = [
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
    ]
    assert TAB_HEADERS[RETURNS_TRACKER] == expected


def test_no_duplicate_columns_within_a_tab():
    for tab, headers in TAB_HEADERS.items():
        assert len(headers) == len(set(headers)), (
            f"tab {tab!r} has duplicate columns: {headers}"
        )


def test_agent_log_has_minimum_columns():
    """AGENT_LOG must have at least: Timestamp, Level, Module, Message."""
    h = TAB_HEADERS[AGENT_LOG]
    for required in ("Timestamp", "Level", "Module", "Message"):
        assert required in h, f"AGENT_LOG missing column: {required}"


def test_returns_tracker_includes_vs_spy():
    """v4 spec section 'Returns Tracker' explicitly requires vs SPY tracking."""
    assert "vs SPY" in TAB_HEADERS[RETURNS_TRACKER]


def test_suggestions_has_mandate_columns():
    """The Suggestions tab is the operator's standing mandate; the approval
    layer depends on these exact columns."""
    h = TAB_HEADERS[SUGGESTIONS]
    for required in ("Timestamp", "Ticker", "Action", "Max Allocation ($)", "Status"):
        assert required in h, f"SUGGESTIONS missing column: {required}"
