"""Hermetic tests for the independent validation agent registry + prompt."""

from __future__ import annotations

import json

from portfoliomind.agent import (
    VALIDATOR_SKILLS,
    VALIDATOR_SYSTEM_PROMPT,
    invoke_validator_skill,
    validator_tools,
)


EXPECTED = {
    "read_proposed_trades",
    "backtest_ticker",
    "recheck_news",
    "validate_trade",
    "record_validation",
}


# --- Registry shape -----------------------------------------------------------


def test_registry_has_expected_skills():
    assert set(VALIDATOR_SKILLS) == EXPECTED


def test_validator_has_no_execution_skill():
    """Separation of duties: the validator must not be able to place orders."""
    forbidden = {"execute_approved_trades", "propose_trades", "place_order", "login_xtb"}
    assert not (set(VALIDATOR_SKILLS) & forbidden)


def test_tool_schema_is_serializable():
    tools = validator_tools()
    assert len(tools) == len(VALIDATOR_SKILLS)
    for t in tools:
        assert t["type"] == "function"
        assert t["function"]["name"] in VALIDATOR_SKILLS
        json.dumps(t)


# --- Dispatch -----------------------------------------------------------------


def test_unknown_skill_is_structured_error():
    result = invoke_validator_skill("execute_approved_trades")
    assert result["status"] == "error"
    assert "unknown validator skill" in result["error"]


def test_bad_json_is_structured_error():
    result = invoke_validator_skill("backtest_ticker", "{nope")
    assert result["status"] == "error"


def test_validate_trade_skill_dispatch(monkeypatch):
    """validate_trade dispatched through the registry, with the backtest
    and news layers stubbed."""
    from portfoliomind import validation
    from portfoliomind.backtest import BacktestResult

    good = BacktestResult(
        ticker="AAPL", n_trades=20, n_wins=12, win_rate=0.60,
        avg_return=0.012, expectancy=0.012, avg_p_bullish=0.62,
    )
    monkeypatch.setattr(validation, "backtest_ticker", lambda t: good)
    monkeypatch.setattr(validation, "_default_sentiment_fn", lambda: (lambda t: 0.3))

    result = invoke_validator_skill(
        "validate_trade",
        json.dumps(
            {"ticker": "AAPL", "entry_price": 100.0, "sl": 97.0, "tp": 106.0,
             "allocation": 900.0, "p_bullish": 0.62, "equity": 10000.0}
        ),
    )
    assert result["status"] == "ok"
    assert result["decision"] == "APPROVE"


def test_backtest_ticker_skill_dispatch(monkeypatch):
    import portfoliomind.backtest as bt_pkg
    from portfoliomind.backtest import BacktestResult

    res = BacktestResult(ticker="NVDA", n_trades=10, n_wins=6, win_rate=0.6,
                         avg_return=0.01, expectancy=0.01, avg_p_bullish=0.6)
    # The skill does `from ..backtest import backtest_ticker` at call time,
    # so patching the package attribute is what the import resolves to.
    monkeypatch.setattr(bt_pkg, "backtest_ticker", lambda t: res)

    out = invoke_validator_skill("backtest_ticker", {"ticker": "NVDA"})
    assert out["status"] == "ok"
    assert out["ticker"] == "NVDA"
    assert out["n_trades"] == 10


def test_read_proposed_trades_dispatch(monkeypatch):
    from portfoliomind.agent import skills as skills_mod
    from portfoliomind.sheets.schema import APPROVED_TRADES

    class FakeSheets:
        def read_range(self, sheet_id, tab, a1):
            assert tab == APPROVED_TRADES
            return [["2026-06-11", "AAPL", "Stock", "bullish-patterns", "MEDIUM",
                     "900", "9", "100", "97", "106", "note"]]

    monkeypatch.setattr(skills_mod, "_sheets_and_id", lambda: (FakeSheets(), "sid"))
    out = invoke_validator_skill("read_proposed_trades")
    assert out["status"] == "ok"
    assert out["count"] == 1
    assert out["proposed_trades"][0]["ticker"] == "AAPL"


def test_record_validation_writes_disqualified_on_reject(monkeypatch):
    from portfoliomind.agent import skills as skills_mod
    from portfoliomind.sheets.schema import AGENT_LOG, DISQUALIFIED

    appended: dict[str, list] = {}

    class FakeSheets:
        def append_rows(self, sheet_id, tab, values):
            appended.setdefault(tab, []).extend(values)
            return len(appended[tab])

    monkeypatch.setattr(skills_mod, "_sheets_and_id", lambda: (FakeSheets(), "sid"))
    out = invoke_validator_skill(
        "record_validation", {"ticker": "AAPL", "decision": "REJECT", "detail": "negative edge"}
    )
    assert out["status"] == "ok"
    assert AGENT_LOG in appended
    assert DISQUALIFIED in appended
    assert appended[DISQUALIFIED][0][1] == "AAPL"


def test_record_validation_approve_skips_disqualified(monkeypatch):
    from portfoliomind.agent import skills as skills_mod
    from portfoliomind.sheets.schema import AGENT_LOG, DISQUALIFIED

    appended: dict[str, list] = {}

    class FakeSheets:
        def append_rows(self, sheet_id, tab, values):
            appended.setdefault(tab, []).extend(values)
            return len(appended[tab])

    monkeypatch.setattr(skills_mod, "_sheets_and_id", lambda: (FakeSheets(), "sid"))
    invoke_validator_skill("record_validation", {"ticker": "AAPL", "decision": "APPROVE"})
    assert AGENT_LOG in appended
    assert DISQUALIFIED not in appended


# --- Prompt -------------------------------------------------------------------


def test_prompt_establishes_independence():
    for needle in ("INDEPENDENT", "separate agent", "skeptical", "backtest"):
        assert needle in VALIDATOR_SYSTEM_PROMPT


def test_prompt_forbids_execution():
    assert "CANNOT execute" in VALIDATOR_SYSTEM_PROMPT
    assert "user makes the final call" in VALIDATOR_SYSTEM_PROMPT


def test_prompt_defines_verdicts():
    for needle in ("APPROVE", "FLAG", "REJECT"):
        assert needle in VALIDATOR_SYSTEM_PROMPT
