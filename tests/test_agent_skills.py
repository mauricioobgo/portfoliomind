"""Hermetic tests for :mod:`portfoliomind.agent` — prompt + skill registry."""

from __future__ import annotations

import json

import pytest

from portfoliomind.agent import (
    AGENT_SYSTEM_PROMPT,
    SKILLS,
    build_system_prompt,
    get_skill,
    invoke_skill,
    to_openai_tools,
)
from portfoliomind.agent import skills as skills_mod


EXPECTED_SKILLS = {
    "connect_google_sheets",
    "login_investingpro",
    "login_xtb",
    "read_suggestions",
    "scan_bullish_patterns",
    "analyze_news",
    "score_universe",
    "propose_trades",
    "execute_approved_trades",
    "log_action",
}


# --- Registry shape -------------------------------------------------------------


def test_registry_has_expected_skills():
    assert set(SKILLS) == EXPECTED_SKILLS


def test_get_skill_returns_skill():
    s = get_skill("scan_bullish_patterns")
    assert s.name == "scan_bullish_patterns"
    assert callable(s.handler)
    with pytest.raises(KeyError):
        get_skill("not_a_skill")


def test_openai_tool_schema_shape():
    tools = to_openai_tools()
    assert len(tools) == len(SKILLS)
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert fn["name"] in SKILLS
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
        # Every schema must be JSON-serializable (it goes over the wire).
        json.dumps(t)


def test_ticker_skills_require_ticker():
    for name in ("scan_bullish_patterns", "analyze_news"):
        params = get_skill(name).parameters
        assert "ticker" in params["properties"]
        assert "ticker" in params["required"]


# --- Dispatch --------------------------------------------------------------------


def test_invoke_unknown_skill_is_structured_error():
    result = invoke_skill("teleport_money")
    assert result["status"] == "error"
    assert "unknown skill" in result["error"]


def test_invoke_with_bad_json_is_structured_error():
    result = invoke_skill("scan_bullish_patterns", "{not json")
    assert result["status"] == "error"
    assert "bad arguments JSON" in result["error"]


def test_handlers_never_raise():
    """A handler whose dependencies are unavailable (no env config here)
    returns a structured error instead of raising."""
    result = invoke_skill("connect_google_sheets")
    assert result["status"] in ("ok", "error")  # no exception either way


def test_scan_bullish_patterns_dispatch(monkeypatch):
    """End-to-end dispatch through the registry with the network call
    monkeypatched away."""
    from portfoliomind.signals import technicals

    series = [160.0 - i for i in range(60)] + [101.0 + 2.0 * (i + 1) for i in range(20)]
    monkeypatch.setattr(technicals, "fetch_ohlcv", lambda ticker: series)

    result = invoke_skill("scan_bullish_patterns", json.dumps({"ticker": "nvda"}))
    assert result["status"] == "ok"
    assert result["ticker"] == "NVDA"
    assert result["p_bullish"] > 0.5
    assert "golden_cross" in result["patterns"]


def test_log_action_dispatch(monkeypatch):
    appended: list = []

    class FakeSheets:
        def append_rows(self, sheet_id, tab, values):
            appended.extend(values)
            return 1

    monkeypatch.setattr(skills_mod, "_sheets_and_id", lambda: (FakeSheets(), "sid"))
    result = invoke_skill("log_action", {"level": "info", "message": "hello"})
    assert result["status"] == "ok"
    assert appended and appended[0][1] == "INFO" and appended[0][3] == "hello"


# --- Prompt ------------------------------------------------------------------------


def test_prompt_mentions_all_accounts():
    for needle in ("Google", "InvestingPro", "XTB"):
        assert needle in AGENT_SYSTEM_PROMPT


def test_prompt_carries_hard_guardrails():
    for needle in (
        "LONG-ONLY",
        "MANDATE-ONLY",
        "stop-loss",
        "take-profit",
        "TWO-TOGGLE",
        "Suggestions",
        "never echo credentials",
    ):
        assert needle in AGENT_SYSTEM_PROMPT, f"prompt missing guardrail: {needle}"


def test_prompt_embeds_live_risk_parameters():
    """The prompt renders from the sizer/combined constants so the
    prompt and the code can never disagree about the caps."""
    from portfoliomind.signals.combined import MIN_P_BULLISH
    from portfoliomind.signals.sizer import MAX_POSITION_FRACTION

    assert f"{MIN_P_BULLISH:.2f}" in AGENT_SYSTEM_PROMPT
    assert f"{MAX_POSITION_FRACTION:.0%}" in AGENT_SYSTEM_PROMPT
    assert build_system_prompt() == AGENT_SYSTEM_PROMPT


def test_prompt_mentions_probabilistic_and_news():
    assert "posterior probability" in AGENT_SYSTEM_PROMPT
    assert "News sentiment" in AGENT_SYSTEM_PROMPT
