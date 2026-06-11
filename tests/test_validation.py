"""Hermetic tests for :mod:`portfoliomind.validation`.

``sentiment_fn`` and ``backtest_fn`` are always injected — no news
layer, no yfinance, no network.
"""

from __future__ import annotations

from dataclasses import dataclass


from portfoliomind.backtest import BacktestResult
from portfoliomind.validation import (
    APPROVE,
    FLAG,
    REJECT,
    ValidationVerdict,
    validate_batch,
    validate_trade,
)


@dataclass
class StubOrder:
    ticker: str = "AAPL"
    entry_price: float = 100.0
    sl: float = 97.0
    tp: float = 106.0
    allocation: float = 900.0
    p_bullish: float = 0.62


def good_backtest(ticker: str = "AAPL") -> BacktestResult:
    """A backtest with a healthy, well-calibrated edge."""
    return BacktestResult(
        ticker=ticker,
        n_trades=20,
        n_wins=12,
        win_rate=0.60,
        avg_return=0.012,
        total_return=0.30,
        expectancy=0.012,
        avg_p_bullish=0.62,
        calibration_gap=0.02,
    )


def losing_backtest(ticker: str = "AAPL") -> BacktestResult:
    return BacktestResult(
        ticker=ticker,
        n_trades=20,
        n_wins=5,
        win_rate=0.25,
        avg_return=-0.01,
        total_return=-0.20,
        expectancy=-0.01,
        avg_p_bullish=0.62,
        calibration_gap=0.37,
    )


def thin_backtest(ticker: str = "AAPL") -> BacktestResult:
    return BacktestResult(
        ticker=ticker,
        n_trades=2,
        n_wins=1,
        win_rate=0.50,
        avg_return=0.005,
        expectancy=0.005,
        avg_p_bullish=0.62,
    )


# --- Happy path ---------------------------------------------------------------


def test_clean_trade_is_approved():
    v = validate_trade(
        StubOrder(), equity=10_000.0, sentiment_fn=lambda t: 0.3, backtest_fn=good_backtest
    )
    assert isinstance(v, ValidationVerdict)
    assert v.decision == APPROVE
    assert v.approved
    assert v.confidence == 1.0
    assert v.backtest is not None
    assert all(c.passed for c in v.checks)


# --- Hard rejects -------------------------------------------------------------


def test_negative_news_rejects():
    v = validate_trade(
        StubOrder(), sentiment_fn=lambda t: -0.4, backtest_fn=good_backtest
    )
    assert v.decision == REJECT
    assert any(c.name == "news_recheck" and not c.passed for c in v.checks)


def test_negative_historical_edge_rejects():
    v = validate_trade(
        StubOrder(), sentiment_fn=lambda t: 0.3, backtest_fn=losing_backtest
    )
    assert v.decision == REJECT
    assert any(c.name == "backtest_support" and not c.passed and c.severity == "hard" for c in v.checks)


def test_broken_iron_rules_rejects():
    bad = StubOrder(sl=105.0)  # SL above entry — wrong side
    v = validate_trade(bad, sentiment_fn=lambda t: 0.3, backtest_fn=good_backtest)
    assert v.decision == REJECT
    assert any(c.name == "iron_rules" and not c.passed for c in v.checks)


def test_inverted_reward_risk_rejects():
    # tp barely above entry, sl far below → R:R < 1
    bad = StubOrder(entry_price=100.0, sl=80.0, tp=101.0)
    v = validate_trade(bad, sentiment_fn=lambda t: 0.3, backtest_fn=good_backtest)
    assert v.decision == REJECT
    assert any(c.name == "reward_risk" and not c.passed and c.severity == "hard" for c in v.checks)


def test_over_concentration_rejects():
    over = StubOrder(allocation=5000.0)  # 50% of 10k, cap is 10%
    v = validate_trade(over, equity=10_000.0, sentiment_fn=lambda t: 0.3, backtest_fn=good_backtest)
    assert v.decision == REJECT
    assert any(c.name == "concentration" and not c.passed for c in v.checks)


# --- Soft flags ---------------------------------------------------------------


def test_thin_backtest_sample_flags():
    v = validate_trade(StubOrder(), sentiment_fn=lambda t: 0.3, backtest_fn=thin_backtest)
    assert v.decision == FLAG
    assert any(c.name == "backtest_support" and not c.passed and c.severity == "soft" for c in v.checks)


def test_overconfident_model_flags_calibration():
    overconfident = BacktestResult(
        ticker="AAPL",
        n_trades=20,
        n_wins=8,
        win_rate=0.40,
        avg_return=0.006,
        expectancy=0.006,
        avg_p_bullish=0.62,
        calibration_gap=0.22,
    )
    v = validate_trade(
        StubOrder(p_bullish=0.62), sentiment_fn=lambda t: 0.3, backtest_fn=lambda t: overconfident
    )
    assert v.decision == FLAG
    assert any(c.name == "calibration" and not c.passed for c in v.checks)


def test_weak_reward_risk_flags():
    # R:R between hard floor (1.0) and preferred (1.8)
    weak = StubOrder(entry_price=100.0, sl=97.0, tp=104.0)  # R:R = 4/3 ≈ 1.33
    v = validate_trade(weak, sentiment_fn=lambda t: 0.3, backtest_fn=good_backtest)
    assert v.decision == FLAG
    assert any(c.name == "reward_risk" and not c.passed and c.severity == "soft" for c in v.checks)


# --- Robustness ---------------------------------------------------------------


def test_sentiment_failure_is_neutral_not_crash():
    def bad_sentiment(t):
        raise RuntimeError("LLM down")

    v = validate_trade(StubOrder(), sentiment_fn=bad_sentiment, backtest_fn=good_backtest)
    # Neutral (0.0) passes the >= 0 news gate.
    assert v.decision in (APPROVE, FLAG)
    assert any("news re-check unavailable" in r for r in v.reasons)


def test_backtest_failure_does_not_crash():
    def bad_backtest(t):
        raise RuntimeError("yfinance down")

    v = validate_trade(StubOrder(), sentiment_fn=lambda t: 0.3, backtest_fn=bad_backtest)
    assert isinstance(v, ValidationVerdict)
    # No backtest → soft fail on support → at most a FLAG, never a crash.
    assert v.decision in (FLAG, REJECT)


def test_verdict_to_dict_is_serializable():
    import json

    v = validate_trade(StubOrder(), sentiment_fn=lambda t: 0.3, backtest_fn=good_backtest)
    json.dumps(v.to_dict())


def test_validate_batch():
    orders = [StubOrder(ticker="AAA"), StubOrder(ticker="BBB", sl=105.0)]
    verdicts = validate_batch(orders, sentiment_fn=lambda t: 0.3, backtest_fn=good_backtest)
    assert len(verdicts) == 2
    assert verdicts[0].decision == APPROVE
    assert verdicts[1].decision == REJECT


def test_summary_line_lists_failures():
    v = validate_trade(StubOrder(sl=105.0), sentiment_fn=lambda t: 0.3, backtest_fn=good_backtest)
    line = v.summary_line()
    assert "REJECT" in line
    assert "iron_rules" in line
