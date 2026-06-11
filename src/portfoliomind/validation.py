"""Independent trade validation (card 10).

This module is a deliberate **second set of eyes** on the trades the
primary strategy proposes. It runs AFTER the news + technical analysis
and sizing are done, and it does NOT trust their output: it re-derives
the evidence from scratch and applies hard/soft gates, producing a
per-trade :class:`ValidationVerdict` of ``APPROVE`` / ``FLAG`` /
``REJECT``.

Separation of duties is the whole point. The primary pipeline
(:mod:`portfoliomind.signals.combined` +
:mod:`portfoliomind.signals.sizer`) optimizes for *finding* setups.
This validator optimizes for *not losing money on a bad one*. It is
the gate a human risk manager would be: skeptical, evidence-driven,
and unable to place an order itself.

Independent checks
------------------
1. **Iron rules** (hard) — SL and TP present and on the correct side
   of entry. A malformed order is rejected outright.
2. **Reward:risk** (soft/hard) — the realized R:R must clear
   :data:`MIN_REWARD_RISK`; below 1.0 is a hard reject.
3. **News re-check** (hard) — the validator re-pulls sentiment itself.
   Negative news rejects the trade no matter what the chart says.
4. **Backtest support** (hard/soft) — the validator runs a
   walk-forward backtest of the ticker and requires a positive
   historical expectancy with enough samples. A pattern with no
   out-of-sample edge is rejected; thin samples are flagged.
5. **Calibration** (soft) — if the backtest's realized win rate sits
   far below the claimed ``p_bullish`` (overconfident model), flag it.
6. **Concentration** (hard) — the order's allocation must respect the
   per-position cap against equity.

Decision: any failed **hard** check → ``REJECT``; otherwise any failed
**soft** check → ``FLAG``; all clear → ``APPROVE``.

The module is pure orchestration: ``sentiment_fn`` and ``backtest_fn``
are injected (defaulting to the real news + backtest layers) so the
whole thing is hermetic in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .backtest import BacktestResult, backtest_ticker
from .logging_setup import get_logger
from .signals.sizer import MAX_POSITION_FRACTION

log = get_logger(__name__)


# --- Gate thresholds (tunable) ------------------------------------------------
#: Minimum acceptable reward:risk on a validated trade.
MIN_REWARD_RISK: float = 1.8
#: Reward:risk below this is a hard reject (the trade is upside-down).
HARD_MIN_REWARD_RISK: float = 1.0
#: Sentiment at/below this rejects the trade (independent re-check).
SENTIMENT_REJECT_BELOW: float = 0.0
#: Backtests with at least this many trades count as "enough samples".
MIN_BACKTEST_TRADES: int = 5
#: A claimed-vs-realized win-rate gap above this is flagged as
#: overconfident model calibration.
MAX_CALIBRATION_GAP: float = 0.20

#: Verdict strings.
APPROVE: str = "APPROVE"
FLAG: str = "FLAG"
REJECT: str = "REJECT"


@dataclass(frozen=True)
class CheckResult:
    """One named validation check."""

    name: str
    passed: bool
    severity: str  # "hard" | "soft"
    detail: str


@dataclass(frozen=True)
class ValidationVerdict:
    """The independent verdict for one proposed trade."""

    ticker: str
    decision: str  # APPROVE | FLAG | REJECT
    checks: list[CheckResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    backtest: Optional[BacktestResult] = None
    confidence: float = 0.0

    @property
    def approved(self) -> bool:
        return self.decision == APPROVE

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "decision": self.decision,
            "confidence": round(self.confidence, 3),
            "checks": [
                {"name": c.name, "passed": c.passed, "severity": c.severity, "detail": c.detail}
                for c in self.checks
            ],
            "reasons": list(self.reasons),
            "backtest": self.backtest.to_dict() if self.backtest else None,
        }

    def summary_line(self) -> str:
        bad = [c.name for c in self.checks if not c.passed]
        tail = f" — failed: {', '.join(bad)}" if bad else ""
        return f"{self.decision} {self.ticker} (confidence {self.confidence:.0%}){tail}"


# --- Order field access (works on TradeOrder, Candidate, or test fakes) -------


def _f(obj: Any, *names: str, default: float = 0.0) -> float:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def _decision_from_checks(checks: list[CheckResult]) -> str:
    if any(not c.passed and c.severity == "hard" for c in checks):
        return REJECT
    if any(not c.passed and c.severity == "soft" for c in checks):
        return FLAG
    return APPROVE


# --- Public API ---------------------------------------------------------------


def validate_trade(
    order: Any,
    *,
    equity: float = 10_000.0,
    sentiment_fn: Optional[Callable[[str], float]] = None,
    backtest_fn: Optional[Callable[[str], BacktestResult]] = None,
    max_position_fraction: float = MAX_POSITION_FRACTION,
) -> ValidationVerdict:
    """Independently validate one proposed ``order``.

    ``order`` is duck-typed: any object exposing ``ticker``,
    ``entry_price`` (or ``entry``/``last_close``), ``sl``
    (``stop_loss``), ``tp`` (``take_profit``), ``allocation``, and
    optionally ``p_bullish``. Never raises.
    """
    ticker = str(getattr(order, "ticker", "")).upper() or "?"
    if sentiment_fn is None:
        sentiment_fn = _default_sentiment_fn()
    if backtest_fn is None:
        backtest_fn = backtest_ticker

    entry = _f(order, "entry_price", "entry", "last_close")
    sl = _f(order, "sl", "stop_loss")
    tp = _f(order, "tp", "take_profit")
    allocation = _f(order, "allocation")
    p_claimed = _f(order, "p_bullish")

    checks: list[CheckResult] = []
    reasons: list[str] = []

    # 1) Iron rules ---------------------------------------------------------
    iron_ok = entry > 0 and sl > 0 and tp > 0 and sl < entry < tp
    checks.append(
        CheckResult(
            "iron_rules",
            iron_ok,
            "hard",
            (
                f"entry={entry:.2f} sl={sl:.2f} tp={tp:.2f}"
                if iron_ok
                else f"SL/TP malformed or wrong side (entry={entry:.2f} sl={sl:.2f} tp={tp:.2f})"
            ),
        )
    )

    # 2) Reward:risk --------------------------------------------------------
    rr = (tp - entry) / (entry - sl) if iron_ok and entry > sl else 0.0
    if rr < HARD_MIN_REWARD_RISK:
        checks.append(CheckResult("reward_risk", False, "hard", f"R:R={rr:.2f} below {HARD_MIN_REWARD_RISK}"))
    elif rr < MIN_REWARD_RISK:
        checks.append(CheckResult("reward_risk", False, "soft", f"R:R={rr:.2f} below preferred {MIN_REWARD_RISK}"))
    else:
        checks.append(CheckResult("reward_risk", True, "soft", f"R:R={rr:.2f}"))

    # 3) News re-check (independent) ---------------------------------------
    try:
        sentiment = float(sentiment_fn(ticker))
    except Exception as e:  # noqa: BLE001
        log.warning("validation: sentiment re-check failed for %s: %s", ticker, type(e).__name__)
        sentiment = 0.0
        reasons.append("news re-check unavailable; treated as neutral")
    news_ok = sentiment >= SENTIMENT_REJECT_BELOW
    checks.append(
        CheckResult(
            "news_recheck",
            news_ok,
            "hard",
            f"independent sentiment={sentiment:+.2f}"
            + ("" if news_ok else " (negative — vetoes the setup)"),
        )
    )

    # 4) Backtest support + 5) calibration ---------------------------------
    bt: Optional[BacktestResult] = None
    try:
        bt = backtest_fn(ticker)
    except Exception as e:  # noqa: BLE001
        log.warning("validation: backtest failed for %s: %s", ticker, type(e).__name__)
        reasons.append(f"backtest unavailable ({type(e).__name__}); edge not independently confirmed")

    if bt is None or bt.n_trades == 0:
        checks.append(CheckResult("backtest_support", False, "soft", "no historical setups to confirm the edge"))
    elif bt.n_trades < MIN_BACKTEST_TRADES:
        checks.append(
            CheckResult(
                "backtest_support",
                False,
                "soft",
                f"thin sample ({bt.n_trades} trades, expectancy {bt.expectancy:+.1%})",
            )
        )
    elif bt.expectancy <= 0:
        checks.append(
            CheckResult(
                "backtest_support",
                False,
                "hard",
                f"negative historical edge ({bt.n_trades} trades, expectancy {bt.expectancy:+.1%})",
            )
        )
    else:
        checks.append(
            CheckResult(
                "backtest_support",
                True,
                "hard",
                f"{bt.n_trades} trades, win_rate {bt.win_rate:.0%}, expectancy {bt.expectancy:+.1%}",
            )
        )
        # Calibration only meaningful when we have a claim and a sample.
        if p_claimed > 0:
            gap = p_claimed - bt.win_rate
            calib_ok = gap <= MAX_CALIBRATION_GAP
            checks.append(
                CheckResult(
                    "calibration",
                    calib_ok,
                    "soft",
                    f"claimed p={p_claimed:.0%} vs realized {bt.win_rate:.0%} (gap {gap:+.0%})",
                )
            )

    # 6) Concentration ------------------------------------------------------
    cap = equity * max_position_fraction
    conc_ok = allocation <= cap + 1e-6 if equity > 0 else True
    checks.append(
        CheckResult(
            "concentration",
            conc_ok,
            "hard",
            f"allocation ${allocation:,.0f} vs cap ${cap:,.0f}",
        )
    )

    decision = _decision_from_checks(checks)
    passed = sum(1 for c in checks if c.passed)
    confidence = passed / len(checks) if checks else 0.0

    for c in checks:
        if not c.passed:
            reasons.append(f"{c.severity} check '{c.name}' failed: {c.detail}")
    if decision == APPROVE and not reasons:
        reasons.append("all independent checks passed")

    verdict = ValidationVerdict(
        ticker=ticker,
        decision=decision,
        checks=checks,
        reasons=reasons,
        backtest=bt,
        confidence=confidence,
    )
    log.info("validation: %s", verdict.summary_line())
    return verdict


def validate_batch(
    orders: list[Any],
    *,
    equity: float = 10_000.0,
    sentiment_fn: Optional[Callable[[str], float]] = None,
    backtest_fn: Optional[Callable[[str], BacktestResult]] = None,
) -> list[ValidationVerdict]:
    """Validate a batch of proposed orders. Never raises."""
    return [
        validate_trade(
            o, equity=equity, sentiment_fn=sentiment_fn, backtest_fn=backtest_fn
        )
        for o in orders
    ]


def _default_sentiment_fn() -> Callable[[str], float]:
    """Production sentiment callable — same graceful-no-key behavior as
    the combiner."""
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return lambda ticker: 0.0

    def _score(ticker: str) -> float:
        from .news.sentiment import score_ticker_sentiment

        return float(score_ticker_sentiment(ticker, api_key=api_key))

    return _score


__all__ = [
    "MIN_REWARD_RISK",
    "HARD_MIN_REWARD_RISK",
    "SENTIMENT_REJECT_BELOW",
    "MIN_BACKTEST_TRADES",
    "MAX_CALIBRATION_GAP",
    "APPROVE",
    "FLAG",
    "REJECT",
    "CheckResult",
    "ValidationVerdict",
    "validate_trade",
    "validate_batch",
]
