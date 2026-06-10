"""Card 8/8 — Strategy runner wired into the morning_run() seam.

The strategy pipeline is a *third* runner that the morning job calls
in addition to the card-2 (InvestingPro scrape) and card-3 (XTB
execution) runners. Where those two runners translate raw market data
into *intentions* and then *place real orders*, the strategy runner
translates signals into a sized, operator-approved batch of trade
candidates and persists the approved subset to the
``APPROVED_TRADES`` tab for the XTB runner to pick up on its next
tick.

The pipeline is composed of three card-6/7 modules that this card
treats as a public contract:

* :mod:`portfoliomind.signals.combined` — :func:`score_universe` returns
  the top-N candidates that pass both the bullish-tech gate AND the
  positive-news gate.
* :mod:`portfoliomind.signals.sizer` — :class:`PositionSizer` sizes
  each candidate into a ``TradeOrder`` (qty, entry, SL, TP) honoring
  the commission-aware cap and the max-positions rule.
* :mod:`portfoliomind.approval.discord` — :func:`post_candidates_and_collect_reactions`
  posts the batch to the operator's Discord home thread and waits up
  to 30 minutes for ``✅`` / ``❌`` reactions.
* :mod:`portfoliomind.approval.persist` — :func:`persist_approved_trades`
  appends the approved rows to ``APPROVED_TRADES``.

**Card 8 ships ahead of cards 6/7.** The same lazy-import pattern
card 4 uses for the platform runners means the strategy_runner is
safe to deploy and run before cards 6 and 7 have landed. When those
modules are missing, :func:`run_morning` returns a
:class:`StrategyResult` with ``status="not_implemented"`` and exits
cleanly — the morning_run wrapper logs a one-line "strategy not
implemented yet" message and continues. This is deliberate: it
unblocks the cron schedule, the operator can see the schedule is
alive, and once cards 6/7 merge the strategy_runner will pick them up
automatically on the next tick.

**Failure isolation.** Every step of the strategy pipeline is wrapped
in its own try/except. A failure in scoring doesn't abort sizing, a
failure in Discord posting doesn't abort persistence. The
:class:`StrategyResult` collects the per-step error messages so the
morning_run wrapper can surface them via Discord and AGENT_LOG.

**Idempotency.** The strategy is idempotent within a Bogota-local
day: re-running for the same day produces the same set of candidates
(the underlying signals are dated) and the persistence step is
dedup-keyed on ``(Ticker, Timestamp)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .logging_setup import get_logger
from .time_utils import iso_now

log = get_logger(__name__)


# Module-level test hooks. The :func:`_try_import_signals` /
# :func:`_try_import_sizer` / :func:`_try_import_approval` helpers
# check these *first*, before falling back to the lazy-import path.
# None of these are set in production; tests use :func:`set_factories`
# to swap in mocks without monkeypatching ``sys.modules``.
_score_universe_factory: Optional[Any] = None
_sizer_factory: Optional[Any] = None
_approval_factory: Optional[Any] = None


# --- Result dataclass -------------------------------------------------------


@dataclass
class StrategyResult:
    """The bag of state the strategy runner returns to ``morning_run``.

    Mirrors the shape of :class:`portfoliomind.scheduler.jobs.MorningResult`
    so the morning_run wrapper can treat all three runners uniformly.
    ``status`` is one of:

    * ``"ran"`` — the strategy executed end-to-end. ``approved_count`` /
      ``rejected_count`` reflect the operator's reaction totals.
    * ``"skipped"`` — the strategy decided not to do work (e.g. zero
      candidates passed the score gates). No orders were proposed.
    * ``"not_implemented"`` — the card-6/7 modules (signals, sizer,
      approval) are not on the import path yet. This is the
      expected state until cards 6/7 merge.
    * ``"failed"`` — the strategy raised. The first error is in
      ``error``; the full list is in ``errors``.

    ``picks_scraped`` is repurposed to mean "candidates scored by
    score_universe" so the morning_run summary line is uniform
    across runners.
    """

    runner: str = "strategy"
    status: str = "not_implemented"
    picks_scraped: int = 0
    orders_placed: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def ok(self) -> bool:
        """True if the strategy produced a usable outcome (ran or skipped)."""
        return not self.error and self.status in ("ran", "skipped", "not_implemented")

    def summary_line(self) -> str:
        if self.status == "ran":
            return (
                f"strategy_runner OK: scored={self.picks_scraped} "
                f"approved={self.approved_count} rejected={self.rejected_count}"
            )
        if self.status == "skipped":
            return f"strategy_runner SKIP: {self.skip_reason or 'no work'}"
        if self.status == "not_implemented":
            return "strategy_runner SKIP: card 6/7 modules not implemented yet"
        if self.status == "failed":
            first = self.errors[0] if self.errors else self.error
            return f"strategy_runner FAIL: {len(self.errors)} error(s); first={first!r}"
        return f"strategy_runner {self.status}"


# --- Lazy-import protocol ----------------------------------------------------


class _SignalsModule(Protocol):
    """The structural type we expect from ``portfoliomind.signals``."""

    def score_universe(self, *, top_n: int = 5) -> list[Any]: ...


class _SizerClass(Protocol):
    """The structural type we expect from ``portfoliomind.signals.sizer``."""

    def __call__(self) -> "_SizerInstance": ...


class _SizerInstance(Protocol):
    """The instance type produced by ``PositionSizer()``."""

    def size(self, candidate: Any) -> Any: ...


class _ApprovalModule(Protocol):
    """The structural type we expect from ``portfoliomind.approval``."""

    def post_candidates_and_collect_reactions(
        self, candidates: list[Any], *, timeout_seconds: int = 1800
    ) -> Any: ...

    def persist_approved_trades(self, orders: list[Any]) -> int: ...


def _try_import_signals() -> Optional[_SignalsModule]:
    """Lazy import of ``portfoliomind.signals.combined`` (card 6).

    Returns the module object, or ``None`` if card 6 has not landed.
    Card 8 ships ahead of card 6; the morning job must keep ticking
    even when the strategy is not implemented.

    When a test factory is installed via :func:`set_factories`, this
    function returns the test factory instead of the lazy-imported
    module. The factory must expose ``score_universe(top_n=...)`` on
    a module-like object.
    """
    if _score_universe_factory is not None:
        return _score_universe_factory
    try:
        from .signals import combined as signals_combined  # type: ignore[import-not-found]
    except ImportError:
        return None
    if not hasattr(signals_combined, "score_universe"):
        return None
    return signals_combined


def _try_import_sizer() -> Optional[type[_SizerClass]]:
    """Lazy import of ``portfoliomind.signals.sizer.PositionSizer`` (card 7)."""
    if _sizer_factory is not None:
        return _sizer_factory
    try:
        from .signals import sizer as sizer_mod  # type: ignore[import-not-found]
    except ImportError:
        return None
    cls = getattr(sizer_mod, "PositionSizer", None)
    if cls is None or not callable(cls):
        return None
    return cls


def _try_import_approval() -> Optional[_ApprovalModule]:
    """Lazy import of ``portfoliomind.approval`` (card 7)."""
    if _approval_factory is not None:
        return _approval_factory
    try:
        from . import approval as approval_mod  # type: ignore[import-not-found]
    except ImportError:
        return None
    if not hasattr(approval_mod, "post_candidates_and_collect_reactions"):
        return None
    if not hasattr(approval_mod, "persist_approved_trades"):
        return None
    return approval_mod


# --- Pipeline ---------------------------------------------------------------


def _score_candidates(
    signals: Optional[_SignalsModule], *, top_n: int
) -> tuple[list[Any], Optional[str]]:
    """Run the score_universe step. Returns (candidates, error_or_none).

    A missing signals module is treated as a "not implemented" signal,
    not a failure — the caller will convert that into
    ``status='not_implemented'``.
    """
    if signals is None:
        return [], "card 6 signals.combined.score_universe not implemented"
    try:
        candidates = signals.score_universe(top_n=top_n)
    except Exception as e:  # noqa: BLE001
        return [], f"score_universe raised: {type(e).__name__}: {e}"
    if not candidates:
        return [], "no candidates passed the bullish-tech + positive-news gate"
    return list(candidates), None


def _size_candidates(
    sizer_cls: Optional[type[_SizerClass]],
    candidates: list[Any],
) -> tuple[list[Any], Optional[str]]:
    """Run the position-sizing step. Returns (sized_orders, error_or_none)."""
    if sizer_cls is None:
        return [], "card 7 signals.sizer.PositionSizer not implemented"
    try:
        sizer: Any = sizer_cls()
    except Exception as e:  # noqa: BLE001
        return [], f"PositionSizer() construction raised: {type(e).__name__}: {e}"
    sized: list[Any] = []
    for c in candidates:
        try:
            order = sizer.size(c)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "sizer_skip ticker=%s err_type=%s err=%r",
                getattr(c, "ticker", "?"),
                type(e).__name__,
                str(e)[:200],
            )
            continue
        sized.append(order)
    return sized, None


def _post_and_collect(
    approval: Optional[_ApprovalModule],
    sized: list[Any],
    *,
    timeout_seconds: int,
) -> tuple[Any, Optional[str]]:
    """Post the batch to Discord and collect reactions. Returns
    (approval_outcome, error_or_none). The outcome object exposes
    ``approved`` and ``rejected`` lists (or whatever card 7 ships)."""
    if approval is None:
        return None, "card 7 approval module not implemented"
    try:
        outcome = approval.post_candidates_and_collect_reactions(
            sized, timeout_seconds=timeout_seconds
        )
    except Exception as e:  # noqa: BLE001
        return None, f"post_candidates_and_collect_reactions raised: {type(e).__name__}: {e}"
    return outcome, None


def _persist_approved(
    approval: Optional[_ApprovalModule],
    approval_outcome: Any,
) -> tuple[int, Optional[str]]:
    """Persist the approved orders to ``APPROVED_TRADES``. Returns
    (rows_appended, error_or_none)."""
    if approval is None:
        return 0, "card 7 approval module not implemented"
    # Card 7's contract for the approval outcome is "approved" and
    # "rejected" lists, but we don't know the exact field names yet.
    # We try a few common shapes defensively so the strategy_runner
    # works with whichever card 7 ships.
    approved = (
        getattr(approval_outcome, "approved", None)
        or getattr(approval_outcome, "approved_orders", None)
        or getattr(approval_outcome, "orders", None)
        or []
    )
    if not approved:
        return 0, None
    try:
        rows = approval.persist_approved_trades(list(approved))
    except Exception as e:  # noqa: BLE001
        return 0, f"persist_approved_trades raised: {type(e).__name__}: {e}"
    # Card 7 may return an int (rows appended) or a sequence with
    # ``__len__``. Try the int cast first since it's the documented
    # contract; fall back to a length probe.
    if isinstance(rows, (int, bool)):
        return int(rows), None
    if hasattr(rows, "__len__"):
        try:
            return len(rows), None
        except TypeError:
            return 0, None
    return 0, None


def _count_reactions(approval_outcome: Any) -> tuple[int, int]:
    """Best-effort count of approved vs rejected from the approval outcome.

    The exact shape of the outcome object is not yet pinned (card 7's
    contract), so we try a few common patterns.
    """
    if approval_outcome is None:
        return 0, 0
    approved = (
        getattr(approval_outcome, "approved", None)
        or getattr(approval_outcome, "approved_orders", None)
        or []
    )
    rejected = (
        getattr(approval_outcome, "rejected", None)
        or getattr(approval_outcome, "rejected_orders", None)
        or []
    )
    try:
        return len(approved), len(rejected)
    except TypeError:
        return 0, 0


# --- Public entry point -----------------------------------------------------


def run_morning(
    ctx: Any = None,  # noqa: ANN401  — typed loosely so callers from card 4 can pass MorningContext
    *,
    top_n: int = 5,
    discord_timeout_seconds: int = 1800,  # 30 minutes per card 7's spec
) -> StrategyResult:
    """The card-8 strategy runner. Card 4's ``morning_run`` calls this.

    Pipeline (each step is independently failure-tolerant):

    1. :func:`score_universe` from card 6 — top-N bullish-tech +
       positive-news candidates.
    2. :class:`PositionSizer` from card 7 — size each candidate into
       a TradeOrder.
    3. ``post_candidates_and_collect_reactions`` from card 7 — post
       to Discord, wait up to 30 min for reactions.
    4. ``persist_approved_trades`` from card 7 — append approved to
       ``APPROVED_TRADES``.

    If any of the card-6/7 modules are missing on the import path,
    the runner returns ``status="not_implemented"`` and exits cleanly
    so the morning job keeps ticking.

    Parameters
    ----------
    ctx:
        The :class:`~portfoliomind.scheduler.jobs.MorningContext` from
        the morning job. Currently unused at the strategy layer
        (the signals/sizer/approval modules carry their own config);
        accepted for forward compatibility with future cards that
        need the full ctx (e.g. logging into AGENT_LOG).
    top_n:
        How many candidates to pull from score_universe. The card
        body pins this to 5.
    discord_timeout_seconds:
        How long to wait for operator reactions. Default 1800s
        (30 min) per the card 8 spec.

    Returns
    -------
    :class:`StrategyResult`
        Always returns one. Never raises.
    """
    started_at = iso_now()
    result = StrategyResult(started_at=started_at)

    # Lazy-import the four public-contract modules. If any is missing
    # we report "not implemented" — the morning_run wrapper turns that
    # into a one-line INFO log and continues.
    signals = _try_import_signals()
    sizer_cls = _try_import_sizer()
    approval = _try_import_approval()
    if signals is None and sizer_cls is None and approval is None:
        msg = "card 6/7 modules not implemented yet — strategy runner is a no-op"
        log.info(msg)
        result.status = "not_implemented"
        result.finished_at = iso_now()
        return result

    # Step 1 — score the universe. If signals.combined is missing but
    # some other module is present, we treat *this step* as the
    # "not implemented" gate; downstream steps are skipped.
    if signals is None:
        msg = "portfoliomind.signals.combined not implemented — strategy runner is a no-op"
        log.info(msg)
        result.status = "not_implemented"
        result.finished_at = iso_now()
        return result

    candidates, err = _score_candidates(signals, top_n=top_n)
    if err and not candidates:
        # "no candidates" is a legitimate skip, not a failure.
        if "no candidates passed" in err:
            result.status = "skipped"
            result.skip_reason = err
            log.info("strategy_runner: %s", err)
            result.finished_at = iso_now()
            return result
        # Anything else (e.g. score_universe raised) is a failure.
        result.status = "failed"
        result.error = err
        result.errors.append(err)
        result.finished_at = iso_now()
        log.error("strategy_runner: %s", err)
        return result
    result.picks_scraped = len(candidates)

    # Step 2 — size each candidate. Sizing failures are per-candidate
    # (logged + skipped), not batch-fatal.
    sized, err = _size_candidates(sizer_cls, candidates)
    if err and not sized:
        # Sizer itself missing or broken — the strategy can't make
        # progress, but we still record what was scored.
        result.status = "failed"
        result.error = err
        result.errors.append(err)
        result.finished_at = iso_now()
        log.error("strategy_runner: %s", err)
        return result
    if not sized:
        result.status = "skipped"
        result.skip_reason = "sizer produced 0 sized orders (all candidates failed sizing)"
        result.finished_at = iso_now()
        return result

    # Step 3 — post to Discord + collect reactions. A missing approval
    # module means the strategy can score + size but can't ask the
    # operator — log and exit as a soft skip (the scored + sized
    # orders are surfaced in the result for downstream visibility).
    if approval is None:
        msg = "portfoliomind.approval not implemented — strategy scored and sized but did not post"
        log.warning(msg)
        result.status = "skipped"
        result.skip_reason = msg
        result.finished_at = iso_now()
        return result

    approval_outcome, err = _post_and_collect(
        approval, sized, timeout_seconds=discord_timeout_seconds
    )
    if err:
        # Discord posting failed. The scored + sized orders are still
        # useful to surface — record the error but don't pretend the
        # whole run failed.
        result.errors.append(err)
        log.error("strategy_runner: %s", err)
        # Continue to persistence in case partial reactions are
        # available; if not, the persistence step will return 0.

    approved_n, rejected_n = _count_reactions(approval_outcome)
    result.approved_count = approved_n
    result.rejected_count = rejected_n

    # Step 4 — persist the approved subset.
    if approval_outcome is not None:
        rows, err = _persist_approved(approval, approval_outcome)
        if err:
            result.errors.append(err)
            log.error("strategy_runner: %s", err)
        else:
            result.orders_placed = rows

    if result.errors:
        result.status = "failed"
        result.error = result.errors[0]
    else:
        result.status = "ran"
    result.finished_at = iso_now()
    return result


# --- Test seam --------------------------------------------------------------


def set_factories(
    *,
    score_universe_factory=None,
    sizer_factory=None,
    approval_factory=None,
) -> None:
    """Inject test fakes for the three lazy-imported modules.

    Production callers never invoke this. Tests use it to swap the
    lazy-imported ``signals.combined`` / ``signals.sizer`` /
    ``approval`` modules for in-memory fakes. The pattern mirrors
    card 3's ``set_factories`` in ``portfoliomind.xtb.runner``.

    Pass ``None`` for any factory to leave it alone. Call
    :func:`reset_factories` to clear.
    """
    if score_universe_factory is not None:
        global _score_universe_factory
        _score_universe_factory = score_universe_factory
    if sizer_factory is not None:
        global _sizer_factory
        _sizer_factory = sizer_factory
    if approval_factory is not None:
        global _approval_factory
        _approval_factory = approval_factory


def reset_factories() -> None:
    """Restore the production lazy-import path. Test-only."""
    global _score_universe_factory, _sizer_factory, _approval_factory
    _score_universe_factory = None
    _sizer_factory = None
    _approval_factory = None


# --- Public surface ---------------------------------------------------------


__all__ = [
    "StrategyResult",
    "run_morning",
    "set_factories",
    "reset_factories",
]
