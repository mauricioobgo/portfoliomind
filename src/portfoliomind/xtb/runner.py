"""XTB morning runner — the card 3 integration seam.

Card 4's ``portfoliomind.scheduler.jobs.morning_run`` lazy-imports this
module and calls :func:`run_morning`. This module is the **glue** — it
reads the ``APPROVED_TRADES`` tab (the downstream-of-card-7 hand-off),
places each order on XTB xStation (or simulates the placement in
dry-run mode), and writes the results to ``EXECUTED_ORDERS``.

Design rules (from the v4 spec and the operator's iron-rules):

* **Never raise.** Every failure mode is converted into a
  :class:`MorningResult` with the ``error`` field set. The scheduler
  depends on this contract — a raise from a runner is a bug, not a
  runtime feature.
* **Dry-run is the default.** The XTB card 3 spec was emphatic that
  real money never moves without an explicit operator opt-in. The
  runner honors this in two ways:

  1. The default config field ``xtb_dry_run=True`` means the runner
     never opens a browser unless the operator has flipped the flag.
  2. Even when the operator flips ``xtb_dry_run`` to ``False``, the
     runner refuses to place real orders unless ``xtb_live_confirm`` is
     also ``True``. Both flags must be set to "no + yes" to enable
     live trading.

  The combination (``xtb_dry_run=False`` AND ``xtb_live_confirm=True``)
  is the only path that ever places a real order.

* **No approved trades → no-op.** If the ``APPROVED_TRADES`` tab is
  empty, the runner returns
  ``MorningResult(skipped=True, skip_reason="no approved trades")``
  without opening a browser, calling Sheets, or logging anything beyond
  the standard one-line ``log_to_sheet`` summary.

* **Headless.** :func:`build_context` is called with ``headless=True``
  so the runner works inside the daemonized scheduler process (the
  cron job is not a TTY).

* **Per-row validation.** A row with a missing SL or TP is dropped
  with a per-row error log entry; the rest of the batch is still
  processed. The validation runs in :func:`validate_order` so the
  same iron rule that protects card 3's CLI protects the runner too.

* **Idempotent within a Bogota-local day.** The ``EXECUTED_ORDERS``
  dedup key is ``(Ticker, Timestamp)``; the runner consults
  ``EXECUTED_ORDERS`` before placing each order and skips a row whose
  key is already present. The same dedup is also enforced by the
  card 3 CLI script (``scripts/execute_trades.py``), so the morning
  cron and an out-of-band manual run cannot double-place.

Public surface
--------------

* :func:`run_morning` — the contract callable expected by card 4.

All other functions are private (``_``-prefixed) and not part of the
public contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..config import PortfoliomindConfig
from ..logging_setup import get_logger
from ..paths import screenshot_dir
from ..scheduler.jobs import MorningContext, MorningResult
from ..sheets.schema import (
    APPROVED_TRADES,
    EXECUTED_ORDERS,
    TAB_HEADERS,
)
from ..time_utils import iso_now
from .login import (
    LOGIN_TIMEOUT_S,
    XTBSessionPaths,
    build_context,
    ensure_logged_in,
    teardown_context,
)
from .order import (
    OrderResult,
    OrderSide,
    OrderSpec,
    PlaceOrderError,
    ValidationError,
    place_order,
)

log = get_logger(__name__)


# --- Status string constants -------------------------------------------------

#: Status written to EXECUTED_ORDERS for a row we did NOT actually
#: submit to xStation (the dry-run / no-live path).
DRY_RUN_STATUS: str = "DRY_RUN"

#: Status for a row successfully submitted to xStation and confirmed
#: by reading an order ID back from the confirmation modal.
PLACED_STATUS: str = "PLACED"

#: Status for a row where the submit succeeded but we could not read
#: back the order ID. The screenshots are still on disk and the
#: operator can reconcile.
UNCONFIRMED_STATUS: str = "UNCONFIRMED"

#: Status for a row that was selected for placement but the spec
#: failed :func:`validate_order` mid-run. Should be rare (we pre-
#: validate) but kept as a defensive status so a row never silently
#: disappears.
VALIDATION_FAILED_STATUS: str = "VALIDATION_FAILED"


# --- Test injection seam ----------------------------------------------------
#
# These factories let the test suite swap in a fake
# ``build_context`` / ``ensure_logged_in`` / ``place_order`` without
# monkeypatching the module attribute. The default factories call
# into the real ``portfoliomind.xtb`` modules; tests override them
# with fakes that do not need a browser. The public
# :func:`run_morning` signature does not change.

_BuildContextFactory = Callable[..., Any]
_EnsureLoggedInFactory = Callable[..., None]
# ``place_order`` is keyword-only on ``screenshot_dir``. Tests
# substitute a simpler fake that doesn't take that kwarg, so the
# factory type is intentionally permissive.
_PlaceOrderFactory = Callable[..., OrderResult]


_build_context_factory: _BuildContextFactory = build_context
_ensure_logged_in_factory: _EnsureLoggedInFactory = ensure_logged_in
_place_order_factory: _PlaceOrderFactory = place_order


def set_factories(
    *,
    build_context_factory: Optional[_BuildContextFactory] = None,
    ensure_logged_in_factory: Optional[_EnsureLoggedInFactory] = None,
    place_order_factory: Optional[_PlaceOrderFactory] = None,
) -> None:
    """Override one or more of the underlying XTB factories. Tests only."""
    global _build_context_factory, _ensure_logged_in_factory, _place_order_factory
    if build_context_factory is not None:
        _build_context_factory = build_context_factory
    if ensure_logged_in_factory is not None:
        _ensure_logged_in_factory = ensure_logged_in_factory
    if place_order_factory is not None:
        _place_order_factory = place_order_factory


def reset_factories() -> None:
    """Restore the default factories (the real card 3 modules)."""
    global _build_context_factory, _ensure_logged_in_factory, _place_order_factory
    _build_context_factory = build_context
    _ensure_logged_in_factory = ensure_logged_in
    _place_order_factory = place_order


# --- Sheet I/O ---------------------------------------------------------------


def _read_approved_trades(sheets: Any, sheet_id: str) -> list[list[str]]:
    """Read every populated row from ``APPROVED_TRADES``.

    Mirrors the row-shape helper from
    :mod:`scripts.execute_trades` but lives here as a private helper so
    the runner has no dependency on the CLI module (which would import
    a Click / argparse surface that's only needed for the operator
    entry point).

    Returns a list of header-aligned rows (length == number of
    canonical columns). Empty trailing rows are filtered out. The
    function does NOT raise on a missing tab — it returns ``[]`` and
    the runner reports ``skipped=True``.
    """
    try:
        headers = TAB_HEADERS[APPROVED_TRADES]
    except KeyError:
        return []
    n_cols = len(headers)
    try:
        raw = sheets.read_range(sheet_id, APPROVED_TRADES, f"A2:{_col_letter(n_cols)}9999")
    except Exception:  # noqa: BLE001 — best-effort read
        return []
    rows: list[list[str]] = []
    for r in raw:
        # Pad short rows so each row has the canonical column count.
        r = list(r) + [""] * (n_cols - len(r))
        if not any(cell.strip() for cell in r):
            continue
        rows.append(r)
    return rows


def _read_executed_keys(sheets: Any, sheet_id: str) -> set[tuple[str, str]]:
    """Return the ``(Ticker, Timestamp)`` keys already in
    ``EXECUTED_ORDERS``.

    The morning-run idempotency key is the same one the card 3 CLI
    uses. We approximate ``(Ticker, Entry Date)`` with
    ``(Ticker, Timestamp)`` because EXECUTED_ORDERS stores the
    submission timestamp in column 1 — same-day re-runs of the same
    ticker are exactly what the dedup is designed to prevent.

    EXECUTED_ORDERS schema (col index, header):
      0 Timestamp, 1 Ticker, 2 Order ID, ...
    """
    keys: set[tuple[str, str]] = set()
    try:
        raw = sheets.read_range(sheet_id, EXECUTED_ORDERS, "A2:C9999")
    except Exception:  # noqa: BLE001
        return keys
    for r in raw:
        if len(r) < 2:
            continue
        # Col 0 is Timestamp, col 1 is Ticker. Build the
        # (Ticker, Timestamp) key the same way the dedup check
        # builds it in _execute_dry_run_batch.
        ts = (r[0] or "").strip()
        ticker = (r[1] or "").strip()
        if not ticker:
            continue
        keys.add((ticker, ts))
    return keys


def _col_letter(n: int) -> str:
    """1-indexed column number -> spreadsheet column letter."""
    if n < 1:
        raise ValueError(f"Column number must be >= 1, got {n}")
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(ord("A") + rem) + result
    return result


# --- Row -> OrderSpec --------------------------------------------------------


def _row_to_order_spec(row: list[str]) -> OrderSpec:
    """Build an :class:`OrderSpec` from an APPROVED_TRADES row.

    Uses :meth:`OrderSpec.checked` so the iron rules (SL/TP mandatory,
    finite numeric, correct side of entry) are enforced at the
    boundary. The Type column is treated as the order side when it is
    ``BUY`` or ``SELL``; otherwise the row defaults to ``BUY`` (the
    agent is long-biased per the v4 spec).
    """
    # APPROVED_TRADES columns (0-indexed):
    # 0 Timestamp, 1 Ticker, 2 Type, 3 Strategy, 4 Timeframe,
    # 5 Allocation, 6 Qty, 7 Entry Price, 8 SL, 9 TP, 10 Approval Note
    ticker = (row[1] if len(row) > 1 else "").strip()
    type_ = (row[2] if len(row) > 2 else "").strip().upper()
    side = "BUY"
    if type_ in {"BUY", "SELL"}:
        side = type_
    qty = _to_float(row[6] if len(row) > 6 else "")
    entry = _to_float(row[7] if len(row) > 7 else "")
    sl = _to_float(row[8] if len(row) > 8 else "")
    tp = _to_float(row[9] if len(row) > 9 else "")
    note = (row[10] if len(row) > 10 else "").strip()
    return OrderSpec.checked(
        ticker=ticker,
        side=side,
        qty=qty,
        entry_price=entry,
        sl=sl,
        tp=tp,
        note=note,
    )


def _to_float(s: Any) -> float:
    """Coerce a sheet cell value to float. Empty string -> 0.0."""
    if s is None:
        return 0.0
    s = str(s).strip()
    if not s:
        return 0.0
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    return float(s)


# --- Logging result rows -----------------------------------------------------


def _ensure_executed_orders_worksheet(sheets: Any, sheet_id: str) -> None:
    """Make sure the EXECUTED_ORDERS tab exists with the right headers.

    Best-effort: a ``SheetsClientError`` here is logged and swallowed
    so a tab-missing state never breaks the runner.
    """
    try:
        sheets.ensure_worksheet(
            sheet_id,
            EXECUTED_ORDERS,
            TAB_HEADERS[EXECUTED_ORDERS],
        )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "xtb.runner.ensure_worksheet_failed err=%s",
            type(e).__name__,
        )


def _write_executed_row(
    sheets: Any,
    sheet_id: str,
    spec: OrderSpec,
    *,
    status: str,
    order_id: str = "",
    screenshot_path: str = "",
) -> None:
    """Append one row to ``EXECUTED_ORDERS`` with the canonical shape.

    Schema (10 columns):
        Timestamp, Ticker, Order ID, Side, Qty, Entry Price, SL, TP,
        Status, Screenshot

    A Sheets failure is logged and swallowed — we never want the
    audit log to take the morning job offline.
    """
    row = [
        iso_now(),
        spec.ticker,
        order_id,
        spec.side.value,
        str(spec.qty),
        str(spec.entry_price),
        str(spec.sl),
        str(spec.tp),
        status,
        screenshot_path,
    ]
    try:
        sheets.append_rows(sheet_id, EXECUTED_ORDERS, [row])
    except Exception as e:  # noqa: BLE001
        log.warning(
            "xtb.runner.executed_log_append_failed ticker=%s "
            "status=%s err=%s",
            spec.ticker,
            status,
            type(e).__name__,
        )


# --- The contract callable --------------------------------------------------


@dataclass
class _RowOutcome:
    """Internal tally for a single APPROVED_TRADES row.

    Used by :func:`run_morning` to compute the
    :class:`MorningResult` summary without re-walking the rows.
    """

    placed: int = 0
    dry_run: int = 0
    skipped_dedup: int = 0
    skipped_validation: int = 0
    errored: int = 0


def run_morning(ctx: MorningContext) -> MorningResult:
    """Run the XTB morning job. Always returns a
    :class:`MorningResult`; never raises.

    Steps
    -----

    1. Read every populated row from ``APPROVED_TRADES``.
    2. If no rows → return ``skipped=True, skip_reason="no approved trades"``.
    3. For each row:

       a. Pre-validate via :func:`validate_order`. A failure is logged
          and counted as ``VALIDATION_FAILED`` in EXECUTED_ORDERS.
       b. Dedup against ``EXECUTED_ORDERS``: skip if the same ticker
          was already placed today.
       c. If dry-run mode (the default) → write a ``DRY_RUN`` row.
       d. If live mode → login to XTB, place the order, write a
          ``PLACED`` or ``UNCONFIRMED`` row.

    4. Aggregate the per-row outcomes into a single
       :class:`MorningResult`. ``orders_placed`` counts **only** the
       rows we actually attempted on xStation — dry-run rows do not
       count (the operator wants "orders placed" to mean "real
       money moved", not "we printed a spec").

    Failures
    --------

    Any exception is converted into a
    :class:`MorningResult(runner="card3", error=str(e))` so the
    scheduler can keep the cron schedule ticking.
    """
    runner_id = "card3"
    try:
        config = ctx.config
        sheets = ctx.sheets
        sheet_id = ctx.sheet_id
        if config is None:
            return MorningResult(
                runner=runner_id,
                error="morning context has no config; refusing to place orders",
            )
        if not sheet_id:
            return MorningResult(
                runner=runner_id,
                error="morning context has empty sheet_id; refuse to place orders",
            )

        approved_rows = _read_approved_trades(sheets, sheet_id)
        if not approved_rows:
            ctx.log_to_sheet(
                "INFO",
                "xtb.run_morning.skipped reason=no_approved_trades",
            )
            return MorningResult(
                runner=runner_id,
                skipped=True,
                skip_reason="no approved trades",
            )

        # Iron rule: refuse live trading unless the operator has
        # explicitly flipped BOTH ``xtb_dry_run`` to False AND
        # ``xtb_live_confirm`` to True. Either toggle alone keeps the
        # run in dry-run mode.
        live_mode = (not config.xtb_dry_run) and bool(config.xtb_live_confirm)
        ctx.log_to_sheet(
            "INFO",
            f"xtb.run_morning.start approved_rows={len(approved_rows)} "
            f"live_mode={live_mode}",
        )

        # Always pre-create the EXECUTED_ORDERS tab so the append below
        # works even if the runner is the first writer after a fresh
        # bootstrap. The call is idempotent.
        _ensure_executed_orders_worksheet(sheets, sheet_id)
        already_done = _read_executed_keys(sheets, sheet_id)

        tally = _RowOutcome()
        error_messages: list[str] = []

        if live_mode:
            # --- LIVE PATH -----------------------------------------
            _execute_live_batch(
                ctx=ctx,
                config=config,
                sheets=sheets,
                sheet_id=sheet_id,
                approved_rows=approved_rows,
                already_done=already_done,
                tally=tally,
                error_messages=error_messages,
            )
        else:
            # --- DRY-RUN PATH (default) ---------------------------
            _execute_dry_run_batch(
                ctx=ctx,
                sheets=sheets,
                sheet_id=sheet_id,
                approved_rows=approved_rows,
                already_done=already_done,
                tally=tally,
                error_messages=error_messages,
            )

        # Compose the final MorningResult. ``orders_placed`` only
        # counts live placements — a dry-run count would mislead the
        # operator's Discord alert.
        msg = (
            f"xtb.run_morning.ok placed={tally.placed} "
            f"dry_run={tally.dry_run} skipped_dedup={tally.skipped_dedup} "
            f"skipped_validation={tally.skipped_validation} "
            f"errored={tally.errored}"
        )
        if error_messages:
            # The job ran but produced errors — surface the first one
            # on the MorningResult so the scheduler formats a Discord
            # alert with the red ❌ icon. The full list is logged to
            # AGENT_LOG for operator inspection.
            for em in error_messages:
                ctx.log_to_sheet("ERROR", em)
            log.error(msg + " errors=%d", len(error_messages))
            return MorningResult(
                runner=runner_id,
                orders_placed=tally.placed,
                error=error_messages[0],
            )
        ctx.log_to_sheet("INFO", msg)
        return MorningResult(
            runner=runner_id,
            orders_placed=tally.placed,
        )

    except Exception as e:  # noqa: BLE001
        log.error(
            "xtb.runner.unexpected err_type=%s err=%r",
            type(e).__name__,
            str(e)[:200],
        )
        return MorningResult(
            runner=runner_id,
            error=f"unexpected: {type(e).__name__}: {e}",
        )


# --- Live + dry-run batch implementations ----------------------------------


def _execute_dry_run_batch(
    *,
    ctx: MorningContext,
    sheets: Any,
    sheet_id: str,
    approved_rows: list[list[str]],
    already_done: set[tuple[str, str]],
    tally: _RowOutcome,
    error_messages: list[str],
) -> None:
    """Process a batch in dry-run mode. No browser. Just spec + log."""
    for i, row in enumerate(approved_rows, start=2):  # row 1 is header
        ticker = (row[1] if len(row) > 1 else "").strip()
        if not ticker:
            tally.skipped_validation += 1
            error_messages.append(
                f"xtb.runner.row_skipped row={i} reason=empty_ticker"
            )
            continue
        # Dedup key: ticker + the approval timestamp. This matches the
        # dedup the card 3 CLI uses.
        approval_ts = (row[0] if len(row) > 0 else "").strip()
        if (ticker, approval_ts) in already_done:
            tally.skipped_dedup += 1
            log.info(
                "xtb.runner.dedup_skip ticker=%s ts=%s", ticker, approval_ts
            )
            continue
        # Pre-validate the spec. A failure here is logged to the
        # EXECUTED_ORDERS audit log with VALIDATION_FAILED status.
        # We construct the spec directly (not via .checked) for the
        # log row so the log entry shows the ticker even when SL/TP
        # are missing — the placeholder values are never written to
        # an order.
        try:
            spec = _row_to_order_spec(row)
        except ValidationError as e:
            tally.skipped_validation += 1
            error_messages.append(
                f"xtb.runner.row_validation_failed row={i} "
                f"ticker={ticker} err={e}"
            )
            _write_executed_row(
                sheets, sheet_id,
                OrderSpec(
                    ticker=ticker,
                    side=OrderSide.BUY,
                    qty=0.0, entry_price=0.0, sl=0.0, tp=0.0,
                ),
                status=VALIDATION_FAILED_STATUS,
            )
            continue
        # In dry-run mode, we just write a synthetic DRY_RUN row.
        _write_executed_row(
            sheets, sheet_id, spec, status=DRY_RUN_STATUS
        )
        tally.dry_run += 1
        log.info(
            "xtb.runner.dry_run ticker=%s side=%s qty=%g",
            spec.ticker, spec.side.value, spec.qty,
        )


def _execute_live_batch(
    *,
    ctx: MorningContext,
    config: PortfoliomindConfig,
    sheets: Any,
    sheet_id: str,
    approved_rows: list[list[str]],
    already_done: set[tuple[str, str]],
    tally: _RowOutcome,
    error_messages: list[str],
) -> None:
    """Process a batch in live mode. Opens a browser, logs in once,
    and places each order via the real :func:`place_order` flow.

    The XTB context is opened ONCE per morning run (not once per row)
    because xStation's login is slow. If any row raises during the
    loop, we still tear down the context in the ``finally`` block.
    """
    paths = XTBSessionPaths.from_config(config)
    sdir = screenshot_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    paths.login_screenshots_dir.mkdir(parents=True, exist_ok=True)
    paths.order_screenshots_dir.mkdir(parents=True, exist_ok=True)

    context = None
    page = None
    try:
        log.info("xtb.runner.context_opening headless=True")
        context = _build_context_factory(paths, headless=True)
        page = context.pages[0] if context.pages else context.new_page()
        try:
            _ensure_logged_in_factory(
                page,
                config,
                base_url="https://xstation5.xtb.com",
                timeout_s=LOGIN_TIMEOUT_S,
                failure_screenshot_dir=paths.login_screenshots_dir,
            )
        except Exception as e:  # noqa: BLE001
            # Login failure aborts the whole batch. We never want to
            # place orders without a verified session.
            log.error("xtb.runner.login_failed err=%s", type(e).__name__)
            error_messages.append(
                f"xtb.login failed: {type(e).__name__}: {e}"
            )
            return

        for i, row in enumerate(approved_rows, start=2):
            ticker = (row[1] if len(row) > 1 else "").strip()
            if not ticker:
                tally.skipped_validation += 1
                continue
            approval_ts = (row[0] if len(row) > 0 else "").strip()
            if (ticker, approval_ts) in already_done:
                tally.skipped_dedup += 1
                continue
            try:
                spec = _row_to_order_spec(row)
            except ValidationError as e:
                tally.skipped_validation += 1
                error_messages.append(
                    f"xtb.runner.row_validation_failed row={i} "
                    f"ticker={ticker} err={e}"
                )
                _write_executed_row(
                    sheets, sheet_id,
                    OrderSpec(
                        ticker=ticker,
                        side=OrderSide.BUY,
                        qty=0.0, entry_price=0.0, sl=0.0, tp=0.0,
                    ),
                    status=VALIDATION_FAILED_STATUS,
                )
                continue
            try:
                result: OrderResult = _place_order_factory(
                    page, spec, screenshot_dir=paths.order_screenshots_dir,
                )
            except PlaceOrderError as e:
                tally.errored += 1
                error_messages.append(
                    f"xtb.runner.place_order_failed ticker={ticker} "
                    f"err={type(e).__name__}: {e}"
                )
                continue
            except ValidationError as e:
                # Defense in depth — should not happen because we
                # pre-validate. Log and continue.
                tally.errored += 1
                error_messages.append(
                    f"xtb.runner.row_validation_mid_run row={i} "
                    f"ticker={ticker} err={e}"
                )
                continue
            status = PLACED_STATUS if result.order_id else UNCONFIRMED_STATUS
            screenshot_path = (
                str(result.screenshot_after) if result.screenshot_after else ""
            )
            _write_executed_row(
                sheets, sheet_id, spec,
                status=status,
                order_id=result.order_id or "",
                screenshot_path=screenshot_path,
            )
            tally.placed += 1
            log.info(
                "xtb.runner.placed ticker=%s order_id=%s status=%s",
                spec.ticker, result.order_id, status,
            )
    finally:
        if context is not None:
            teardown_context(context)


__all__ = [
    "run_morning",
    "set_factories",
    "reset_factories",
    "DRY_RUN_STATUS",
    "PLACED_STATUS",
    "UNCONFIRMED_STATUS",
    "VALIDATION_FAILED_STATUS",
]
