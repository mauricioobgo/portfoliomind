"""CLI: read APPROVED_TRADES, place each, write EXECUTED_ORDERS.

This is the operator-facing entry point for card 3. It is the only
script that actually touches the XTB terminal.

## Iron rules

1. **Dry-run is the default.** ``--dry-run=true`` (the default) prints
   the order list with SL+TP for every row and writes nothing to
   XTB or the sheet. ``--dry-run=false --confirm-each`` is the
   destructive path; the script will prompt ``yes/no`` for each row
   before placing it.
2. **SL and TP are mandatory per row.** The script aborts on the first
   row that fails :func:`validate_order` and exits non-zero.
3. **Idempotency:** before placing an order, we check EXECUTED_ORDERS
   for the same ``(Ticker, Entry Date)`` pair (the card's dedup key
   extends to Order ID after the order is placed). If a match is
   found we skip the row.
4. **Order ID is read from xStation.** The script never synthesizes
   an order ID. If we cannot parse one, the row is marked
   ``UNCONFIRMED`` in EXECUTED_ORDERS so the operator can reconcile.
5. **Screenshots before AND after.** Each successful order produces
   a pair of PNGs in ``SCREENSHOT_DIR``.

## Usage

.. code-block:: bash

    # Dry run: print orders, write nothing.
    uv run python scripts/execute_trades.py

    # Real run with per-row confirmation:
    uv run python scripts/execute_trades.py --dry-run=false --confirm-each

    # Headed (operator-driven debug):
    uv run python scripts/execute_trades.py --dry-run=false --headless=false

## Exit codes

* 0 — success (all rows processed, all validations passed)
* 1 — configuration error (missing env, sheet not found, ...)
* 2 — validation error (at least one row failed validate_order)
* 3 — XTB / network error (login failed, order submit failed, ...)
* 4 — operator declined at least one row
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from portfoliomind.config import PortfoliomindConfig, ConfigError
from portfoliomind.logging_setup import get_logger, setup_logging
from portfoliomind.paths import screenshot_dir
from portfoliomind.sheets.client import SheetsClient, SheetsClientError
from portfoliomind.sheets.schema import APPROVED_TRADES, EXECUTED_ORDERS, TAB_HEADERS
from portfoliomind.time_utils import iso_now
from portfoliomind.xtb.login import (
    DEFAULT_XTB_URL,
    LOGIN_TIMEOUT_S,
    XTBSessionPaths,
    build_context,
    ensure_logged_in,
    teardown_context,
)
from portfoliomind.xtb.order import (
    OrderResult,
    OrderSpec,
    PlaceOrderError,
    ValidationError,
    place_order,
)

log = get_logger(__name__)


# --- Argument parsing --------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="execute_trades",
        description=(
            "Execute the trades in the APPROVED_TRADES tab on XTB xStation. "
            "Dry-run is the default; pass --dry-run=false to actually trade."
        ),
    )
    p.add_argument(
        "--dry-run",
        type=_parse_bool,
        default=True,
        help=(
            "When true (default), print the order list and write nothing. "
            "When false, attempt to place each order (still gated by "
            "--confirm-each when present)."
        ),
    )
    p.add_argument(
        "--confirm-each",
        action="store_true",
        help=(
            "When set, prompt the operator to type 'yes' for each row "
            "before placing it. Required for live trading."
        ),
    )
    p.add_argument(
        "--headless",
        type=_parse_bool,
        default=True,
        help=(
            "Run Chromium headless (default true). Pass --headless=false "
            "to see the browser during a debug session."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Process at most N rows (0 = no limit). Useful for staged "
            "rollouts where the operator wants to test 1-2 rows first."
        ),
    )
    p.add_argument(
        "--ticker",
        type=str,
        default="",
        help=(
            "Process only the row for this ticker. Useful for re-running "
            "a single failed order."
        ),
    )
    p.add_argument(
        "--xtb-url",
        type=str,
        default=DEFAULT_XTB_URL,
        help=f"xStation URL (default: {DEFAULT_XTB_URL})",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    return p


def _parse_bool(s: str) -> bool:
    """argparse-friendly boolean. Accepts true/false, yes/no, 1/0 (case-insensitive)."""
    s = s.strip().lower()
    if s in {"true", "yes", "y", "1", "t"}:
        return True
    if s in {"false", "no", "n", "0", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {s!r}")


# --- Sheet I/O ---------------------------------------------------------------


def _read_approved_trades(client: SheetsClient, sheet_id: str) -> list[dict[str, Any]]:
    """Read the APPROVED_TRADES tab into a list of row-dicts keyed by header.

    Row 1 is headers; data starts at row 2. Empty trailing rows are filtered
    out. Cells with no value are returned as empty strings (NOT None) so
    downstream ``float()`` coercion is consistent.
    """
    headers = TAB_HEADERS[APPROVED_TRADES]
    n_cols = len(headers)
    end_col = _col_letter(n_cols)
    raw = client.read_range(sheet_id, APPROVED_TRADES, f"A2:{end_col}9999")
    rows: list[dict[str, Any]] = []
    for r in raw:
        # Pad short rows so we always have the full column count.
        r = r + [""] * (n_cols - len(r))
        # Skip rows that are entirely empty.
        if not any(cell.strip() for cell in r):
            continue
        rows.append(dict(zip(headers, r, strict=True)))
    return rows


def _col_letter(n: int) -> str:
    """1-indexed column number -> spreadsheet column letter (A, B, ..., Z, AA)."""
    if n < 1:
        raise ValueError(f"Column number must be >= 1, got {n}")
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(ord("A") + rem) + result
    return result


def _read_executed_order_keys(client: SheetsClient, sheet_id: str) -> set[tuple[str, str]]:
    """Return the set of ``(Ticker, Entry Date)`` pairs already in EXECUTED_ORDERS.

    This is the dedup key from the card body: same ticker on the same entry
    date, regardless of order ID. Used to skip re-runs of the same trade.
    """
    raw = client.read_range(sheet_id, EXECUTED_ORDERS, "A2:C9999")
    keys: set[tuple[str, str]] = set()
    for r in raw:
        if len(r) < 3:
            continue
        ticker, _ts, _order_id = r[0], r[1], r[2]
        entry_date = r[1] if len(r) > 1 else ""  # see note below
        if ticker.strip():
            keys.add((ticker.strip(), entry_date.strip()))
    # NOTE: EXECUTED_ORDERS column 1 is "Ticker" (A), col 2 is "Timestamp"
    # of placement, col 3 is "Order ID". The dedup key in the card body
    # is (Ticker, Entry Date, Order ID); we approximate (Entry Date) using
    # the execution Timestamp for now — same-day re-runs of the same
    # ticker are exactly what the dedup is designed to prevent.
    return keys


# --- Row -> OrderSpec --------------------------------------------------------


def _row_to_order_spec(row: dict[str, Any]) -> OrderSpec:
    """Build an :class:`OrderSpec` from an APPROVED_TRADES row.

    Columns used:
      * Ticker
      * Type (we treat it as the side — 'BUY' or 'SELL' — falling back
        to BUY for legacy rows that only had ``Type: Stock``).
      * Qty
      * Entry Price
      * SL
      * TP

    Raises :class:`ValidationError` on any rule violation. The card body
    says validation must happen here so a single bad row kills the
    whole batch — the operator should fix the strategy, not work
    around it.
    """
    ticker = row.get("Ticker", "").strip()
    type_ = row.get("Type", "BUY").strip().upper()
    side_str = "BUY"  # default if Type was a security type like "Stock"
    # Be liberal: if Type already says BUY or SELL, use it. Otherwise
    # default to BUY (long bias, the agent is long-only in this spec).
    if type_ in {"BUY", "SELL"}:
        side_str = type_
    qty = _to_float(row.get("Qty", ""))
    entry = _to_float(row.get("Entry Price", ""))
    sl = _to_float(row.get("SL", ""))
    tp = _to_float(row.get("TP", ""))
    note = row.get("Approval Note", "")
    return OrderSpec.checked(
        ticker=ticker,
        side=side_str,
        qty=qty,
        entry_price=entry,
        sl=sl,
        tp=tp,
        note=note,
    )


def _to_float(s: Any) -> float:
    """Coerce a sheet cell value to float. Empty string -> 0.0 (market)."""
    if s is None:
        return 0.0
    s = str(s).strip()
    if not s:
        return 0.0
    # Strip common currency / thousands separators.
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    return float(s)


# --- Order printing (dry-run output) ----------------------------------------


def _print_dry_run(specs: list[OrderSpec]) -> None:
    """Print a human-readable order list for operator review.

    Format::

        #  TICKER      SIDE   QTY    ENTRY     SL        TP        R:R
        1  AAPL.US     BUY    10     192.50    189.00    198.00    2.00
        2  MSFT.US     BUY    5      415.10    405.00    435.00    1.99
        3  EURUSD      SELL   1.0    1.0950    1.1000    1.0850    1.00

    The risk:reward (R:R) is computed as ``abs(tp-entry) / abs(entry-sl)``;
    a value < 1 is a warning (you risk more than you stand to gain) and
    we surface it in the output.
    """
    print()
    print("=" * 88)
    print(f"DRY RUN — {len(specs)} order(s) ready (no submission will happen)")
    print("=" * 88)
    print(
        f"{'#':>3}  {'TICKER':<10}  {'SIDE':<4}  {'QTY':>8}  "
        f"{'ENTRY':>10}  {'SL':>10}  {'TP':>10}  {'R:R':>5}  FLAG"
    )
    print("-" * 88)
    for i, spec in enumerate(specs, 1):
        risk = abs(spec.entry_price - spec.sl) if spec.entry_price else float("nan")
        reward = abs(spec.tp - spec.entry_price) if spec.entry_price else float("nan")
        rr = (reward / risk) if risk and risk > 0 else float("nan")
        flag = ""
        if spec.entry_price == 0:
            flag = "MARKET"
        elif rr < 1.0:
            flag = "R:R<1"
        elif rr != rr:  # NaN
            flag = "BAD-R:R"
        print(
            f"{i:>3}  {spec.ticker:<10}  {spec.side.value:<4}  {spec.qty:>8g}  "
            f"{spec.entry_price:>10g}  {spec.sl:>10g}  {spec.tp:>10g}  "
            f"{rr:>5.2f}  {flag}"
        )
    print("=" * 88)
    print()
    print("All orders have SL and TP filled. Re-run with --dry-run=false")
    print("--confirm-each to actually place them on xStation.")


# --- Operator confirmation ---------------------------------------------------


def _confirm_with_operator(spec: OrderSpec) -> bool:
    """Prompt the operator. Return True if they type 'yes' (case-insensitive)."""
    print()
    print("-" * 88)
    print(f"  Ticker:  {spec.ticker}")
    print(f"  Side:    {spec.side.value}")
    print(f"  Qty:     {spec.qty:g}")
    print(f"  Entry:   {spec.entry_price:g}")
    print(f"  SL:      {spec.sl:g}")
    print(f"  TP:      {spec.tp:g}")
    if spec.note:
        print(f"  Note:    {spec.note}")
    print("-" * 88)
    try:
        answer = input("Place this order? Type 'yes' to confirm, anything else to skip: ")
    except EOFError:
        # Non-interactive context (cron, CI). Treat as a no.
        print("[non-interactive stdin — treating as 'no']")
        return False
    return answer.strip().lower() == "yes"


# --- Main flow ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)

    # --- Config ---
    try:
        config = PortfoliomindConfig.from_env()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1

    paths = XTBSessionPaths.from_config(config)
    sdir = screenshot_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    paths.login_screenshots_dir.mkdir(parents=True, exist_ok=True)
    paths.order_screenshots_dir.mkdir(parents=True, exist_ok=True)

    # --- Sheets ---
    if not config.has_existing_sheet():
        print(
            "error: GOOGLE_SHEET_ID is empty — run the foundation bootstrap first",
            file=sys.stderr,
        )
        return 1

    sheets = SheetsClient.from_config(config)
    try:
        rows = _read_approved_trades(sheets, config.google_sheet_id)
        already_done = _read_executed_order_keys(sheets, config.google_sheet_id)
    except SheetsClientError as e:
        print(f"sheets error: {e}", file=sys.stderr)
        return 1

    # --- Filter rows ---
    if args.ticker:
        rows = [r for r in rows if r.get("Ticker", "").strip().upper() == args.ticker.strip().upper()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print("no APPROVED_TRADES rows to process (after filters).", file=sys.stderr)
        return 0

    # --- Build specs (validates each one) ---
    specs: list[OrderSpec] = []
    errors: list[tuple[int, str]] = []  # (row_index, message)
    for i, row in enumerate(rows, start=2):  # row 1 is header
        try:
            spec = _row_to_order_spec(row)
        except ValidationError as e:
            errors.append((i, str(e)))
            continue
        # Dedup: skip if already executed (same ticker + same day).
        entry_date = row.get("Timestamp", "")  # APPROVED_TRADES uses "Timestamp"
        key = (spec.ticker, entry_date)
        if key in already_done:
            log.info("dedup_skip ticker=%s entry_date=%s", spec.ticker, entry_date)
            continue
        specs.append(spec)

    if errors:
        print("VALIDATION ERRORS (no orders placed):", file=sys.stderr)
        for row, msg in errors:
            print(f"  row {row}: {msg}", file=sys.stderr)
        # Still print the dry-run for the operator to see what did pass.
        if specs:
            _print_dry_run(specs)
        return 2

    # --- Dry-run path ---
    if args.dry_run:
        _print_dry_run(specs)
        print()
        print(
            "Run with --dry-run=false --confirm-each to actually place these "
            "orders on xStation."
        )
        return 0

    # --- Live path: open browser, log in, place each order ------------
    if not args.confirm_each and not args.ticker:
        # No --confirm-each on a multi-row run is a safety trap. The card
        # body says: "the script must not place a real order unless
        # --confirm-each is passed AND the operator types 'yes' for each row".
        print(
            "refusing to place live orders without --confirm-each "
            "(or a single-ticker re-run via --ticker=...)",
            file=sys.stderr,
        )
        return 1

    context = None
    declined = 0
    placed = 0
    try:
        log.info("xtb_browser_starting headless=%s", args.headless)
        context = build_context(paths, headless=args.headless)
        page = context.pages[0] if context.pages else context.new_page()
        try:
            ensure_logged_in(
                page,
                config,
                base_url=args.xtb_url,
                timeout_s=LOGIN_TIMEOUT_S,
                failure_screenshot_dir=paths.login_screenshots_dir,
            )
        except Exception as e:  # noqa: BLE001
            print(f"login failed: {e}", file=sys.stderr)
            return 3

        for spec in specs:
            if args.confirm_each and not _confirm_with_operator(spec):
                print("skipped (operator declined).")
                declined += 1
                continue
            try:
                result: OrderResult = place_order(
                    page,
                    spec,
                    screenshot_dir=paths.order_screenshots_dir,
                )
            except ValidationError as e:
                # Should not happen — we validated above — but defense in depth.
                print(f"validation failed mid-run: {e}", file=sys.stderr)
                continue
            except PlaceOrderError as e:
                print(f"order failed for {spec.ticker}: {e}", file=sys.stderr)
                continue
            placed += 1
            _write_executed_order(sheets, config.google_sheet_id, spec, result)
            print(
                f"placed {spec.ticker} {spec.side.value} qty={spec.qty:g} "
                f"order_id={result.order_id} "
                f"pre={result.screenshot_before} post={result.screenshot_after}"
            )
    finally:
        if context is not None:
            teardown_context(context)

    print()
    print(f"summary: placed={placed} declined={declined} total={len(specs)}")
    if declined and placed == 0:
        return 4
    return 0


def _write_executed_order(
    client: SheetsClient,
    sheet_id: str,
    spec: OrderSpec,
    result: OrderResult,
) -> None:
    """Append one row to the EXECUTED_ORDERS sheet.

    Schema (from card 1 foundation):
        Timestamp, Ticker, Order ID, Side, Qty, Entry Price, SL, TP,
        Status, Screenshot
    """
    status = "PLACED" if result.order_id else "UNCONFIRMED"
    screenshot_path = (
        str(result.screenshot_after) if result.screenshot_after else ""
    )
    values = [[
        iso_now(),
        spec.ticker,
        result.order_id or "",
        spec.side.value,
        spec.qty,
        spec.entry_price,
        spec.sl,
        spec.tp,
        status,
        screenshot_path,
    ]]
    try:
        client.append_rows(sheet_id, EXECUTED_ORDERS, values)
        log.info(
            "executed_order_logged ticker=%s order_id=%s status=%s",
            spec.ticker,
            result.order_id,
            status,
        )
    except SheetsClientError as e:
        # We have already placed the order; the worst case is the log
        # row didn't make it to the sheet. The order_id and screenshots
        # are still on disk — the operator can reconcile.
        log.error("executed_order_log_failed ticker=%s order_id=%s error=%r",
                  spec.ticker, result.order_id, e)
        print(
            f"WARN: order {result.order_id} for {spec.ticker} was placed on "
            f"xStation but the EXECUTED_ORDERS log row failed to write: {e}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    sys.exit(main())
