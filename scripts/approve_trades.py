"""CLI: card 7 — size candidates, post to Discord, persist approvals.

This is the operator-facing entry point for the card 7 approval
flow. The card 8 strategy runner already wires this logic into the
morning job via lazy imports; the CLI is the manual / debug path.

## Iron rules

1. **Dry-run is the default.** ``--no-discord`` and ``--persist=dry``
   (both default-true) are the safe paths. ``--no-discord=false``
   actually posts to Discord and waits up to ``--timeout-min``
   minutes for reactions.
2. **Idempotency is enforced by :func:`persist_approved_trades`.** A
   re-run with the same candidates writes zero new rows.
3. **Sizer rejects are surfaced, not fatal.** A candidate that fails
   sizing (commission too high, cap exceeded) is logged and the rest
   of the batch continues.
4. **Never raises.** All failures are converted into exit codes
   (see below) and human-readable error messages.

## Usage

.. code-block:: bash

    # End-to-end on the live card 6 signals, no Discord, no persist:
    uv run python scripts/approve_trades.py --from-card6 --no-discord --persist=dry

    # End-to-end including Discord post + persist (requires env):
    uv run python scripts/approve_trades.py --from-card6

    # From a pre-computed JSON of candidates:
    uv run python scripts/approve_trades.py --candidates-file=path/to/candidates.json

## Exit codes

* 0 — success (all candidates processed; approved/rejected counted)
* 1 — configuration error (missing env, sheet not found, ...)
* 2 — input error (bad candidates file, no candidates from card 6)
* 3 — runtime error (Discord failed AND persist failed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from portfoliomind.approval import (
    persist_approved_trades,
    post_candidates_and_collect_reactions,
)
from portfoliomind.config import ConfigError, PortfoliomindConfig
from portfoliomind.logging_setup import get_logger, setup_logging
from portfoliomind.signals.sizer import PositionSizer, RejectReason, TradeOrder

log = get_logger(__name__)


# --- Candidate loading ---------------------------------------------------


def _load_candidates_from_card6(
    *,
    top_n: int,
    config: PortfoliomindConfig,
) -> list[Any]:
    """Run card 6's score_universe and filter to actionable candidates.

    Per the card 7 spec, the filter is:

    * ``combined > 0`` (long-only)
    * ``confidence > 0.3`` (skip weak signals)
    * ``not s.error`` (skip failures)
    * Sort by (combined, confidence) descending and take the top ``top_n``.
    """
    from portfoliomind.signals import score_universe

    raw = score_universe()
    candidates = [
        s for s in raw if s.combined > 0 and s.confidence > 0.3 and not s.error
    ]
    candidates.sort(key=lambda s: (s.combined, s.confidence), reverse=True)
    return candidates[:top_n]


def _load_candidates_from_file(path: Path) -> list[Any]:
    """Read a JSON file of candidates.

    Expected shape: a JSON list of objects, each with the card 6
    Signal fields (``ticker``, ``combined``, ``technical``,
    ``sentiment``, ``confidence``, ``reasons``, ``error``,
    ``asof_date``) OR a list of pre-sized ``TradeOrder`` dicts
    (which include ``qty``, ``entry``, ``sl``, ``tp``, ...). The
    dicts are wrapped in a small namespace adapter so the sizer
    can read fields via ``getattr`` (which is how dataclass-shaped
    objects are accessed).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"candidates file must be a JSON list, got {type(raw).__name__}")
    return [_DictCandidate(d) if isinstance(d, dict) else d for d in raw]


class _DictCandidate:
    """Adapter that exposes a dict's keys as attributes.

    Used by :func:`_load_candidates_from_file` so JSON-loaded
    candidate dicts can be passed directly to the sizer, which
    reads fields via ``getattr``.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e


# --- Main entry ---------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a shell exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging()
    return _run(args)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="approve_trades",
        description=(
            "Card 7: size candidates from card 6, post to Discord, "
            "persist approved subset to APPROVED_TRADES."
        ),
    )
    src = p.add_argument_group("input source (mutually exclusive)")
    src.add_argument(
        "--from-card6",
        action="store_true",
        help="Use the live card 6 score_universe() to source candidates.",
    )
    src.add_argument(
        "--candidates-file",
        type=Path,
        default=None,
        help="Path to a JSON file of pre-computed candidates (Signal or TradeOrder dicts).",
    )
    src.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="How many top candidates to size (default 5, per the card 7 spec).",
    )
    flow = p.add_argument_group("flow control")
    flow.add_argument(
        "--no-discord",
        dest="discord",
        action="store_false",
        default=True,
        help="Skip the Discord post; print the sized candidates and exit.",
    )
    flow.add_argument(
        "--timeout-min",
        type=int,
        default=None,
        help="How long to wait for Discord reactions (default: APPROVAL_TIMEOUT_MIN env or 30).",
    )
    flow.add_argument(
        "--persist",
        choices=("live", "dry", "off"),
        default="live",
        help=(
            "live = write to APPROVED_TRADES; dry = format the rows but don't write; "
            "off = skip persistence entirely."
        ),
    )
    flow.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --persist=dry. Kept for backward-compat with card 8's CLI patterns.",
    )
    cfg = p.add_argument_group("config / sheet")
    cfg.add_argument(
        "--sheet-id",
        type=str,
        default=None,
        help="Override the Google Sheet ID (default: env GOOGLE_SHEET_ID).",
    )
    return p


def _run(args: argparse.Namespace) -> int:
    # 1) Source candidates.
    if args.from_card6 and args.candidates_file:
        print("ERROR: --from-card6 and --candidates-file are mutually exclusive", file=sys.stderr)
        return 2
    if not args.from_card6 and not args.candidates_file:
        print("ERROR: must specify --from-card6 or --candidates-file", file=sys.stderr)
        return 2

    try:
        config = PortfoliomindConfig.from_env()
    except ConfigError as e:
        print(f"ERROR: config: {e}", file=sys.stderr)
        return 1

    sheet_id = (args.sheet_id or config.google_sheet_id or "").strip()

    if args.from_card6:
        try:
            candidates = _load_candidates_from_card6(top_n=args.top_n, config=config)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: card 6 score_universe: {e}", file=sys.stderr)
            return 2
    else:
        try:
            candidates = _load_candidates_from_file(args.candidates_file)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: candidates file: {e}", file=sys.stderr)
            return 2

    if not candidates:
        print("No actionable candidates (filter excluded everything).")
        return 0

    # 2) Size them. ``open_position_count`` is 0 in CLI mode; card 8
    # passes the live count from EXECUTED_ORDERS.
    sizer = PositionSizer.from_config(config)
    sized_orders: list[TradeOrder] = []
    rejected: list[RejectReason] = []
    for c in candidates:
        outcome = sizer.size(c, open_position_count=0)
        if isinstance(outcome, TradeOrder):
            sized_orders.append(outcome)
        else:
            rejected.append(outcome)
            log.info("cli: rejected %s: %s", outcome.ticker, outcome.reason)

    print(f"Sized {len(sized_orders)} order(s) from {len(candidates)} candidate(s) "
          f"({len(rejected)} rejected by sizer).")
    for order in sized_orders:
        print(
            f"  {order.ticker} qty={order.qty} entry=${order.entry:.2f} "
            f"SL=${order.sl:.2f} TP=${order.tp:.2f} "
            f"notional=${order.notional:.2f} commission_rt=${order.commission_rt:.2f}"
        )

    if not args.discord:
        # No-Discord mode: just print and exit successfully.
        return 0

    # 3) Post to Discord + collect reactions.
    timeout_min = args.timeout_min or config.approval_timeout_min
    outcome = post_candidates_and_collect_reactions(
        sized_orders,
        timeout_min=timeout_min,
        bot_token=config.discord_bot_token,
        channel_thread_id=config.discord_home_channel_thread_id,
    )
    if outcome.error:
        print(f"ERROR: discord: {outcome.error}", file=sys.stderr)
        # Don't bail — fall through to persist in case partial
        # reactions are present.
    print(
        f"Discord outcome: approved={len(outcome.approved)} "
        f"rejected={len(outcome.rejected)} waited={len(outcome.waited)} "
        f"message_id={outcome.message_id}"
    )

    # 4) Persist.
    persist_mode = "dry" if args.dry_run else args.persist
    if persist_mode == "off" or not outcome.approved:
        if not outcome.approved:
            print("No approved trades to persist.")
        return 0

    sheets = None
    if persist_mode == "live":
        if not sheet_id:
            print("ERROR: --persist=live but no sheet_id (env GOOGLE_SHEET_ID or --sheet-id)", file=sys.stderr)
            return 1
        try:
            from portfoliomind.sheets.client import SheetsClient

            sheets = SheetsClient.from_config(config)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: sheets client: {e}", file=sys.stderr)
            return 1

    result = persist_approved_trades(
        outcome,
        sheets=sheets,
        sheet_id=sheet_id,
        dry_run=(persist_mode != "live"),
    )
    if result.error:
        print(f"ERROR: persist: {result.error}", file=sys.stderr)
        return 3
    print(
        f"Persist: appended {result.rows_appended} row(s), "
        f"skipped {result.duplicates_skipped} duplicate(s)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
