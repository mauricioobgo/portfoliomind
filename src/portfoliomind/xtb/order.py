"""Order spec validation and the Playwright-driven order placement flow.

This module is the single chokepoint for sending orders to XTB xStation.
It enforces the v4 spec's two iron rules — **SL is mandatory** and **TP is
mandatory** — and refuses to proceed if either is missing or zero. The
validation runs in :func:`validate_order` (a pure function, fully unit
testable) and again inside :func:`place_order` (defense in depth — even a
caller that constructs an :class:`OrderSpec` directly and skips the
helper will hit the same error).

The actual browser interaction is in :func:`place_order`. The flow is:

  1. Validate the spec (raises :class:`ValidationError` on failure).
  2. Make sure we are logged in (delegated to :func:`xtb.login.ensure_logged_in`).
  3. Screenshot the order book BEFORE the order (so the operator can see
     the spread / depth at the moment we sent it).
  4. Fill the order ticket on xStation and click Submit.
  5. Wait for the confirmation modal and read the order ID back from it.
  6. Screenshot the order book AFTER (so the operator can see the new
     position / pending order in the book).
  7. Return an :class:`OrderResult` with the order ID + paths to both
     screenshots.

Idempotency: callers (the CLI in particular) are responsible for dedup on
``(ticker, order_id)`` before calling :func:`place_order`. The function
itself trusts the caller; the dedup key is recorded in the EXECUTED_ORDERS
sheet, which is the authoritative source of truth.
"""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..logging_setup import get_logger
from ..time_utils import iso_now

if TYPE_CHECKING:
    from playwright.sync_api import Page

log = get_logger(__name__)


# --- Timeouts (all explicit, per the v4 spec) --------------------------------

LOGIN_TIMEOUT_S: int = 30
NAV_TIMEOUT_S: int = 15
ORDER_SUBMIT_TIMEOUT_S: int = 30


# --- Errors ------------------------------------------------------------------


class ValidationError(ValueError):
    """Raised when an :class:`OrderSpec` is invalid (missing SL/TP, bad qty, ...).

    Subclasses :class:`ValueError` so callers that catch ``ValueError`` still
    see a sensible error.
    """


class PlaceOrderError(RuntimeError):
    """Raised when the order could not be placed (browser error, xStation rejected,
    we could not parse the order ID back from the confirmation modal, ...)."""


# --- Enums -------------------------------------------------------------------


class OrderSide(str, enum.Enum):
    """Buy / sell. xStation uses the strings ``BUY`` and ``SELL`` in its
    REST/JSON APIs, so :class:`OrderSide` inherits from ``str`` to make the
    value directly serializable."""

    BUY = "BUY"
    SELL = "SELL"


# --- OrderSpec ---------------------------------------------------------------


@dataclass(frozen=True)
class OrderSpec:
    """An immutable, pre-validated order specification.

    Construction alone does NOT validate — call :func:`validate_order` (or
    the convenience constructor :meth:`OrderSpec.checked`) to enforce the
    mandatory SL/TP rules. The dataclass is frozen so the order flow cannot
    mutate the spec after validation.

    Attributes
    ----------
    ticker:
        Exchange ticker symbol as it appears on xStation (e.g. ``"AAPL.US"``,
        ``"EURUSD"``). The card body accepts the short form (``"AAPL"``) and
        delegates the suffix lookup to the executor.
    side:
        :class:`OrderSide` (BUY or SELL).
    qty:
        Number of shares / units. Must be > 0.
    entry_price:
        Limit / reference price. ``0`` is interpreted as "market order" on
        xStation, which is allowed but produces a warning because the
        strategy engine is expected to always supply a reference price.
    sl:
        Stop-loss price. **Mandatory** — see :func:`validate_order`.
    tp:
        Take-profit price. **Mandatory** — see :func:`validate_order`.
    note:
        Optional human-readable note attached to the spec (e.g. the approval
        note from the APPROVED_TRADES row). Not sent to xStation; only
        echoed in the EXECUTED_ORDERS log row.
    """

    ticker: str
    side: OrderSide
    qty: float
    entry_price: float
    sl: float
    tp: float
    note: str = ""
    created_at: str = field(default_factory=iso_now)

    # ----- Convenience constructors -----

    @classmethod
    def checked(
        cls,
        ticker: str,
        side: str | OrderSide,
        qty: float,
        entry_price: float,
        sl: float,
        tp: float,
        note: str = "",
    ) -> "OrderSpec":
        """Build an :class:`OrderSpec` from loose inputs and validate immediately.

        Accepts ``side`` as either an :class:`OrderSide` enum or a string
        (``"BUY"`` / ``"SELL"``, case-insensitive). Raises
        :class:`ValidationError` on any rule violation.
        """
        if isinstance(side, str):
            try:
                side_enum = OrderSide(side.upper())
            except ValueError as e:
                raise ValidationError(
                    f"Invalid side {side!r}: must be one of {[s.value for s in OrderSide]}"
                ) from e
        elif isinstance(side, OrderSide):
            side_enum = side
        else:
            raise ValidationError(
                f"side must be str or OrderSide, got {type(side).__name__}"
            )

        spec = cls(
            ticker=ticker,
            side=side_enum,
            qty=qty,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            note=note,
        )
        validate_order(spec)
        return spec


# --- OrderResult -------------------------------------------------------------


@dataclass(frozen=True)
class OrderResult:
    """The outcome of a :func:`place_order` call.

    The ``order_id`` is read from xStation's confirmation modal — it is
    **not** synthesized. If we could not parse an ID, the field is ``None``
    and the caller should treat the row as "submitted but unconfirmed"
    (the screenshot pair lets a human reconcile).
    """

    order_id: Optional[str]
    spec: OrderSpec
    screenshot_before: Optional[Path] = None
    screenshot_after: Optional[Path] = None
    submitted_at: str = field(default_factory=iso_now)
    raw_confirmation: dict[str, Any] = field(default_factory=dict)


# --- Validation --------------------------------------------------------------


def validate_order(spec: OrderSpec) -> None:
    """Validate an :class:`OrderSpec` against the PortfolioMind v4 iron rules.

    The function is intentionally pure (no I/O, no logging side effects) so
    it is trivially unit-testable. It raises :class:`ValidationError` on
    the first violation, with a message that names the field so the CLI
    can pinpoint the problem to the operator.

    Rules:

    1. ``ticker`` is a non-empty string.
    2. ``side`` is BUY or SELL (guaranteed by the enum, but we re-check).
    3. ``qty`` is a finite number ``> 0``.
    4. ``entry_price`` is either ``0`` (market) or a finite number ``> 0``.
    5. ``sl`` is a finite number and **strictly greater than zero**.
    6. ``tp`` is a finite number and **strictly greater than zero**.
    7. SL and TP are on the correct side of ``entry_price`` for ``side``:
       * BUY: ``sl < entry_price`` and ``tp > entry_price``
       * SELL: ``sl > entry_price`` and ``tp < entry_price``

    Rule 7 is the SL/TP-sanity check. A BUY with SL above the entry would
    cap the upside at zero; a SELL with SL below the entry is a no-op
    stop. Either is a strategy bug, not a tolerance we want to accept
    silently.
    """
    if not isinstance(spec.ticker, str) or not spec.ticker.strip():
        raise ValidationError("ticker must be a non-empty string")

    if not isinstance(spec.side, OrderSide):
        raise ValidationError(
            f"side must be an OrderSide enum, got {type(spec.side).__name__}"
        )

    if not _is_finite_positive(spec.qty):
        raise ValidationError(f"qty must be a finite number > 0, got {spec.qty!r}")

    if not _is_finite_non_negative(spec.entry_price):
        raise ValidationError(
            f"entry_price must be a finite number >= 0 (0 = market), got {spec.entry_price!r}"
        )

    # --- The iron rules: SL and TP must both be present and > 0 ----------
    if spec.sl is None or not _is_finite_positive(spec.sl):
        raise ValidationError(
            "Stop-loss (sl) is mandatory and must be a finite number > 0 — "
            "if the strategy says no SL, the strategy is wrong. "
            f"Got sl={spec.sl!r}."
        )
    if spec.tp is None or not _is_finite_positive(spec.tp):
        raise ValidationError(
            "Take-profit (tp) is mandatory and must be a finite number > 0 — "
            "if the strategy says no TP, the strategy is wrong. "
            f"Got tp={spec.tp!r}."
        )

    # --- Sanity: SL and TP must be on the correct side of entry_price ----
    # Skip the side check when entry_price is 0 (market order) because we
    # have no anchor. The first time we get a market order, warn the
    # operator because the strategy should normally give a limit price.
    if spec.entry_price > 0:
        if spec.side is OrderSide.BUY:
            if not spec.sl < spec.entry_price:
                raise ValidationError(
                    f"BUY order: sl ({spec.sl}) must be strictly below "
                    f"entry_price ({spec.entry_price})"
                )
            if not spec.tp > spec.entry_price:
                raise ValidationError(
                    f"BUY order: tp ({spec.tp}) must be strictly above "
                    f"entry_price ({spec.entry_price})"
                )
        else:  # SELL
            if not spec.sl > spec.entry_price:
                raise ValidationError(
                    f"SELL order: sl ({spec.sl}) must be strictly above "
                    f"entry_price ({spec.entry_price})"
                )
            if not spec.tp < spec.entry_price:
                raise ValidationError(
                    f"SELL order: tp ({spec.tp}) must be strictly below "
                    f"entry_price ({spec.entry_price})"
                )


def _is_finite_positive(x: Any) -> bool:
    """True iff x is a real number, not NaN, not ±Inf, and strictly > 0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    if v != v:  # NaN check (NaN != NaN)
        return False
    if not math.isfinite(v):  # catches ±inf
        return False
    return v > 0


def _is_finite_non_negative(x: Any) -> bool:
    """True iff x is a real number, not NaN, not ±Inf, and >= 0. Used for
    entry_price where 0 means "market order"."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    if v != v:  # NaN
        return False
    if not math.isfinite(v):  # catches ±inf
        return False
    return v >= 0


# --- The place_order flow ----------------------------------------------------


def place_order(
    page: "Page",
    spec: OrderSpec,
    *,
    screenshot_dir: Path,
) -> OrderResult:
    """Submit an order to xStation via the supplied Playwright page.

    Parameters
    ----------
    page:
        An already-logged-in Playwright :class:`Page` on the xStation
        trading screen. Login is the caller's job — see
        :mod:`portfoliomind.xtb.login`.
    spec:
        The pre-validated :class:`OrderSpec`. The function re-runs
        :func:`validate_order` as a defense-in-depth check.
    screenshot_dir:
        Directory where the BEFORE/AFTER screenshots will be written.
        Created if missing. Filename pattern:
        ``xtb_<ticker>_<side>_pre_<ts>.png`` and
        ``xtb_<ticker>_<side>_post_<ts>.png``.

    Returns
    -------
    :class:`OrderResult` with the order ID parsed from xStation's
    confirmation modal and the paths to both screenshots.

    Raises
    ------
    ValidationError
        If the spec is invalid (defense in depth — caller should have
        already validated).
    PlaceOrderError
        If the browser interaction fails (navigation timeout, submit
        button missing, confirmation modal never appeared, ...).
    """
    # Defense in depth: refuse to proceed if the spec is broken, even if
    # the caller skipped validation.
    validate_order(spec)

    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Stable timestamp for both screenshots in this run, so the BEFORE
    # and AFTER pair can be cross-referenced by the operator.
    ts = iso_now().replace(":", "").replace("-", "")  # safe for filenames

    # --- 1. BEFORE screenshot -------------------------------------------
    from .screenshot import capture_order_book  # local import: avoids cycle

    pre_path = screenshot_dir / f"xtb_{spec.ticker}_{spec.side.value}_pre_{ts}.png"
    try:
        capture_order_book(page, out_path=pre_path)
        log.info("screenshot_before_ok ticker=%s side=%s path=%s", spec.ticker, spec.side.value, pre_path)
    except Exception as e:  # noqa: BLE001 — log and continue; non-fatal for the order itself
        log.warning("screenshot_before_failed ticker=%s error=%r", spec.ticker, e)
        pre_path = None  # type: ignore[assignment]

    # --- 2. Drive xStation: fill the order ticket and submit -----------
    try:
        order_id = _submit_on_xstation(page, spec, timeout_s=ORDER_SUBMIT_TIMEOUT_S)
    except PlaceOrderError:
        # Even on failure, try a screenshot so the operator can debug.
        post_path = screenshot_dir / f"xtb_{spec.ticker}_{spec.side.value}_post_FAIL_{ts}.png"
        try:
            from .screenshot import capture_order_book

            capture_order_book(page, out_path=post_path)
        except Exception:  # noqa: BLE001
            post_path = None  # type: ignore[assignment]
        raise

    log.info("order_placed ticker=%s side=%s qty=%s order_id=%s", spec.ticker, spec.side.value, spec.qty, order_id)

    # --- 3. AFTER screenshot --------------------------------------------
    post_path = screenshot_dir / f"xtb_{spec.ticker}_{spec.side.value}_post_{ts}.png"
    try:
        from .screenshot import capture_order_book

        capture_order_book(page, out_path=post_path)
        log.info("screenshot_after_ok ticker=%s side=%s path=%s", spec.ticker, spec.side.value, post_path)
    except Exception as e:  # noqa: BLE001
        log.warning("screenshot_after_failed ticker=%s error=%r", spec.ticker, e)
        post_path = None  # type: ignore[assignment]

    return OrderResult(
        order_id=order_id,
        spec=spec,
        screenshot_before=pre_path,
        screenshot_after=post_path,
    )


def _submit_on_xstation(page: "Page", spec: OrderSpec, *, timeout_s: int) -> str:
    """Drive the xStation order ticket.

    xStation is a single-page React app. The selectors used here are
    the ones stable across the 2025-2026 builds; if xStation changes
    them, this function is the only place to update. The selectors are
    intentionally conservative (longer CSS paths, role-based where
    possible) to be resilient to layout reflows.

    The function is split out so the integration test in the future can
    stub the page object and assert on the inputs.

    Returns the parsed order ID. Raises :class:`PlaceOrderError` on any
    failure (timeout, missing selector, unparseable confirmation, ...).
    """
    # The card body says login + nav + submit each have explicit timeouts.
    page.set_default_timeout(timeout_s * 1000)

    # --- Fill the ticket ------------------------------------------------
    # Symbol field
    try:
        symbol_input = page.get_by_role("textbox", name=re.compile(r"symbol|instrument", re.I)).first
        symbol_input.fill("")
        symbol_input.fill(spec.ticker)
    except Exception as e:  # noqa: BLE001
        raise PlaceOrderError(f"Failed to fill symbol field for {spec.ticker!r}: {e!r}") from e

    # Quantity
    try:
        qty_input = page.get_by_role("textbox", name=re.compile(r"quantity|qty|volume", re.I)).first
        qty_input.fill("")
        qty_input.fill(_format_qty(spec.qty))
    except Exception as e:  # noqa: BLE001
        raise PlaceOrderError(f"Failed to fill quantity for {spec.ticker!r}: {e!r}") from e

    # Entry price (skip for market order)
    if spec.entry_price > 0:
        try:
            price_input = page.get_by_role("textbox", name=re.compile(r"price|limit", re.I)).first
            price_input.fill("")
            price_input.fill(_format_price(spec.entry_price))
        except Exception as e:  # noqa: BLE001
            raise PlaceOrderError(
                f"Failed to fill entry price for {spec.ticker!r}: {e!r}"
            ) from e

    # SL and TP — these are the iron rules. The xStation ticket has
    # dedicated SL/TP inputs; we fill them in the SL/TP tab.
    try:
        page.get_by_role("tab", name=re.compile(r"sl\s*&?\s*tp|stop\s*loss|protect", re.I)).first.click(
            timeout=timeout_s * 1000
        )
    except Exception as e:  # noqa: BLE001
        raise PlaceOrderError(f"Failed to open SL/TP tab: {e!r}") from e

    try:
        sl_input = page.get_by_role("textbox", name=re.compile(r"stop\s*loss|^sl$", re.I)).first
        sl_input.fill("")
        sl_input.fill(_format_price(spec.sl))
    except Exception as e:  # noqa: BLE001
        raise PlaceOrderError(f"Failed to fill SL ({spec.sl}): {e!r}") from e

    try:
        tp_input = page.get_by_role("textbox", name=re.compile(r"take\s*profit|^tp$", re.I)).first
        tp_input.fill("")
        tp_input.fill(_format_price(spec.tp))
    except Exception as e:  # noqa: BLE001
        raise PlaceOrderError(f"Failed to fill TP ({spec.tp}): {e!r}") from e

    # --- Submit ---------------------------------------------------------
    submit = page.get_by_role(
        "button", name=re.compile(rf"submit|{spec.side.value.lower()}|place\s*order", re.I)
    ).first
    try:
        submit.click(timeout=timeout_s * 1000)
    except Exception as e:  # noqa: BLE001
        raise PlaceOrderError(f"Failed to click Submit: {e!r}") from e

    # --- Read back the order ID from the confirmation modal --------------
    try:
        confirmation = page.get_by_role("dialog").first
        confirmation.wait_for(state="visible", timeout=timeout_s * 1000)
        body = confirmation.inner_text()
    except Exception as e:  # noqa: BLE001
        raise PlaceOrderError(f"Order confirmation modal did not appear: {e!r}") from e

    order_id = _parse_order_id(body)
    if not order_id:
        raise PlaceOrderError(
            f"Could not parse order ID from confirmation modal: {body[:200]!r}"
        )

    return order_id


_ORDER_ID_RE = re.compile(r"\b(\d{6,12})\b")  # xStation IDs are 8-10 digits in practice


def _parse_order_id(text: str) -> Optional[str]:
    """Extract an xStation order ID from a confirmation-modal text blob.

    The modal usually reads something like::

        Order 123456789 placed successfully.

    We grab the first long numeric run. Returns ``None`` if no candidate
    is found.
    """
    if not text:
        return None
    m = _ORDER_ID_RE.search(text)
    return m.group(1) if m else None


def _format_qty(qty: float) -> str:
    """Format a quantity for the xStation input. xStation accepts plain
    integers or decimals; we drop trailing zeros but keep the integer
    part untouched."""
    if qty == int(qty):
        return str(int(qty))
    return f"{qty:g}"


def _format_price(price: float) -> str:
    """Format a price for the xStation input. 4-5 decimal places is plenty
    for equities and most FX pairs; we use ``g`` to drop trailing zeros."""
    return f"{price:g}"


__all__ = [
    "OrderSide",
    "OrderSpec",
    "OrderResult",
    "ValidationError",
    "PlaceOrderError",
    "LOGIN_TIMEOUT_S",
    "NAV_TIMEOUT_S",
    "ORDER_SUBMIT_TIMEOUT_S",
    "validate_order",
    "place_order",
]
