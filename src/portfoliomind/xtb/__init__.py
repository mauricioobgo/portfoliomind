"""PortfolioMind XTB integration.

This subpackage wraps Playwright-driven interactions with the XTB xStation
web terminal. The design follows three iron rules that come straight from
the PortfolioMind v4 spec and the agent's SOUL:

1. **Stop-loss is mandatory.** No SL, no order. The validation is enforced
   at the lowest level (:func:`place_order` and :class:`OrderSpec`) so that
   no caller — script, test, future card — can bypass it.
2. **Take-profit is mandatory.** Same reasoning.
3. **Dry-run is the default.** The CLI script that drives :func:`place_order`
   refuses to touch the live terminal unless the operator has explicitly
   opted in via ``--confirm-each``.

The public surface that downstream cards (specifically card 4 — the
scheduler) depend on is:

* :class:`OrderSpec` — the validated, immutable order spec.
* :class:`OrderResult` — the result of a submission attempt.
* :func:`validate_order` — pure function; used by tests and the CLI.
* :func:`place_order` — the real thing; launches Playwright.

Browser-driven helpers live in :mod:`.login` and :mod:`.screenshot`; the
order flow in :mod:`.order` consumes them.
"""

from __future__ import annotations

from .order import (
    OrderResult,
    OrderSide,
    OrderSpec,
    PlaceOrderError,
    ValidationError,
    place_order,
    validate_order,
)

__all__ = [
    "OrderSide",
    "OrderSpec",
    "OrderResult",
    "ValidationError",
    "PlaceOrderError",
    "validate_order",
    "place_order",
]
