"""Screenshot helpers for the XTB order flow.

Two functions:

* :func:`capture_order_book` — full-viewport PNG of the trading screen,
  used as the BEFORE/AFTER for an order.
* :func:`capture_login_failure_screenshot` — best-effort PNG when the
  login form fails (invalid creds, captcha, network, ...).

Both functions are deliberately defensive: a screenshot is a debugging
artifact, never a precondition. If the page object is unusable (browser
crashed mid-run, etc.) we log and move on rather than raising — the
order flow has its own error handling.

Why two functions and not one?

The order flow wants PNGs in :attr:`XTBSessionPaths.order_screenshots_dir`,
the login flow wants them in :attr:`XTBSessionPaths.login_screenshots_dir`.
Combining them would mean a single ``dir`` parameter and the caller
forgetting to set it correctly. Splitting them is clearer.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging_setup import get_logger
from ..time_utils import iso_now

if TYPE_CHECKING:
    from playwright.sync_api import Page

log = get_logger(__name__)


def capture_order_book(page: "Page", *, out_path: Path) -> Path:
    """Save a full-page PNG of the current xStation trading screen.

    Parameters
    ----------
    page:
        The :class:`Page` showing the trading screen. The function
        assumes the page is at the right URL and has the order book
        visible; we don't navigate or scroll.
    out_path:
        Destination file path. The parent directory must already exist
        (caller's responsibility). The PNG is written atomically by
        Chromium's ``screenshot()`` call.

    Returns
    -------
    The same ``out_path``, for fluent use.

    Raises
    ------
    RuntimeError
        If the screenshot cannot be taken (browser closed, page crashed,
        etc.). The caller decides whether to swallow or re-raise.
    """
    if not out_path.parent.is_dir():
        raise FileNotFoundError(
            f"Screenshot directory does not exist: {out_path.parent}"
        )
    try:
        page.screenshot(path=str(out_path), full_page=False)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to capture order book screenshot to {out_path}: {e!r}"
        ) from e
    log.info("screenshot_saved path=%s", out_path)
    return out_path


def capture_login_failure_screenshot(
    page: "Page",
    dest_dir: Path,
) -> Path | None:
    """Best-effort screenshot of a failed login attempt.

    Filesystem layout::

        <dest_dir>/login_fail_<YYYYMMDDTHHMMSS>.png

    The function does NOT raise on failure — the caller is already in
    an error path. Returns the path on success, ``None`` on failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = iso_now().replace(":", "").replace("-", "")
    out_path = dest_dir / f"login_fail_{ts}.png"
    try:
        page.screenshot(path=str(out_path), full_page=True)
        log.warning("login_failure_screenshot_saved path=%s", out_path)
        return out_path
    except Exception as e:  # noqa: BLE001
        log.warning("login_failure_screenshot_failed error=%r", e)
        return None


def screenshot_basename_for(
    ticker: str, side: str, phase: str, timestamp: datetime | None = None
) -> str:
    """Compute the canonical screenshot filename for a given phase.

    Exposed so the order flow and the CLI agree on the same name format.
    The card body specifies: ``SCREENSHOT_DIR/xtb_<ticker>_<side>_<ts>.png``;
    we keep that and add a ``_pre`` / ``_post`` infix for BEFORE/AFTER
    pairing.

    Parameters
    ----------
    ticker:
        The order's ticker. Sanitized: anything other than alphanumerics,
        dots, hyphens, and underscores is replaced with an underscore.
    side:
        ``"BUY"`` or ``"SELL"``.
    phase:
        ``"pre"`` (BEFORE) or ``"post"`` (AFTER) or ``"post_FAIL"`` (failure
        capture).
    timestamp:
        Defaults to ``now()`` if not provided.
    """
    ts = (timestamp or datetime.now()).strftime("%Y%m%dT%H%M%S")
    safe_ticker = "".join(c if c.isalnum() or c in ".-_" else "_" for c in ticker)
    safe_side = "".join(c if c.isalnum() else "_" for c in side)
    safe_phase = "".join(c if c.isalnum() else "_" for c in phase)
    return f"xtb_{safe_ticker}_{safe_side}_{safe_phase}_{ts}.png"


__all__ = [
    "capture_order_book",
    "capture_login_failure_screenshot",
    "screenshot_basename_for",
]
