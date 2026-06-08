"""XTB xStation Playwright login.

We use a **persistent browser context** so the XTB session cookie survives
across runs. The context directory is :func:`paths.session_dir()` + ``/xtb``;
the same XTB account can run once a day without re-typing the password.

Public surface:

* :func:`build_context` — open a persistent context, headless or not.
* :func:`login` — fill the XTB login form and wait for the trading screen.
* :func:`ensure_logged_in` — convenience wrapper: if the cookie is still
  valid, skip the form; otherwise re-login. Used by the order flow.

The selectors here are conservative and role-based where possible. xStation
is a React SPA and they reflow periodically; the resilient strategy is
to use ``page.get_by_role`` (matches the ARIA tree) and to take a screenshot
on every failure so the operator can see what the page looked like.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..config import PortfoliomindConfig
from ..logging_setup import get_logger
from ..paths import session_dir
from .screenshot import capture_login_failure_screenshot

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page

log = get_logger(__name__)


# Timeouts — explicit, per the v4 spec.
LOGIN_TIMEOUT_S: int = 30
NAV_TIMEOUT_S: int = 15

# The xStation live URL. xStation also offers a demo URL
# (``https://xstation5-demo.xtb.com``); the operator can switch by setting
# ``XTB_URL`` in the env. We default to live because that's where the
# real money is and the agent is forbidden from trading without operator
# approval anyway.
DEFAULT_XTB_URL: str = "https://xstation5.xtb.com"


@dataclass(frozen=True)
class XTBSessionPaths:
    """Filesystem layout for the persistent browser context.

    The Playwright persistent context is rooted at ``context_dir`` and
    uses a single Chromium profile there. We keep login / order
    screenshots in separate subdirectories so the operator can eyeball
    a day's activity at a glance.
    """

    context_dir: Path
    login_screenshots_dir: Path
    order_screenshots_dir: Path

    @classmethod
    def from_config(cls, config: PortfoliomindConfig) -> "XTBSessionPaths":
        sdir = session_dir()
        cdir = sdir / "xtb"
        cdir.mkdir(parents=True, exist_ok=True)
        return cls(
            context_dir=cdir,
            login_screenshots_dir=cdir / "screenshots_login",
            order_screenshots_dir=cdir / "screenshots_orders",
        )


class LoginError(RuntimeError):
    """Raised when the login form cannot be submitted successfully."""


def build_context(
    paths: XTBSessionPaths,
    *,
    headless: bool = True,
) -> "BrowserContext":
    """Open a persistent Playwright browser context rooted at ``paths.context_dir``.

    Using a persistent context (vs. a fresh ``launch()`` + ``new_context()``)
    is what lets the XTB session cookie survive between runs. The
    Chromium user-data-dir is ``paths.context_dir`` itself.

    Parameters
    ----------
    paths:
        The :class:`XTBSessionPaths` for this run.
    headless:
        ``True`` (default) for cron / CI; ``False`` for operator-driven
        debug runs (the browser window pops up so the operator can see
        what the agent is doing).
    """
    # Local import: Playwright is a heavy import and we want CLI help /
    # --dry-run to start fast.
    from playwright.sync_api import sync_playwright

    paths.context_dir.mkdir(parents=True, exist_ok=True)
    pw = sync_playwright().start()
    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(paths.context_dir),
        headless=headless,
        # XTB uses a few popups and modals; a slightly larger viewport
        # avoids layout reflows that hide the order ticket.
        viewport={"width": 1440, "height": 900},
        # Operator-controlled runs (headless=False) get slow_mo so the
        # operator can follow what's happening. Headless runs do not.
        slow_mo=200 if not headless else 0,
    )
    log.info(
        "xtb_context_opened headless=%s dir=%s", headless, paths.context_dir
    )
    return context


def login(
    page: "Page",
    config: PortfoliomindConfig,
    *,
    base_url: str = DEFAULT_XTB_URL,
    timeout_s: int = LOGIN_TIMEOUT_S,
    failure_screenshot_dir: Optional[Path] = None,
) -> None:
    """Drive the XTB login form.

    Parameters
    ----------
    page:
        An already-open :class:`playwright.sync_api.Page` from a persistent
        context. We do NOT open the page here — the caller passes one in
        so the same page can be reused for the order flow.
    config:
        Agent config; ``config.xtb_user_id`` and ``config.xtb_password``
        are read but never logged.
    base_url:
        xStation URL; default is the live terminal. Set ``XTB_URL`` via
        the env or pass ``base_url="https://xstation5-demo.xtb.com"``
        for the demo account.
    timeout_s:
        How long to wait for the login form to appear + the trading
        screen to load. 30s is the v4 spec default.
    failure_screenshot_dir:
        If the login fails, we save a screenshot here so the operator
        can see what the page looked like. Defaults to
        :attr:`XTBSessionPaths.login_screenshots_dir`.
    """
    if not config.xtb_user_id or not config.xtb_password:
        # Critical: never echo the password. The user_id is the XTB
        # account number, which the operator already knows; we can name
        # which env var is missing without leaking the value.
        raise LoginError(
            "XTB credentials missing: set XTB_USER_ID and XTB_PASSWORD in env"
        )

    page.set_default_timeout(timeout_s * 1000)

    log.info("xtb_login_navigating url=%s", base_url)
    page.goto(base_url, wait_until="domcontentloaded")

    # --- Fill credentials ----------------------------------------------
    # XTB's form (verified against the live xStation 5 page in 2025-2026)
    # has:
    #   * ``<input name="xslogin" type="text">``     — user_id (email)
    #   * ``<input name="xspass" type="password">``  — password
    #   * ``<button class="xs-btn-ok-login">``       — submit (text "Login")
    #
    # We use ``name`` attribute selectors (the most stable identifier
    # Angular exposes) plus a class selector for the submit button. The
    # previous role-based approach was unreliable because the inputs
    # carry no ``aria-label`` and are not associated with their visible
    # labels via ``for=``.
    #
    # If the page is already authenticated (cookie still alive) the
    # login form is hidden; the next check handles that case.
    try:
        user_input = page.locator('input[name="xslogin"]').first
        user_input.wait_for(state="visible", timeout=timeout_s * 1000)
        user_input.fill(config.xtb_user_id)
        pass_input = page.locator('input[name="xspass"]').first
        pass_input.fill(config.xtb_password)
    except Exception as e:  # noqa: BLE001
        # The form might not be present if we're already logged in.
        # Verify by checking for a known authenticated-page element.
        if _is_authenticated(page):
            log.info("xtb_login_already_authenticated url=%s", base_url)
            return
        raise LoginError(f"Could not find XTB login form: {e!r}") from e

    # --- Submit ---------------------------------------------------------
    # The Login button is identified by its CSS class (``xs-btn-ok-login``)
    # which is stable across xStation releases. As a fallback, role-based
    # lookup on visible text "Login" is also tried.
    try:
        page.locator("button.xs-btn-ok-login").first.click(timeout=timeout_s * 1000)
    except Exception:  # noqa: BLE001
        try:
            page.get_by_role("button", name=re.compile(r"log\s*in|sign\s*in", re.I)).first.click(
                timeout=timeout_s * 1000
            )
        except Exception as e:  # noqa: BLE001
            raise LoginError(f"Could not click XTB login button: {e!r}") from e

    # --- Wait for the trading screen ------------------------------------
    # xStation's trading screen has an element with role "navigation" that
    # contains the market-watch panel; if we see it, we know we're in.
    try:
        page.get_by_role("navigation").first.wait_for(state="visible", timeout=timeout_s * 1000)
        log.info("xtb_login_succeeded url=%s", base_url)
    except Exception as e:  # noqa: BLE001
        # Take a screenshot of whatever the page actually shows — the
        # modal might say "Invalid credentials" or the captcha might be
        # blocking us. Never log the password.
        if failure_screenshot_dir is not None:
            try:
                capture_login_failure_screenshot(page, failure_screenshot_dir)
            except Exception:  # noqa: BLE001
                pass  # screenshot is best-effort
        raise LoginError(
            f"XTB login did not reach the trading screen within {timeout_s}s: {e!r}"
        ) from e


def _is_authenticated(page: "Page") -> bool:
    """Heuristic: return True if the current page already shows the trading
    screen (i.e. we're already logged in and the cookie is alive)."""
    try:
        page.get_by_role("navigation").first.wait_for(state="visible", timeout=2000)
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_logged_in(
    page: "Page",
    config: PortfoliomindConfig,
    *,
    base_url: str = DEFAULT_XTB_URL,
    timeout_s: int = LOGIN_TIMEOUT_S,
    failure_screenshot_dir: Optional[Path] = None,
) -> None:
    """No-op if the cookie is still alive; otherwise fill the form.

    This is the entry point for the order flow. It assumes the caller
    has already opened a persistent :class:`BrowserContext` (via
    :func:`build_context`) and a :class:`Page`.
    """
    # Fast path: we are already authenticated.
    if _is_authenticated(page):
        log.info("xtb_session_alive url=%s", base_url)
        return

    # Slow path: navigate (in case we're on a different URL) and log in.
    login(
        page,
        config,
        base_url=base_url,
        timeout_s=timeout_s,
        failure_screenshot_dir=failure_screenshot_dir,
    )


def teardown_context(context: "BrowserContext") -> None:
    """Close the persistent context cleanly. Safe to call on errors.

    The Playwright context holds a Chromium child process; if we don't
    close it, the process leaks until the agent process exits.
    """
    try:
        context.close()
    except Exception as e:  # noqa: BLE001
        log.warning("xtb_context_close_failed error=%r", e)


__all__ = [
    "LOGIN_TIMEOUT_S",
    "NAV_TIMEOUT_S",
    "DEFAULT_XTB_URL",
    "XTBSessionPaths",
    "LoginError",
    "build_context",
    "login",
    "ensure_logged_in",
    "teardown_context",
]
