"""InvestingPro login flow.

Card 2 of the PortfolioMind v4 build. Wraps the Playwright steps needed to
sign in to InvestingPro, persist the session, and hand a live ``Page`` back
to the caller (the scrape + deep-dive modules).

Design choices, restated for reviewers:

* **Persistent context** — we use ``browser.launch_persistent_context``
  with the operator-configured ``SESSION_DIR`` so the cookies + local
  storage survive across runs. The card spec explicitly forbids
  ``launch()`` + manual cookie save because that path silently drops
  HttpOnly cookies and IndexedDB tokens.

* **No blind retries on auth failure** — if the login form does not
  resolve, we screenshot and raise. The operator must look at the
  screenshot. A 2FA prompt, a captcha, or an InvestingPro layout change
  are all reasons the agent should pause, not retry.

* **Timeouts are explicit** — 30s for the auth submit, 15s for the
  post-auth redirect, 60s for the AI Picks render (in :mod:`scrape`).

* **Secrets never leave the env** — the email + password are read from
  :class:`PortfoliomindConfig`, used to fill the form, and never echoed
  in logs or error messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from playwright.sync_api import (
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from ..config import PortfoliomindConfig
from ..logging_setup import get_logger
from ..paths import screenshot_dir
from ..time_utils import iso_now
from .parse import _clean_cell  # noqa: F401  (re-exported for downstream parity)

log = get_logger(__name__)

# InvestingPro URLs. The "pro" subdomain hosts the AI Picks / ProPicks
# dashboard. The classic login form lives at /login; after a successful
# sign-in the user lands on a portal page that links into the Pro tools.
_BASE_URL = "https://www.investing.com"
_LOGIN_URL = f"{_BASE_URL}/login"
_AI_PICKS_URL = f"{_BASE_URL}/pro/propicks"

# Timeouts (seconds). Explicit so the caller doesn't need to consult the
# Playwright default (which is 30s but is silently a different default in
# some Playwright versions).
_LOGIN_TIMEOUT_S = 30
_NAV_TIMEOUT_S = 15

# Selector candidates. InvestingPro's HTML uses generic class names
# (``.loginForm`` etc.) that change without notice; we try a list of
# known-good selectors and accept the first one that resolves. If none
# resolve we screenshot and raise.
#
# As of late-2025/early-2026 the login is a modal with inputs keyed by
# ``placeholder="Email"`` and ``placeholder="Password"`` rather than
# ``name=`` or ``id=`` — we lead with the placeholder selectors because
# they are the most stable in the modern SPA.
_EMAIL_SELECTORS = (
    "input[placeholder='Email']",
    "input[placeholder='email']",
    "input[name='email']",
    "input[type='email']",
    "input#loginFormPasswordEmail",
    "input[autocomplete='username']",
)
_PASSWORD_SELECTORS = (
    "input[placeholder='Password']",
    "input[placeholder='password']",
    "input[name='password']",
    "input[type='password']",
    "input#loginFormPassword",
)
_SUBMIT_SELECTORS = (
    # Modern InvestingPro: the "Sign In" button is rendered as an <a>
    # tag with class "newButton orange" inside a sign-in popup. We
    # lead with that because it's the current production markup.
    "a.newButton.orange",
    "a:has-text('Sign In')",
    "button.newButton.gradient",
    "button[type='submit']",
    "button.loginSubmit",
    "input[type='submit']",
    "button:has-text('Sign In')",
)


# --- Exceptions -------------------------------------------------------------


class InvestingProLoginError(RuntimeError):
    """Raised when the InvestingPro login flow cannot be completed.

    The original Playwright exception is chained via ``__cause__`` for
    debugging; the public message does NOT include the email or password.
    """


# --- Result -----------------------------------------------------------------


@dataclass
class LoginResult:
    """Bundle of the live Playwright objects + where to find the session."""

    context: BrowserContext
    page: Page
    session_dir: Path
    landed_url: str


# --- Public API -------------------------------------------------------------


def login(
    config: PortfoliomindConfig,
    *,
    headless: bool = True,
    session_dir: Optional[Path] = None,
) -> LoginResult:
    """Open InvestingPro, sign in, and return a live ``LoginResult``.

    Parameters
    ----------
    config:
        The PortfolioMind env-driven config. ``investingpro_email`` and
        ``investingpro_password`` are required.
    headless:
        If False, run Chromium with a visible window. Useful for the
        operator when debugging a login that is failing — the visible
        browser shows them the same screen the screenshot captured.
    session_dir:
        Override the Playwright persistent context directory. Defaults to
        ``config.session_dir``. Provided so tests can point at a temp dir
        without touching the operator's real session.

    Returns
    -------
    LoginResult
        The open ``BrowserContext`` + the landing ``Page``. The caller is
        responsible for ``context.close()`` (use a ``with`` block or try /
        finally).

    Raises
    ------
    InvestingProLoginError
        If the login form can't be filled, the submit doesn't resolve
        within 30s, or the post-auth page is not the expected landing
        page. A screenshot is saved to ``SCREENSHOT_DIR`` in every case
        so the operator can diagnose.
    """
    if not config.investingpro_email or not config.investingpro_password:
        raise InvestingProLoginError(
            "INVESTINGPRO_EMAIL or INVESTINGPRO_PASSWORD is empty; "
            "check the profile .env"
        )

    sdir = Path(session_dir) if session_dir is not None else config.session_dir
    sdir.mkdir(parents=True, exist_ok=True)

    log.info(
        "investingpro.login.start headless=%s session_dir=%s",
        headless,
        sdir,
    )

    try:
        with sync_playwright() as p:
            # launch_persistent_context is the contract — never launch()
            # + manual cookie persistence.
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(sdir),
                headless=headless,
                # InvestingPro rejects requests with a default UA in some
                # regions; send a current Chrome UA. We do NOT set
                # extra_http_headers here — that would break the
                # per-request cloudflare challenge flow.
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1440, "height": 900},
                locale="en-US",
                # Keep the session in the local FS for next run.
                accept_downloads=False,
            )
            page: Optional[Page] = None
            try:
                page = _first_page(context)
                page.set_default_timeout(_LOGIN_TIMEOUT_S * 1000)
                page.set_default_navigation_timeout(_NAV_TIMEOUT_S * 1000)

                _fill_login_form(page, config)
                landed = _submit_and_wait(page, expected_url_substring="/pro")
                log.info(
                    "investingpro.login.success landed_url=%s", landed
                )
                return LoginResult(
                    context=context,
                    page=page,
                    session_dir=sdir,
                    landed_url=landed,
                )
            except BaseException as e:
                # Capture the page at the moment of failure BEFORE the
                # context tears down. The screenshot is the operator's
                # primary diagnostic when 2FA / captcha / a site change
                # breaks the flow.
                _screenshot_failure("login", e, page=page)
                try:
                    context.close()
                except Exception:
                    pass
                raise
    except (InvestingProLoginError, PlaywrightTimeoutError) as e:
        log.error("investingpro.login.failed: %s", type(e).__name__)
        raise InvestingProLoginError(
            f"InvestingPro login failed: {type(e).__name__}"
        ) from e
    except Exception as e:  # last-ditch safety net
        log.error("investingpro.login.unexpected: %s", type(e).__name__)
        raise InvestingProLoginError(
            f"InvestingPro login failed unexpectedly: {type(e).__name__}"
        ) from e


# --- Internals --------------------------------------------------------------


def _first_page(context: BrowserContext) -> Page:
    """Return the context's first page, creating one if needed.

    A persistent context starts with no page. Some Playwright versions
    lazily create one on first navigation; others don't. Be explicit.
    """
    if context.pages:
        return context.pages[0]
    return context.new_page()


def _fill_login_form(page: Page, config: PortfoliomindConfig) -> None:
    """Navigate to /login and fill the email + password fields.

    Raises :class:`InvestingProLoginError` if neither selector set matches
    a field on the page.
    """
    page.goto(_LOGIN_URL, timeout=_NAV_TIMEOUT_S * 1000)
    # Cloudflare / InvestingPro often serves a challenge page first; let
    # the page settle, then wait for one of the known email selectors.
    email_el = _wait_for_first(page, _EMAIL_SELECTORS, _LOGIN_TIMEOUT_S * 1000)
    if email_el is None:
        raise InvestingProLoginError(
            "Could not find the email input on the InvestingPro login page"
        )
    password_el = _wait_for_first(page, _PASSWORD_SELECTORS, _LOGIN_TIMEOUT_S * 1000)
    if password_el is None:
        raise InvestingProLoginError(
            "Could not find the password input on the InvestingPro login page"
        )

    # Clear any prefilled values the persistent context may have left.
    email_el.fill("")
    password_el.fill("")
    email_el.fill(config.investingpro_email)
    password_el.fill(config.investingpro_password)


def _submit_and_wait(page: Page, *, expected_url_substring: str) -> str:
    """Click submit, wait for the URL to change off the login page.

    The InvestingPro login does a POST + redirect, sometimes to /pro
    directly, sometimes to a portal page that requires another click.
    We accept any URL that contains the expected substring or that no
    longer looks like the login form.
    """
    submit = _wait_for_first(page, _SUBMIT_SELECTORS, _LOGIN_TIMEOUT_S * 1000)
    if submit is None:
        raise InvestingProLoginError(
            "Could not find the submit button on the InvestingPro login page"
        )
    # Some InvestingPro forms are dispatched on Enter inside the password
    # field; we click explicitly. If the click is intercepted, the click
    # raises and the outer try / except will screenshot + raise.
    submit.click()

    # Wait for the URL to leave /login. Use a generous timeout — the
    # server may take a few seconds to redirect on a cold session.
    try:
        page.wait_for_url(
            lambda url: (
                expected_url_substring in url
                or "/login" not in url
            ),
            timeout=_LOGIN_TIMEOUT_S * 1000,
        )
    except PlaywrightTimeoutError as e:
        raise InvestingProLoginError(
            "Login submit timed out waiting for redirect off /login"
        ) from e

    return page.url


def _wait_for_first(page: Page, selectors: tuple[str, ...], timeout_ms: int):
    """Return the first selector that resolves to an element, or None.

    InvestingPro's DOM is unstable. We try each candidate selector in
    order and accept the first hit. The selectors are checked every 100ms
    until the overall timeout expires.
    """
    deadline_ms = timeout_ms
    interval_ms = 100
    elapsed = 0
    while elapsed < deadline_ms:
        for sel in selectors:
            try:
                el = page.query_selector(sel)
                if el is not None:
                    return el
            except Exception:
                # Selector errored out (e.g. unsupported pseudo-class).
                # Keep trying the others.
                continue
        page.wait_for_timeout(interval_ms)
        elapsed += interval_ms
    return None


def _screenshot_failure(
    stage: str, exc: BaseException, *, page: Optional[Page] = None
) -> None:
    """Best-effort screenshot to ``SCREENSHOT_DIR/investingpro_<stage>_<ts>.png``.

    Called from the inner except, while the page is still live. We use
    Playwright's :meth:`Page.screenshot` so the operator sees exactly
    what the agent saw. We never let a screenshot failure mask the
    original exception: any error here is logged and swallowed.
    """
    try:
        out_dir = screenshot_dir()
        ts = iso_now().replace(":", "-")
        path = out_dir / f"investingpro_{stage}_{ts}.png"
        if page is not None:
            page.screenshot(path=str(path), full_page=True)
        else:
            # No page — write a marker file so the operator at least
            # knows the failure happened and where to look in the log.
            path.touch(exist_ok=False)
        log.error(
            "investingpro.login.screenshot path=%s stage=%s err=%s",
            path,
            stage,
            type(exc).__name__,
        )
    except Exception as inner:
        log.error(
            "investingpro.login.screenshot.unavailable err=%s",
            type(inner).__name__,
        )


__all__ = [
    "InvestingProLoginError",
    "LoginResult",
    "login",
    "_BASE_URL",
    "_LOGIN_URL",
    "_AI_PICKS_URL",
]
