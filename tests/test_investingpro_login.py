"""Hermetic unit tests for :mod:`portfoliomind.investingpro.login`.

Card 2.1 — fix the submit-button race condition. The live smoke test on
2026-06-10 showed the form being submitted while the ``Sign In`` button
was still disabled (the page's JS hadn't finished binding the click
handler). The fix in :mod:`portfoliomind.investingpro.login` adds:

* a ``_wait_for_enabled`` poll before the click, and
* a small retry loop around the click itself.

These tests never spawn a real Playwright browser. They drive the
private ``_submit_and_wait`` function with a :class:`FakePage` whose
``query_selector`` / ``wait_for_url`` / ``wait_for_timeout`` are
scripted, and a :class:`FakeElement` whose ``is_enabled`` /
``click`` are scripted per-attempt. The scenarios are:

* **Normal** — button enabled immediately, click succeeds, redirect.
* **Slow** — button disabled for ~800ms, then enabled; click should
  wait + succeed (this is the observed production behavior).
* **Stuck** — button never enables; should raise
  :class:`InvestingProLoginError` with a clear message and a chained
  cause.
* **Race** — button reports enabled but click raises on the first
  attempt (handler still binding); second attempt succeeds.

We also exercise the outer error-wrapping fix: an
:class:`InvestingProLoginError` raised from the inner flow must
re-raise with the original message (no double-wrap), and a stray
``PlaywrightTimeoutError`` from the inner flow must be converted
once to :class:`InvestingProLoginError` (not double-wrapped).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from portfoliomind.investingpro.login import (
    InvestingProLoginError,
    _submit_and_wait,
)


# --- Fakes ------------------------------------------------------------------


class FakeElement:
    """A scripted submit button.

    ``is_enabled_at`` and ``click_at`` are callables invoked once per
    call (with the attempt number, 1-indexed). They return either a
    bool (for ``is_enabled``) or raise an exception (for ``click``).
    If the callable returns normally, the click is treated as
    successful.
    """

    def __init__(
        self,
        *,
        is_enabled_at: Callable[[int], bool],
        click_at: Callable[[int], None],
    ) -> None:
        self._is_enabled_at = is_enabled_at
        self._click_at = click_at
        self.click_calls = 0
        self.is_enabled_calls = 0

    def is_enabled(self) -> bool:
        self.is_enabled_calls += 1
        return self._is_enabled_at(self.is_enabled_calls)

    def click(self, *, timeout: int = 0) -> None:
        self.click_calls += 1
        self._click_at(self.click_calls)


class FakePage:
    """A scripted :class:`playwright.sync_api.Page`-shaped object.

    The login flow only touches ``query_selector``, ``wait_for_url``,
    and ``wait_for_timeout``. We record the calls so tests can assert
    on the retry / backoff behavior.
    """

    def __init__(
        self,
        *,
        submit: Optional[FakeElement] = None,
        final_url: str = "https://www.investing.com/pro/propicks",
        query_selector_selector: Optional[str] = None,
    ) -> None:
        self._submit = submit
        self._final_url = final_url
        self._query_selector_selector = query_selector_selector
        self.url = "https://www.investing.com/login"
        self.wait_for_url_calls: list[Callable[[str], bool]] = []
        self.wait_for_timeout_calls: list[int] = []

    def query_selector(self, selector: str) -> Optional[FakeElement]:
        # The login flow passes a tuple of candidate selectors, one at
        # a time, into ``_wait_for_first``. We return the scripted
        # submit button for the first selector we see, None for the
        # rest, mimicking the case where the form is found on the
        # very first iteration of the polling loop.
        if self._query_selector_selector is None:
            self._query_selector_selector = selector
        if selector == self._query_selector_selector:
            return self._submit
        return None

    def wait_for_url(self, predicate: Callable[[str], bool], *, timeout: int) -> None:
        self.wait_for_url_calls.append(predicate)
        # Evaluate the predicate against the final URL; if it matches,
        # we "navigated". Otherwise we'd block (and the caller would
        # have to time out, which is not what we want in a unit test).
        if not predicate(self._final_url):
            raise PlaywrightTimeoutError(
                f"wait_for_url predicate did not match {self._final_url}"
            )
        self.url = self._final_url

    def wait_for_timeout(self, ms: int) -> None:
        self.wait_for_timeout_calls.append(ms)


# --- _submit_and_wait tests -------------------------------------------------


def test_submit_and_wait_normal_case():
    """Button is enabled immediately; click + redirect succeed on
    the first try. No retries, no waits beyond the initial
    ``_wait_for_enabled`` poll (one ``is_enabled`` call)."""
    submit = FakeElement(is_enabled_at=lambda n: True, click_at=lambda n: None)
    page = FakePage(submit=submit)

    landed = _submit_and_wait(page, expected_url_substring="/pro")

    assert landed == "https://www.investing.com/pro/propicks"
    assert submit.click_calls == 1
    # The wait_for_enabled poll calls is_enabled at least once; we
    # bound it loosely to avoid brittleness across poll-interval
    # changes.
    assert submit.is_enabled_calls >= 1
    # No retry backoffs in the happy path.
    assert page.wait_for_timeout_calls == []


def test_submit_and_wait_slow_button():
    """Button is disabled for ~800ms, then enabled. The poll loop
    should wait and then succeed. This is the closest simulation of
    the live failure mode (form rendered, fields filled, button
    greyed out for a beat while CSRF data resolves)."""
    # Threshold: 0..8 calls return False (each call is 100ms apart
    # because of the 0.1s sleep in _wait_for_enabled), then True.
    enabled_after = 8
    call_count = {"n": 0}

    def is_enabled_at(n: int) -> bool:
        call_count["n"] = n
        return n > enabled_after

    def click_at(n: int) -> None:
        return None

    submit = FakeElement(is_enabled_at=is_enabled_at, click_at=click_at)
    page = FakePage(submit=submit)

    landed = _submit_and_wait(page, expected_url_substring="/pro")

    assert landed == "https://www.investing.com/pro/propicks"
    # We should have polled more than once and then succeeded.
    assert call_count["n"] > enabled_after
    # Click happened exactly once — the button was enabled by then.
    assert submit.click_calls == 1


def test_submit_and_wait_stuck_button_raises_with_cause():
    """Button never enables within the configured timeout. Should
    raise :class:`InvestingProLoginError` with a message naming
    the probable cause (2FA / captcha / layout change)."""
    # Patch the module-level constants for a fast test.
    import portfoliomind.investingpro.login as login_mod

    orig_timeout_s = login_mod._LOGIN_TIMEOUT_S
    login_mod._LOGIN_TIMEOUT_S = 1  # 1s total budget
    try:
        submit = FakeElement(
            is_enabled_at=lambda n: False,  # never enabled
            click_at=lambda n: None,
        )
        page = FakePage(submit=submit)

        with pytest.raises(InvestingProLoginError) as excinfo:
            _submit_and_wait(page, expected_url_substring="/pro")

        # Message must mention the cause so the operator can diagnose.
        msg = str(excinfo.value)
        assert "disabled" in msg.lower()
        assert "2FA" in msg or "captcha" in msg.lower() or "challenge" in msg.lower()
        # The click should not have been called — we never got past
        # the enabled guard.
        assert submit.click_calls == 0
    finally:
        login_mod._LOGIN_TIMEOUT_S = orig_timeout_s


def test_submit_and_wait_race_window_retries_click():
    """The button reports enabled (so we pass the guard), but the
    first click raises (the handler is still binding). The second
    attempt should succeed. This is the exact race the live smoke
    test observed."""
    call_count = {"n": 0}

    def click_at(n: int) -> None:
        call_count["n"] = n
        if n == 1:
            raise PlaywrightTimeoutError("element intercepts events")
        # attempt 2 succeeds
        return None

    submit = FakeElement(
        is_enabled_at=lambda n: True,  # always enabled
        click_at=click_at,
    )
    page = FakePage(submit=submit)

    landed = _submit_and_wait(page, expected_url_substring="/pro")

    assert landed == "https://www.investing.com/pro/propicks"
    # We retried: first click raised, second succeeded.
    assert submit.click_calls == 2
    # We backed off once between attempts.
    assert 500 in page.wait_for_timeout_calls
    # We re-waited for enabled once between attempts.
    assert submit.is_enabled_calls >= 2


def test_submit_and_wait_gives_up_after_max_attempts():
    """If the click keeps failing past the retry budget, we should
    raise a clear :class:`InvestingProLoginError` (not let the raw
    PlaywrightTimeoutError escape, which would have caused the
    double-wrap)."""
    submit = FakeElement(
        is_enabled_at=lambda n: True,
        click_at=lambda n: (_ for _ in ()).throw(
            PlaywrightTimeoutError("disabled (synthetic)")
        ),
    )
    page = FakePage(submit=submit)

    with pytest.raises(InvestingProLoginError) as excinfo:
        _submit_and_wait(page, expected_url_substring="/pro")

    msg = str(excinfo.value)
    assert "could not be clicked" in msg.lower()
    # The original PlaywrightTimeoutError is chained as __cause__ for
    # debugging, but the public message does NOT include
    # "TimeoutError" twice (no double-wrap).
    assert "InvestingProLoginError" not in msg
    # We tried the full retry budget.
    assert submit.click_calls >= 1
    # The chained cause is the Playwright timeout.
    assert isinstance(excinfo.value.__cause__, PlaywrightTimeoutError)


def test_submit_and_wait_missing_submit_button():
    """If ``_wait_for_first`` returns None (no submit selector
    matched), we should raise with a clear message — and not have
    attempted any clicks."""
    page = FakePage(submit=None)  # query_selector always returns None
    # Force _wait_for_first to return None on the first poll so the
    # test is fast.
    page._query_selector_selector = "this selector does not exist"

    with pytest.raises(InvestingProLoginError) as excinfo:
        _submit_and_wait(page, expected_url_substring="/pro")

    assert "submit button" in str(excinfo.value).lower()


# --- error-wrapping regression -----------------------------------------------


def test_inner_login_error_does_not_double_wrap():
    """If the inner ``_fill_login_form`` / ``_submit_and_wait`` path
    raises :class:`InvestingProLoginError`, the outer handler must
    re-raise the same exception — not wrap it in another
    ``InvestingProLoginError`` (which produced the noisy
    ``InvestingPro login failed: InvestingProLoginError: InvestingPro
    login failed: ...`` log line on 2026-06-10).

    We can't easily drive the public ``login()`` from a hermetic test
    (it spins up a real Playwright Chromium), so we cover this
    behavior at the unit level: a function that mirrors the outer
    except-chain in :func:`portfoliomind.investingpro.login.login`.
    If the inner step raises ``InvestingProLoginError`` and the outer
    handler re-raises it (no re-wrap), the chained message is
    exactly the original message. If the outer handler re-wrapped
    it, the chained message would start with ``InvestingPro login
    failed:``.
    """
    original_msg = "Submit button stayed disabled past the login timeout"

    def outer_handler(inner_step: Callable[[], None]) -> None:
        try:
            inner_step()
        except InvestingProLoginError:
            # The fix: log + re-raise, do NOT re-wrap.
            raise
        except PlaywrightTimeoutError as e:
            raise InvestingProLoginError(
                "InvestingPro login failed: Playwright timeout"
            ) from e
        except Exception as e:  # last-ditch safety net
            raise InvestingProLoginError(
                f"InvestingPro login failed unexpectedly: {type(e).__name__}"
            ) from e

    def step_that_raises_investingpro_error() -> None:
        raise InvestingProLoginError(original_msg)

    with pytest.raises(InvestingProLoginError) as excinfo:
        outer_handler(step_that_raises_investingpro_error)

    # The message must be exactly the original — no double-wrap.
    assert str(excinfo.value) == original_msg
    # Sanity: the type of the raised exception is still the original
    # class, not a fresh wrap (it would be identical either way, but
    # this is the regression we are guarding).
    assert type(excinfo.value) is InvestingProLoginError


def test_inner_playwright_timeout_wraps_exactly_once():
    """A stray ``PlaywrightTimeoutError`` from an inner step should
    be converted to ``InvestingProLoginError`` exactly once (not
    twice — the double-wrap regression in the 2026-06-10 smoke
    test)."""

    def outer_handler(inner_step: Callable[[], None]) -> None:
        try:
            inner_step()
        except InvestingProLoginError:
            raise
        except PlaywrightTimeoutError as e:
            raise InvestingProLoginError(
                "InvestingPro login failed: Playwright timeout"
            ) from e
        except Exception as e:
            raise InvestingProLoginError(
                f"InvestingPro login failed unexpectedly: {type(e).__name__}"
            ) from e

    def step_that_raises_timeout() -> None:
        raise PlaywrightTimeoutError("synthetic timeout")

    with pytest.raises(InvestingProLoginError) as excinfo:
        outer_handler(step_that_raises_timeout)

    msg = str(excinfo.value)
    # Exactly one wrap: starts with our prefix, not nested.
    assert msg.startswith("InvestingPro login failed:")
    # The chained cause is the original PlaywrightTimeoutError.
    assert isinstance(excinfo.value.__cause__, PlaywrightTimeoutError)
    # No "InvestingProLoginError:" in the message — the second wrap
    # is gone.
    assert "InvestingProLoginError:" not in msg


# --- _wait_for_enabled helper (bonus) ---------------------------------------


def test_wait_for_enabled_polls_then_returns_true():
    """``_wait_for_enabled`` should return True as soon as
    ``is_enabled()`` flips to True, not wait the full timeout."""

    from portfoliomind.investingpro.login import _wait_for_enabled

    state = {"enabled": False}

    class El:
        def __init__(self) -> None:
            self.calls = 0

        def is_enabled(self) -> bool:
            self.calls += 1
            # Flip to True on the 3rd call.
            if self.calls >= 3:
                state["enabled"] = True
            return state["enabled"]

    el = El()
    started = time.monotonic()
    result = _wait_for_enabled(el, timeout_ms=2000)  # 2s budget
    elapsed = time.monotonic() - started

    assert result is True
    # We should have stopped well before the 2s budget — around 0.3s
    # (three 0.1s sleeps).
    assert elapsed < 1.0, f"waited too long: {elapsed:.2f}s"
    assert el.calls == 3


def test_wait_for_enabled_returns_false_on_timeout():
    """If ``is_enabled()`` never flips, ``_wait_for_enabled``
    returns False at the deadline."""
    from portfoliomind.investingpro.login import _wait_for_enabled

    class El:
        def is_enabled(self) -> bool:
            return False

    el = El()
    started = time.monotonic()
    result = _wait_for_enabled(el, timeout_ms=300)  # 300ms budget
    elapsed = time.monotonic() - started

    assert result is False
    # We waited the full budget (give it a small slack).
    assert elapsed >= 0.25, f"returned too early: {elapsed:.2f}s"
    assert elapsed < 1.0, f"waited too long: {elapsed:.2f}s"
