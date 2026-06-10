"""Unit tests for :mod:`portfoliomind.config`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from portfoliomind.config import (
    ALL_VARS,
    OPTIONAL_VARS,
    REQUIRED_VARS,
    ConfigError,
    PortfoliomindConfig,
    _resolve_service_account_json,
    load_env_sources,
)

from .conftest import FAKE_SERVICE_ACCOUNT, full_env


# --- _resolve_service_account_json -----------------------------------------


def test_resolve_sa_json_inline_string():
    info = _resolve_service_account_json(
        '{"type":"service_account","project_id":"inline"}'
    )
    assert info["type"] == "service_account"
    assert info["project_id"] == "inline"


def test_resolve_sa_json_from_file(tmp_path: Path):
    sa_path = tmp_path / "sa.json"
    sa_path.write_text(json.dumps(FAKE_SERVICE_ACCOUNT), encoding="utf-8")
    info = _resolve_service_account_json(str(sa_path))
    assert info["client_email"] == FAKE_SERVICE_ACCOUNT["client_email"]


def test_resolve_sa_json_empty_raises():
    with pytest.raises(ConfigError, match="empty"):
        _resolve_service_account_json("")


def test_resolve_sa_json_missing_path_raises():
    with pytest.raises(ConfigError, match="does not exist"):
        _resolve_service_account_json("/nonexistent/path/sa.json")


def test_resolve_sa_json_malformed_raises():
    with pytest.raises(ConfigError, match="not valid JSON"):
        _resolve_service_account_json("{not json")


def test_resolve_sa_json_wrong_type_raises():
    with pytest.raises(ConfigError, match="service account key"):
        _resolve_service_account_json('{"type":"user","foo":"bar"}')


def test_resolve_sa_json_non_object_raises():
    with pytest.raises(ConfigError, match=r"JSON object"):
        _resolve_service_account_json("[]")


# --- PortfoliomindConfig.from_env -------------------------------------------


def test_from_env_happy_path():
    cfg = PortfoliomindConfig.from_env(env=full_env())
    assert cfg.investingpro_email == "test@example.com"
    assert cfg.investingpro_password == "test-password"  # noqa: S105 (test)
    assert cfg.xtb_user_id == "test-xtb-user"
    assert cfg.xtb_password == "test-xtb-password"  # noqa: S105
    assert cfg.google_sheet_id == ""  # blank -> bootstrap
    assert cfg.openai_api_key == "test-openai-key"  # noqa: S105
    assert not cfg.has_existing_sheet()


def test_from_env_accepts_blank_sheet_id():
    env = full_env(sheet_id="")
    cfg = PortfoliomindConfig.from_env(env=env)
    assert cfg.google_sheet_id == ""
    assert not cfg.has_existing_sheet()


def test_from_env_accepts_real_sheet_id():
    env = full_env(sheet_id="abc123xyz")
    cfg = PortfoliomindConfig.from_env(env=env)
    assert cfg.google_sheet_id == "abc123xyz"
    assert cfg.has_existing_sheet()


def test_from_env_resolves_sa_from_file(tmp_path: Path):
    sa_path = tmp_path / "sa.json"
    sa_path.write_text(json.dumps(FAKE_SERVICE_ACCOUNT), encoding="utf-8")
    env = full_env()
    env["GOOGLE_SERVICE_ACCOUNT_JSON"] = str(sa_path)
    cfg = PortfoliomindConfig.from_env(env=env)
    assert cfg.google_service_account_info["client_email"] == FAKE_SERVICE_ACCOUNT["client_email"]


def test_from_env_lists_all_missing_at_once():
    """If 3 vars are missing, the error should list all 3 in one message."""
    env = full_env()
    for key in ("INVESTINGPRO_EMAIL", "XTB_USER_ID", "OPENAI_API_KEY"):
        del env[key]
    with pytest.raises(ConfigError) as exc_info:
        PortfoliomindConfig.from_env(env=env)
    msg = str(exc_info.value)
    assert "INVESTINGPRO_EMAIL" in msg
    assert "XTB_USER_ID" in msg
    assert "OPENAI_API_KEY" in msg


def test_from_env_blank_required_var_counts_as_missing():
    env = full_env()
    env["XTB_PASSWORD"] = "   "  # whitespace-only
    with pytest.raises(ConfigError, match="XTB_PASSWORD"):
        PortfoliomindConfig.from_env(env=env)


def test_from_env_blank_sheet_id_does_not_count_as_missing():
    env = full_env(sheet_id="")
    # This should NOT raise.
    cfg = PortfoliomindConfig.from_env(env=env)
    assert cfg.google_sheet_id == ""


def test_from_env_repr_does_not_leak_secrets():
    cfg = PortfoliomindConfig.from_env(env=full_env(sheet_id="abc123"))
    r = repr(cfg)
    assert "test-password" not in r
    assert "test-xtb-password" not in r
    assert "test-openai-key" not in r
    # But the (non-secret) sheet id IS shown for debuggability.
    assert "abc123" in r


def test_from_env_uses_optional_defaults():
    env = full_env()
    del env["SESSION_DIR"]
    del env["SCREENSHOT_DIR"]
    cfg = PortfoliomindConfig.from_env(env=env)
    assert str(cfg.session_dir).endswith("sessions")
    assert str(cfg.screenshot_dir).endswith("screenshots")


def test_known_vars_constant():
    """The REQUIRED_VARS + OPTIONAL_VARS tuple should match what the spec mandates."""
    assert set(REQUIRED_VARS) == {
        "INVESTINGPRO_EMAIL",
        "INVESTINGPRO_PASSWORD",
        "XTB_USER_ID",
        "XTB_PASSWORD",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_SHEET_ID",
        "OPENAI_API_KEY",
    }
    # Card 7 added sizing + approval env knobs.
    assert set(OPTIONAL_VARS) == {
        "SESSION_DIR",
        "SCREENSHOT_DIR",
        "XTB_PER_TRADE_CAP",
        "XTB_MAX_OPEN_POSITIONS",
        "XTB_SL_PCT",
        "XTB_TP_PCT",
        "XTB_MAX_COMMISSION_PCT",
        "APPROVAL_TIMEOUT_MIN",
        "DISCORD_BOT_TOKEN",
        "DISCORD_HOME_CHANNEL_THREAD_ID",
    }
    assert set(ALL_VARS) == set(REQUIRED_VARS) | set(OPTIONAL_VARS)


# --- load_env_sources -------------------------------------------------------


def test_load_env_sources_idempotent(tmp_path: Path, monkeypatch):
    """Calling load_env_sources twice should not change already-set vars."""
    profile_env = tmp_path / "profile.env"
    project_env = tmp_path / "project.env"
    profile_env.write_text("X_FROM_PROFILE=profile\nX_FROM_PROJECT=profile\n", encoding="utf-8")
    project_env.write_text("X_FROM_PROJECT=project\n", encoding="utf-8")

    # Clear any pre-existing values from the host env for the keys we touch.
    for k in ("X_FROM_PROFILE", "X_FROM_PROJECT"):
        monkeypatch.delenv(k, raising=False)

    load_env_sources(profile_env=profile_env, project_env=project_env)
    assert os.environ.get("X_FROM_PROFILE") == "profile"
    # Project env should win (later source).
    assert os.environ.get("X_FROM_PROJECT") == "project"

    # Re-calling: the process env already has the value, so dotenv's
    # override=False keeps it.
    monkeypatch.setenv("X_FROM_PROFILE", "set-in-process")
    load_env_sources(profile_env=profile_env, project_env=project_env)
    assert os.environ.get("X_FROM_PROFILE") == "set-in-process"


def test_load_env_sources_honors_hermes_home(tmp_path: Path, monkeypatch):
    """When $HERMES_HOME is set, the loader looks at $HERMES_HOME/.env first
    (this is the active profile's .env file). Without HERMES_HOME, falls back
    to the legacy ~/.hermes/profiles/builder/.env path. The lookup is
    first-hit-wins, so the active profile always beats the legacy fallback.

    This is the back-compat fix for the portfoliomind profile
    (HERMES_HOME=/opt/data/profiles/portfoliomind, env at $HERMES_HOME/.env).
    The foundation card had it hardcoded to ~/.hermes/profiles/builder/.env.
    """
    # Case 1: $HERMES_HOME set, active-profile .env exists.
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    active_env = hermes_home / ".env"
    active_env.write_text("HERMES_HOME_KEY=from-active-profile\n", encoding="utf-8")

    monkeypatch.delenv("HERMES_HOME_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path / "no-legacy-home"))  # ensure no legacy path

    # Note: do NOT pass profile_env/project_env - the test verifies the
    # implicit candidate resolution path.
    loaded = load_env_sources()
    assert active_env in loaded
    assert os.environ.get("HERMES_HOME_KEY") == "from-active-profile"

    # Case 2: $HERMES_HOME set, no active .env, legacy path exists.
    monkeypatch.delenv("HERMES_HOME", raising=False)
    legacy_home = tmp_path / "legacy-home"
    legacy_env = legacy_home / ".hermes" / "profiles" / "builder" / ".env"
    legacy_env.parent.mkdir(parents=True, exist_ok=True)
    legacy_env.write_text("HERMES_HOME_KEY=from-legacy\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(legacy_home))
    monkeypatch.delenv("HERMES_HOME_KEY", raising=False)

    loaded = load_env_sources()
    assert legacy_env in loaded
    assert os.environ.get("HERMES_HOME_KEY") == "from-legacy"

    # Case 3: both active and legacy exist; active wins (first hit).
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(legacy_home))
    monkeypatch.delenv("HERMES_HOME_KEY", raising=False)

    loaded = load_env_sources()
    # active_env is loaded; legacy_env is NOT (we break at first hit).
    assert active_env in loaded
    assert legacy_env not in loaded
    assert os.environ.get("HERMES_HOME_KEY") == "from-active-profile"
