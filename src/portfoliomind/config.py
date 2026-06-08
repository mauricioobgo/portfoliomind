"""Typed environment loader for the PortfolioMind agent.

The loader reads env in this order (later wins):

    1. ``~/.hermes/profiles/builder/.env`` (the Hermes profile env)
    2. ``<project>/.env`` (the project-local env, if present)
    3. The current process env (highest precedence)

Rationale: the operator already maintains their secrets in the Hermes profile
env, so we default to that. The project ``.env`` is for overrides when running
locally. The process env always wins so a container / cron override is honored.

Validation is all-or-nothing: if any required var is missing, we raise
:class:`ConfigError` listing **every** missing var in one shot. The agent
script can then present a single actionable error to the user instead of
making them fix one var, retry, hit the next, retry, ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional

from dotenv import dotenv_values

# Constants — public so tests and CLI scripts can reference them without
# duplicating string literals.
#
# Resolution order for the Hermes profile env (back-compat preserving the
# foundation card's behavior, but honoring whichever profile is active):
#   1. ``$HERMES_HOME/.env`` if $HERMES_HOME is set (the active profile)
#   2. ``~/.hermes/profiles/builder/.env`` (legacy default — the foundation
#      card's hardcoded path)
#   3. ``$HERMES_HOME/profiles/builder/.env`` (extra back-compat)
#
# The constants below expose the candidates; load_env_sources() picks the
# first one that exists.
def _hermes_home_candidates() -> list[Path]:
    """Return candidate .env paths in precedence order (highest first)."""
    hh = os.environ.get("HERMES_HOME", "").strip()
    if hh:
        hh_path = Path(hh)
        return [
            hh_path / ".env",
            Path.home() / ".hermes" / "profiles" / "builder" / ".env",
            hh_path / "profiles" / "builder" / ".env",
        ]
    # No HERMES_HOME: fall back to ~/.hermes/profiles/builder/.env
    return [Path.home() / ".hermes" / "profiles" / "builder" / ".env"]


HERMES_PROFILE_ENV: Path = _hermes_home_candidates()[0]
PROJECT_ENV: Path = Path.cwd() / ".env"

REQUIRED_VARS: tuple[str, ...] = (
    "INVESTINGPRO_EMAIL",
    "INVESTINGPRO_PASSWORD",
    "XTB_USER_ID",
    "XTB_PASSWORD",
    "GOOGLE_SERVICE_ACCOUNT_JSON",
    "GOOGLE_SHEET_ID",  # empty string allowed -> bootstrap new sheet
    "OPENAI_API_KEY",
)
OPTIONAL_VARS: tuple[str, ...] = (
    "SESSION_DIR",
    "SCREENSHOT_DIR",
)
ALL_VARS: tuple[str, ...] = REQUIRED_VARS + OPTIONAL_VARS


class ConfigError(RuntimeError):
    """Raised when one or more required env vars are missing or malformed."""


def load_env_sources(
    *,
    profile_env: Optional[Path] = None,
    project_env: Path = PROJECT_ENV,
) -> list[Path]:
    """Load env files in precedence order (later wins) into ``os.environ``.

    Precedence (highest first):
      1. Process env (values already in ``os.environ`` when we run)
      2. Project ``.env`` (if present)
      3. Hermes profile ``.env`` (first hit wins among the candidates):
         - ``profile_env`` argument if provided (test override path)
         - ``$HERMES_HOME/.env`` (the active profile)
         - ``~/.hermes/profiles/builder/.env`` (legacy default)
         - ``$HERMES_HOME/profiles/builder/.env`` (extra back-compat)

    The Hermes profile lookup is "first hit wins" — only one Hermes profile
    env is loaded. This means the active profile's .env always wins over
    any legacy fallback, even if both exist.

    Returns the list of files that were found and applied. Idempotent — safe
    to call once at process start.

    Implementation: we read each file with :func:`dotenv.dotenv_values` (which
    does NOT mutate ``os.environ``) into a dict, then merge them. Project wins
    over profile. Process env wins over both files (we only ``os.environ``
    keys that are NOT already set).
    """
    merged: dict[str, str] = {}
    loaded: list[Path] = []
    # Pick the first existing Hermes profile env (first hit wins).
    if profile_env is not None:
        candidates = [profile_env]
    else:
        candidates = _hermes_home_candidates()
    for cand in candidates:
        if cand.is_file():
            merged.update({k: v for k, v in dotenv_values(cand).items() if v is not None})
            loaded.append(cand)
            break
    if project_env.is_file():
        merged.update({k: v for k, v in dotenv_values(project_env).items() if v is not None})
        loaded.append(project_env)
    # Apply to process env, but never clobber an already-set value.
    for k, v in merged.items():
        os.environ.setdefault(k, v)
    return loaded


def _resolve_service_account_json(raw: str) -> dict[str, Any]:
    """Accept either a filesystem path to a JSON file OR a raw JSON string.

    Returns the parsed service-account dict. Raises :class:`ConfigError` with a
    clear message (without echoing the value) on parse failure.

    Heuristic for path-vs-raw:
      * If the trimmed input starts with ``{`` or ``[`` we treat it as raw
        JSON content (objects and arrays). We do NOT try the filesystem path
        in this case because a real path starting with those chars would be
        extremely unusual.
      * Otherwise we treat it as a filesystem path and read+parse it.
    """
    raw = raw.strip()
    if not raw:
        raise ConfigError("GOOGLE_SERVICE_ACCOUNT_JSON is empty")

    parsed_obj: object
    if raw.startswith("{") or raw.startswith("["):
        # Raw JSON content. Parse strictly.
        try:
            parsed_obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConfigError(f"GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {e.msg}") from None
    else:
        # Filesystem path.
        path = Path(raw).expanduser()
        if not path.is_file():
            raise ConfigError(
                f"GOOGLE_SERVICE_ACCOUNT_JSON points to a path that does not exist: {path}"
            )
        try:
            parsed_obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"GOOGLE_SERVICE_ACCOUNT_JSON file at {path} is not valid JSON: {e.msg}"
            ) from None

    if not isinstance(parsed_obj, dict):
        raise ConfigError(
            "GOOGLE_SERVICE_ACCOUNT_JSON must decode to a JSON object "
            f"(got {type(parsed_obj).__name__})"
        )
    if parsed_obj.get("type") != "service_account":
        raise ConfigError(
            "GOOGLE_SERVICE_ACCOUNT_JSON must be a service account key "
            '(expected "type": "service_account")'
        )
    return parsed_obj


@dataclass(frozen=True)
class PortfoliomindConfig:
    """Strict, frozen view of all required + optional config.

    Construct via :meth:`from_env`. The frozen dataclass guarantees the agent
    cannot accidentally mutate its config mid-session.
    """

    # Required - InvestingPro
    investingpro_email: str
    investingpro_password: str
    # Required - XTB
    xtb_user_id: str
    xtb_password: str
    # Required - Google Sheets
    google_service_account_info: dict[str, Any]  # parsed JSON, ready for google-auth
    google_sheet_id: str  # empty string -> bootstrap new sheet
    # Required - LLM
    openai_api_key: str
    # Optional
    session_dir: Path = field(default_factory=lambda: Path("./sessions"))
    screenshot_dir: Path = field(default_factory=lambda: Path("./screenshots"))

    KNOWN_VARS: ClassVar[tuple[str, ...]] = ALL_VARS

    # --- Construction ---

    @classmethod
    def from_env(cls, *, env: Optional[Mapping[str, str]] = None) -> "PortfoliomindConfig":
        """Build a config from the current process env (or a passed-in mapping, for tests).

        Triggers ``load_env_sources`` first unless an explicit ``env`` mapping is
        provided — this keeps the test path fully hermetic.
        """
        if env is None:
            load_env_sources()
            env = os.environ

        # Normalize to a plain dict for type-narrowing safety.
        env_dict: dict[str, str] = {k: v for k, v in env.items()}

        # 1) Collect missing required vars (all at once).
        missing = [
            name
            for name in REQUIRED_VARS
            if name not in env_dict or env_dict.get(name, "").strip() == ""
        ]
        # Note: GOOGLE_SHEET_ID is allowed to be an empty string (bootstrap path).
        if "GOOGLE_SHEET_ID" in missing and env_dict.get("GOOGLE_SHEET_ID") == "":
            missing.remove("GOOGLE_SHEET_ID")
        if missing:
            raise ConfigError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + "\nSet them in ~/.hermes/profiles/builder/.env or in a local .env file."
            )

        # 2) Required vars present -> parse them. Errors here are fatal but
        #    raised as ConfigError so the agent exits cleanly.
        try:
            sa_info = _resolve_service_account_json(env_dict["GOOGLE_SERVICE_ACCOUNT_JSON"])
        except ConfigError:
            raise
        except Exception as e:  # last-ditch safety net
            raise ConfigError(f"GOOGLE_SERVICE_ACCOUNT_JSON: unexpected error: {e!r}") from None

        return cls(
            investingpro_email=env_dict["INVESTINGPRO_EMAIL"],
            investingpro_password=env_dict["INVESTINGPRO_PASSWORD"],
            xtb_user_id=env_dict["XTB_USER_ID"],
            xtb_password=env_dict["XTB_PASSWORD"],
            google_service_account_info=sa_info,
            google_sheet_id=env_dict.get("GOOGLE_SHEET_ID", "").strip(),
            openai_api_key=env_dict["OPENAI_API_KEY"],
            session_dir=Path(env_dict.get("SESSION_DIR", "./sessions")).expanduser(),
            screenshot_dir=Path(env_dict.get("SCREENSHOT_DIR", "./screenshots")).expanduser(),
        )

    # --- Helpers ---

    def has_existing_sheet(self) -> bool:
        """True when an explicit sheet ID was provided (vs. blank -> bootstrap)."""
        return bool(self.google_sheet_id)

    def __repr__(self) -> str:
        # CRITICAL: never echo secrets in repr/log. Only the sheet ID is
        # safe to display.
        return (
            f"PortfoliomindConfig("
            f"investingpro_email={self.investingpro_email!r}, "
            f"google_sheet_id={self.google_sheet_id!r}, "
            f"has_existing_sheet={self.has_existing_sheet()}, "
            f"session_dir={self.session_dir}, "
            f"screenshot_dir={self.screenshot_dir})"
        )


__all__ = [
    "HERMES_PROFILE_ENV",
    "PROJECT_ENV",
    "REQUIRED_VARS",
    "OPTIONAL_VARS",
    "ALL_VARS",
    "ConfigError",
    "load_env_sources",
    "PortfoliomindConfig",
]
