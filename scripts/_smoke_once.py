"""Manual smoke test for the morning_run path with valid env.

This is the CLI equivalent of the ``--once`` flag, but builds a
PortfoliomindConfig in-process so we can avoid writing a real service
account JSON to disk just for a smoke test.

Run from the repo root:

    uv run python scripts/_smoke_once.py

It exercises the full --once path: config build → morning_run → no
platform modules → "no_platform_modules" outcome → exit 0.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make sure we're importing from src/ regardless of cwd.
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE / "src"))


def _build_fake_sa() -> str:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    return json.dumps({
        "type": "service_account",
        "project_id": "smoke-test",
        "private_key_id": "smoke",
        "private_key": pem,
        "client_email": "smoke@smoke.iam.gserviceaccount.com",
        "client_id": "0",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    })


def main() -> int:
    # Use plain assignments, not setdefault, because the host's
    # ~/.hermes/profiles/portfoliomind/.env has these set to empty
    # strings (the operator hasn't filled in the secrets yet). We
    # need values that satisfy the strict validator, not just
    # "present".
    os.environ["INVESTINGPRO_EMAIL"] = "smoke@example.com"
    os.environ["INVESTINGPRO_PASSWORD"] = "smoke-pw"
    os.environ["XTB_USER_ID"] = "smoke-xtb"
    os.environ["XTB_PASSWORD"] = "smoke-xtb-pw"
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = _build_fake_sa()
    os.environ["GOOGLE_SHEET_ID"] = ""
    os.environ["OPENAI_API_KEY"] = "smoke-sk"

    from portfoliomind.config import PortfoliomindConfig
    from portfoliomind.logging_setup import setup_logging
    from portfoliomind.scheduler.jobs import morning_run

    setup_logging(level="INFO")
    config = PortfoliomindConfig.from_env()
    print(f"config built: {config!r}")
    outcome = morning_run(config=config)
    print(f"outcome: {outcome.summary_line()}")
    print(f"  status={outcome.status}")
    return 0 if outcome.status in ("ran", "no_platform_modules", "skipped_weekend", "skipped_holiday") else 1


if __name__ == "__main__":
    raise SystemExit(main())
