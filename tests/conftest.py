"""Shared fixtures for the test suite.

We never want unit tests to hit a real Google Sheet. The tests use a fake
service-account dict and a ``SheetsClient`` that talks to an in-memory mock
service.

Note: We deliberately do NOT call :func:`portfoliomind.config.load_env_sources`
from these tests, so the host's ``~/.hermes/profiles/builder/.env`` never
leaks into the test environment. Each test that needs config passes an
explicit env dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable so the tests can `import portfoliomind...` without
# needing the package to be installed. (uv adds this for `uv run`, but the
# pytest run via `uv run pytest` should pick it up via the pyproject
# pythonpath setting. We belt-and-brace it here.)
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Minimal service-account-shaped dict. The tests don't call any real Google
# APIs so the contents don't need to be authentic -- they just need the
# shape ``config._resolve_service_account_json`` validates and that
# ``google-auth`` can parse.
#
# The private key MUST be a parseable PEM because ``SheetsClient.__init__``
# runs ``Credentials.from_service_account_info`` which loads the PEM. We
# generate a fresh RSA-2048 key on first import and cache it for the
# lifetime of the test process.
def _build_test_pem() -> str:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


_TEST_PEM: str = _build_test_pem()

FAKE_SERVICE_ACCOUNT: dict = {
    "type": "service_account",
    "project_id": "test-project",
    "private_key_id": "fake",
    "private_key": _TEST_PEM,
    "client_email": "fake@test-project.iam.gserviceaccount.com",
    "client_id": "0",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/fake",
}


def full_env(sheet_id: str = "") -> dict[str, str]:
    """A complete env mapping that satisfies :class:`PortfoliomindConfig`.

    ``GOOGLE_SERVICE_ACCOUNT_JSON`` is set to a minimal-but-valid inline
    service-account JSON (with the fields ``google-auth`` requires for
    ``from_service_account_info`` to succeed).
    """
    import json
    sa = json.dumps(_FULL_SA_INFO)
    return {
        "INVESTINGPRO_EMAIL": "test@example.com",
        "INVESTINGPRO_PASSWORD": "test-password",
        "XTB_USER_ID": "test-xtb-user",
        "XTB_PASSWORD": "test-xtb-password",
        "GOOGLE_SERVICE_ACCOUNT_JSON": sa,
        "GOOGLE_SHEET_ID": sheet_id,
        "OPENAI_API_KEY": "test-openai-key",
        "SESSION_DIR": "/tmp/pm-test-sessions",
        "SCREENSHOT_DIR": "/tmp/pm-test-screenshots",
    }


_FULL_SA_INFO: dict = {
    "type": "service_account",
    "project_id": "test-project",
    "private_key_id": "fake",
    "private_key": _TEST_PEM,
    "client_email": "fake@test-project.iam.gserviceaccount.com",
    "client_id": "0",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/fake",
}


# Re-export the simplest possible fake for the env path-form test.
FAKE_SERVICE_ACCOUNT_PATH_JSON = '{"type":"service_account","project_id":"x"}'
FAKE_SERVICE_ACCOUNT_PATH_FILE = "/tmp/portfoliomind-test-sa.json"


def write_fake_sa_file(path: str = FAKE_SERVICE_ACCOUNT_PATH_FILE) -> str:
    """Write a fake SA JSON to a temp file. Returns the path. Idempotent."""
    import json

    # Use a stable absolute path under tmp so tests are deterministic.
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(FAKE_SERVICE_ACCOUNT), encoding="utf-8")
    return str(p)
