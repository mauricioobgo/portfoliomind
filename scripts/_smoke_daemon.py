"""Manual smoke test for the --daemon path.

Boots the scheduler in --daemon mode, waits 3 seconds, then sends
SIGTERM and asserts the process exits cleanly. Confirms:
  - The scheduler logs "scheduler started"
  - SIGTERM is handled
  - The process exits with status 0

Run from the repo root:

    uv run python scripts/_smoke_daemon.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
SCRIPT = _HERE / "scripts" / "run_scheduler.py"


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
    env = os.environ.copy()
    env["INVESTINGPRO_EMAIL"] = "smoke@example.com"
    env["INVESTINGPRO_PASSWORD"] = "smoke-pw"
    env["XTB_USER_ID"] = "smoke-xtb"
    env["XTB_PASSWORD"] = "smoke-xtb-pw"
    env["GOOGLE_SERVICE_ACCOUNT_JSON"] = _build_fake_sa()
    env["GOOGLE_SHEET_ID"] = ""
    env["OPENAI_API_KEY"] = "smoke-sk"
    env.pop("VIRTUAL_ENV", None)

    proc = subprocess.Popen(
        ["uv", "run", "python", str(SCRIPT), "--daemon",
         "--morning-hh", "9", "--morning-mm", "0",
         "--returns-hh", "17", "--returns-mm", "0"],
        cwd=str(_HERE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(3)
    if proc.poll() is not None:
        # Process exited before we could test SIGTERM. That's a fail.
        out = proc.stdout.read() if proc.stdout else ""
        print(f"FAIL: daemon exited early with code {proc.returncode}\n{out}")
        return 1
    print(f"daemon running pid={proc.pid}, sending SIGTERM")
    proc.send_signal(signal.SIGTERM)
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("FAIL: daemon did not exit after SIGTERM")
        return 1
    print(f"daemon exited with code {proc.returncode}")
    print("--- daemon output ---")
    print(out)
    if "scheduler started" not in out:
        print("FAIL: 'scheduler started' message missing from output")
        return 1
    if proc.returncode != 0:
        print(f"FAIL: daemon exit code {proc.returncode} != 0")
        return 1
    print("PASS: daemon started, accepted SIGTERM, exited cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
