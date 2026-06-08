# PortfolioMind v4

Multi-strategy equity investing agent. Pulls AI picks from InvestingPro, scores
them under a configurable per-regime strategy, executes user-approved orders
via XTB xStation 5, and tracks the full lifecycle in a Google Sheet.

This repository is split into four cards. **Card 1 (this branch) ships the
foundation**: project scaffold, typed config loader, Google Sheets client +
schema + bootstrap, and a dry-run script. Cards 2/3/4 add the InvestingPro
scraper, the XTB trade executor, and the scheduler respectively.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) 0.11+
- A Google Cloud project with a **service account** that has Editor access
  to the target Google Sheet. See
  [Google's service-account quickstart](https://cloud.google.com/iam/docs/keys-create-delete#creating).
- (For cards 2/3) An InvestingPro subscription and an XTB xStation 5 account.

## Install

```bash
uv sync
uv run playwright install chromium
```

`uv sync` resolves and locks every dependency from `pyproject.toml`. The
Playwright step fetches the Chromium browser binary that cards 2/3 will drive.

## Environment setup

Copy `.env.example` to `.env` and fill in real values, **or** add the same
keys to `~/.hermes/profiles/builder/.env`. The config loader reads the
profile env first, then the project `.env`, then the process env (later
sources win).

| Variable | Required? | Used by |
|---|---|---|
| `INVESTINGPRO_EMAIL` | yes | card 2 |
| `INVESTINGPRO_PASSWORD` | yes | card 2 |
| `XTB_USER_ID` | yes | card 3 |
| `XTB_PASSWORD` | yes | card 3 |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | yes | card 1 (and cards 2/3/4) |
| `GOOGLE_SHEET_ID` | yes (blank allowed) | card 1 |
| `OPENAI_API_KEY` | yes (validated, not exercised in card 1) | future strategy / narrative cards |
| `SESSION_DIR` | no (default `./sessions`) | card 2 (Playwright cookies) |
| `SCREENSHOT_DIR` | no (default `./screenshots`) | card 3 (pre-trade screenshots) |

`GOOGLE_SERVICE_ACCOUNT_JSON` accepts either an absolute path to a JSON file
or the raw JSON contents inline. Path form is recommended.

## Usage

```bash
# 1) Create the sheet (or verify the existing one) and ensure all 11 tabs.
uv run python scripts/bootstrap_sheet.py

# 2) Smoke-test the Sheets integration: write one timestamped row to each tab.
uv run python scripts/dry_run.py
```

`bootstrap_sheet.py` prints `(sheet_id, sheet_url)` to stdout. Re-run it as
many times as you like — it's idempotent.

`dry_run.py` appends one row per tab on each run, so the tab's row count
grows by 1 each time. This is intentional: it lets you watch the
`append_rows` smart-append behavior and the sheet fill up over multiple
runs.

Both scripts accept these flags:

- `--sheet-id <id>` — override `GOOGLE_SHEET_ID` for this run
- `--no-bootstrap` (dry-run only) — skip tab creation/verification
- `--log-level {DEBUG,INFO,WARNING,ERROR}` — verbosity

## What each tab is for

| Tab | Contents | Updated |
|---|---|---|
| 📥 Raw Picks | Full InvestingPro AI picks scraped this session | Per session |
| 🎯 Strategy Selection | Active 3 strategies (S/M/L), regime, rationale | Per session |
| 📊 Signal Scorecard | Per-instrument scoring under driving strategy | Per session |
| 🔮 Forecasts | Full FM forecast block per shortlisted instrument | Per session |
| 🚫 Disqualified | Removed instruments + reason | Per session |
| ✅ Approved Trades | User-confirmed trades, amounts, SL, TP, strategy, timeframe | On approval |
| 📈 Executed Orders | Order IDs, prices, timestamps | On execution |
| 💰 Returns Tracker | Live P&L by timeframe bucket, vs SPY, dividend income | Daily |
| 📊 Forecast Accuracy | Closed position actuals vs. all forecast models | On close |
| 📰 Macro Context | VIX, SPY, Fed rate, sector RS, regime | Per session |
| 🗒️ Agent Log | Full audit trail of all actions, decisions, model weight changes | Continuous |

See `src/portfoliomind/sheets/schema.py` for the exact column headers of
each tab. The Returns Tracker columns are pinned verbatim from the v4
spec; the other tabs use a minimum-viable header set that cards 2/3/4 may
refine as scraping reveals the real shape.

## Tests

```bash
uv run pytest tests/ -v
```

The tests are hermetic — they use an in-memory fake of the Google Sheets
API, so no service-account credentials are required to run them.

## Public contract for cards 2/3/4

The following imports and method signatures are the public contract this
card ships. Treat them as stable; cards 2/3/4 will import these directly.

```python
from portfoliomind.config import PortfoliomindConfig
from portfoliomind.sheets.client import SheetsClient
from portfoliomind.sheets.schema import TAB_NAMES, TAB_HEADERS
from portfoliomind.sheets.bootstrap import bootstrap_sheet
```

```python
class SheetsClient:
    @classmethod
    def from_config(cls, config: PortfoliomindConfig) -> "SheetsClient": ...
    def get_sheet(self, sheet_id: str) -> dict: ...
    def ensure_worksheet(self, sheet_id: str, title: str, headers: list[str]) -> dict: ...
    def read_range(self, sheet_id: str, tab_name: str, range_a1: str) -> list[list[str]]: ...
    def write_range(self, sheet_id: str, tab_name: str, range_a1: str, values: list[list]) -> None: ...
    def append_rows(self, sheet_id: str, tab_name: str, values: list[list]) -> int: ...
    def row_count(self, sheet_id: str, tab_name: str) -> int: ...
    def find_worksheet(self, sheet_id: str, title: str) -> dict | None: ...
    def create_spreadsheet(self, title: str) -> tuple[str, str]: ...
    def delete_worksheet(self, sheet_id: str, title: str) -> bool: ...
    def list_worksheets(self, sheet_id: str) -> list[dict]: ...
```

## Spec

The full v4 agent prompt lives at
`/opt/data/cache/documents/doc_cedcf4aba1b6_xtb-investment-agent-prompt-v4.md`
(referenced as `xtb-investment-agent-prompt-v4.md`). This README and the
schema in `sheets/schema.py` are derived from it.

## Layout

```
portfoliomind/
  pyproject.toml              # uv project, all deps pinned
  uv.lock                     # generated by uv sync
  README.md
  .env.example
  .gitignore
  src/portfoliomind/
    __init__.py
    config.py                 # typed env loader (REQUIRED_VARS, PortfoliomindConfig)
    logging_setup.py          # structured logging (Bogota tz)
    paths.py                  # SESSION_DIR, SCREENSHOT_DIR resolution
    time_utils.py             # America/Bogota tz, ISO timestamps
    sheets/
      __init__.py
      client.py               # google-api-python-client wrapper
      schema.py               # 11 tab name constants + header row definitions
      bootstrap.py            # bootstrap_sheet() — create sheet, ensure all 11 tabs
  scripts/
    bootstrap_sheet.py        # CLI: create/verify the Google Sheet, idempotent
    dry_run.py                # CLI: env load + Sheets auth + write test row to each tab
  sessions/                   # gitignored, persisted Chromium cookies
  screenshots/                # gitignored, pre-trade screenshots
  tests/
    __init__.py
    conftest.py
    test_config.py
    test_schema.py
    test_bootstrap_idempotent.py
```

## License

MIT — see `LICENSE`.
