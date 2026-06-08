# PortfolioMind v4 — agent guidance for future cards

This file is a hand-off note for the next agents (cards 2/3/4) that will
build on the foundation laid by card 1.

## Public contract (do not change without coordinating all 4 cards)

These imports and method signatures are stable. Cards 2/3/4 import from
them directly.

```python
from portfoliomind.config import PortfoliomindConfig
from portfoliomind.sheets.client import SheetsClient
from portfoliomind.sheets.schema import TAB_NAMES, TAB_HEADERS
from portfoliomind.sheets.bootstrap import bootstrap_sheet
```

The `SheetsClient` API surface (5 stable methods):
- `get_sheet(sheet_id) -> dict`
- `ensure_worksheet(sheet_id, title, headers) -> dict` (idempotent)
- `read_range(sheet_id, tab_name, range_a1) -> list[list[str]]`
- `write_range(sheet_id, tab_name, range_a1, values) -> None`
- `append_rows(sheet_id, tab_name, values) -> int` (returns 1-indexed first-row)
- `row_count(sheet_id, tab_name) -> int`
- `find_worksheet(sheet_id, title) -> dict | None`

## Conventions

- **Time**: all log lines, sheet timestamps, and order records use
  `portfoliomind.time_utils.now_bogota()` / `iso_now()`. Never
  `datetime.now()` without tz.
- **Logging**: every module does `from ..logging_setup import get_logger`
  and uses the returned logger. Setup is done once at process start by
  the CLI script.
- **Secrets**: never log, write, or include in error messages. The
  config dataclass repr only shows the (non-secret) sheet ID.
- **Idempotency**: every Sheets operation is idempotent. Re-running any
  script on the same sheet is safe.
- **Paths**: use absolute paths in error messages; use
  `portfoliomind.paths.session_dir()` and `screenshot_dir()` for runtime
  state dirs (they read env + create-on-demand).

## Test discipline

- The tests in `tests/` are hermetic — they use an in-memory fake of the
  Sheets API (see `test_bootstrap_idempotent.py` for the pattern). No
  real Google credentials are needed to run `uv run pytest tests/`.
- New cards should add tests in the same hermetic style. Cards 2/3 will
  need a similar fake for Playwright (use a `FakePage` or
  `playwright.sync_api`'s `BrowserContext` mock).

## Env

The required env vars are validated by `PortfoliomindConfig.from_env()`.
The validator lists ALL missing vars in a single error message — do not
fail-fast on the first one. See `src/portfoliomind/config.py` for the
list.

## Things still in scope for future cards (not done in card 1)

- Card 2: InvestingPro login + scrape (Playwright).
- Card 3: XTB xStation login + order placement with mandatory SL (Playwright).
- Card 4: APScheduler weekday loop at 08:30 America/Bogota + daily
  returns refresh.
- Strategy engine, forecast models, signal scoring, KB application.
- Real trading logic, scoring weights, R/R checks, KB-5
  disqualifications.
