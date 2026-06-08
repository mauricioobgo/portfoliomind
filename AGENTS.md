# PortfolioMind v4 — agent guidance for future cards

This file is a hand-off note for the next agents (cards 2/3/4/5+) that
will build on the foundation laid by card 1 and the scheduler added by
card 4.

## Public contract (do not change without coordinating all 4 cards)

These imports and method signatures are stable. Cards 2/3/4 import from
them directly.

```python
from portfoliomind.config import PortfoliomindConfig
from portfoliomind.sheets.client import SheetsClient
from portfoliomind.sheets.schema import TAB_NAMES, TAB_HEADERS
from portfoliomind.sheets.bootstrap import bootstrap_sheet
```

The `SheetsClient` API surface (7 stable methods):
- `get_sheet(sheet_id) -> dict`
- `ensure_worksheet(sheet_id, title, headers) -> dict` (idempotent)
- `read_range(sheet_id, tab_name, range_a1) -> list[list[str]]`
- `write_range(sheet_id, tab_name, range_a1, values) -> None`
- `append_rows(sheet_id, tab_name, values) -> int` (returns 1-indexed first-row)
- `row_count(sheet_id, tab_name) -> int`
- `find_worksheet(sheet_id, title) -> dict | None`

The `portfoliomind.scheduler` package (card 4) exposes:
- `morning_run(*, config=None, sheets=None, sheet_id=None, today=None, calendar=None) -> MorningOutcome`
- `refresh_returns(*, config=None, sheets=None, sheet_id=None, today=None, price_fetcher=None) -> RefreshOutcome`
- `build_scheduler(cfg=None, *, scheduler_factory=None) -> BaseScheduler`
- `is_morning_trading_day(today=None, *, calendar=HolidayCalendar()) -> bool`
- `HolidayCalendar.from_env(env=None) -> HolidayCalendar`

## Card 4 / scheduler integration seam

The morning job (`portfoliomind.scheduler.jobs.morning_run`) imports
the card 2 and card 3 runners **lazily** at run time, so card 4 can
ship ahead of cards 2/3 without breaking the schedule. Each platform
runner is expected to expose a callable named `run_morning` at
`portfoliomind.investingpro.runner` (card 2) and
`portfoliomind.xtb.runner` (card 3) with this signature:

```python
def run_morning(ctx: MorningContext) -> MorningResult: ...
```

`MorningContext` carries the config, a sheets client, the sheet ID,
the current Bogota timestamp, and a `log_to_sheet(level, message)`
helper. `MorningResult` is a dataclass with `picks_scraped`,
`orders_placed`, `skipped`, `skip_reason`, and `error` fields.

When the modules are not yet implemented, `morning_run` logs a
`no_platform_modules` line to AGENT_LOG and exits cleanly with status
`no_platform_modules` (exit 0 from the CLI). This is the seam — cards
2/3 just need to register a `run_morning` callable and the morning
job picks it up.

## Scheduler cron schedule

The scheduler (card 4) runs two recurring jobs on Bogota-local time:

| Job                 | Cadence                  | Bogota time | Action                                                                 |
|---------------------|--------------------------|-------------|------------------------------------------------------------------------|
| `morning_run`       | Mon–Fri (skip weekends)  | 08:30       | InvestingPro scrape → strategy picks → operator approval → XTB orders  |
| `refresh_returns`   | Daily                    | 16:30       | yfinance lookup for every row in RETURNS_TRACKER; update + prune       |

Both triggers are pinned to `zoneinfo.ZoneInfo("America/Bogota")` —
NOT the host's local time. The production Docker image runs UTC, so
the cron expressions are the only thing keeping the schedule on
Colombia time.

The morning job also consults `PORTFOLIOMIND_HOLIDAYS` (comma-separated
`YYYY-MM-DD` list) to skip configured market holidays. Default = no
holidays (skip = false for every day except Sat/Sun).

## Cron deployment (long-running)

The long-running scheduler is daemonized under the **portfoliomind**
Hermes profile, NOT the default profile. The cron job lives in
`/opt/data/profiles/portfoliomind/cron/`. Register it with:

```bash
hermes cron create \
  --name portfoliomind-scheduler \
  --schedule "0 3 * * *" \
  --workdir /opt/data/portfoliomind \
  --profile portfoliomind \
  --no-agent \
  --script run_scheduler.py --daemon
```

The `0 3 * * *` is 03:00 UTC = 22:00 COT (the daily re-launch covers
any unexpected exit; the scheduler itself runs forever under that
parent). The `--no-agent` flag means cron just spawns the script — no
LLM tokens burned, no agent loop.

To disable the scheduler temporarily (e.g. while debugging):

```bash
hermes cron pause portfoliomind-scheduler
hermes cron resume portfoliomind-scheduler
```

The CLI also supports a one-shot mode for external triggers:

```bash
uv run python scripts/run_scheduler.py --once
```

That runs the morning job once and exits with code 0 (ran/skipped) or
4 (ran but had errors). It is the right entry point for the
`hermes/scheduler` card's CI smoke test.

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
- The scheduler tests in `test_scheduler_logic.py` use a `FakeSheetsClient`
  and an injected `price_fetcher` callable so they never hit Google
  Sheets or yfinance. Network access is **never** required for tests.
- New cards should add tests in the same hermetic style. Cards 2/3 will
  need a similar fake for Playwright (use a `FakePage` or
  `playwright.sync_api`'s `BrowserContext` mock).

## Env

The required env vars are validated by `PortfoliomindConfig.from_env()`.
The validator lists ALL missing vars in a single error message — do not
fail-fast on the first one. See `src/portfoliomind/config.py` for the
list.

Card 4 adds one optional env var:

- `PORTFOLIOMIND_HOLIDAYS` (optional, comma-separated `YYYY-MM-DD`) — the
  dates on which `morning_run` should be skipped. Bad entries are logged
  and ignored; a typo never takes the morning job offline.

## Things still in scope for future cards (not done in cards 1-4)

- Card 2: InvestingPro login + scrape (Playwright). Must register a
  `portfoliomind.investingpro.runner.run_morning` callable.
- Card 3: XTB xStation login + order placement with mandatory SL
  (Playwright). Must register a `portfoliomind.xtb.runner.run_morning`
  callable.
- Strategy engine, forecast models, signal scoring, KB application.
- Real trading logic, scoring weights, R/R checks, KB-5
  disqualifications.
- Paper-to-live account migration.

## Failure alerting (card 4)

The morning and returns jobs both log a one-line `summary_line()` to
the agent's structured logger on every run. The cron wrapper (or the
`--daemon` parent process) is expected to surface this line to the
operator's Discord home channel so silent misses are impossible. The
operator pattern is:

- `morning_run` status `ran` → "✅ morning OK: picks=N orders=M"
- `morning_run` status `failed` → "❌ morning FAIL: N errors, first=…"
- `morning_run` status `skipped_weekend` → log at DEBUG (not
  delivered — operator doesn't want a Friday-spam about the weekend)
- `morning_run` status `no_platform_modules` → log at INFO once per
  first-run, not a failure
