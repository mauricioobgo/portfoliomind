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

## Card 2/3/4 runner integration seam

The morning job (`portfoliomind.scheduler.jobs.morning_run`) imports
the card 2 and card 3 runners **lazily** at run time, so the lazy
import + the runner module coexist cleanly. Each platform runner
exposes a callable named `run_morning` at
`portfoliomind.investingpro.runner` (card 2) and
`portfoliomind.xtb.runner` (card 3) with this signature:

```python
def run_morning(ctx: MorningContext) -> MorningResult: ...
```

`MorningContext` carries the config, a sheets client, the sheet ID,
the current Bogota timestamp, and a `log_to_sheet(level, message)`
helper. `MorningResult` is a dataclass with `picks_scraped`,
`orders_placed`, `skipped`, `skip_reason`, and `error` fields.

When both runner modules are present, the morning job picks them up
automatically. The "no platform runners registered" line is no
longer logged on a normal weekday morning; the morning job calls
both runners, aggregates their `picks_scraped` and `orders_placed`
counters, and surfaces the first error to the scheduler alert.

### Card 2 runner — `portfoliomind.investingpro.runner`

Composes `login` → `scrape_ai_picks` → `deepdive_top_n` into one
`run_morning` callable. Idempotent within a Bogota-local day via a
date-pinned `scraped_at` (`YYYY-MM-DDT08:30:00-05:00`) so two
morning_run calls in the same day produce the same dedup key. Never
raises — every failure mode is converted into a `MorningResult` with
the `error` field set. The runner uses a test injection seam
(`set_factories` / `reset_factories`) so unit tests can swap in
in-memory fakes for login / scrape / deep-dive without touching a
real Playwright browser.

### Card 3 runner — `portfoliomind.xtb.runner`

Reads `APPROVED_TRADES`, and for each row:

* Pre-validates via `OrderSpec.checked` (the SL/TP iron rules). A
  failure is logged to `EXECUTED_ORDERS` with status
  `VALIDATION_FAILED` and the batch continues.
* Dedups against `EXECUTED_ORDERS` — a `(Ticker, Timestamp)` pair
  already present is skipped.
* In dry-run mode (the default — `config.xtb_dry_run=True`), the
  runner does NOT open a browser; it writes a `DRY_RUN` row to
  `EXECUTED_ORDERS` per trade.
* In live mode (`xtb_dry_run=False` AND `xtb_live_confirm=True`),
  the runner opens a persistent Playwright context, logs in to
  xStation once, calls `place_order` per spec, and writes a
  `PLACED` (order ID parsed from the confirmation modal) or
  `UNCONFIRMED` (submit succeeded but we couldn't read the ID)
  row.

The two-toggle gate (both `xtb_dry_run=False` AND
`xtb_live_confirm=True`) is the only path that ever moves real
money; flipping only one keeps the runner in dry-run mode.

If `APPROVED_TRADES` is empty, the runner returns
`MorningResult(skipped=True, skip_reason="no approved trades")` and
does not open a browser.

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

- Card 2: InvestingPro runner (`portfoliomind.investingpro.runner`) —
  done. Wires login + scrape + deep-dive into the morning-run seam.
- Card 3: XTB runner (`portfoliomind.xtb.runner`) — done. Reads
  `APPROVED_TRADES`, places each order (dry-run by default), writes
  `EXECUTED_ORDERS`. Honors the two-toggle `xtb_dry_run` /
  `xtb_live_confirm` gate.
- Strategy engine, forecast models, signal scoring, KB application.
- Real trading logic, scoring weights, R/R checks, KB-5
  disqualifications.
- Paper-to-live account migration.

## Card 6 — Technical + combined signal (portfoliomind.signals)

The signal module computes 4 technical indicators per ticker (via
yfinance + the `ta`-equivalent pure-Python math) and combines them
with the card 5 news sentiment.

Public surface (importable as the contract for card 7):

```python
from portfoliomind.signals import (
    Candidate,                 # the public dataclass for card 7
    TechnicalSignal,           # the 4 booleans + underlying numbers
    compute_technical_signal,  # ticker → TechnicalSignal (with yfinance + cache)
    score_universe,            # tickers → top-N Candidate (AND-of-two gate)
    PriceCache,                # the SQLite price cache
    MIN_TECHNICAL_BULLISH,     # 2
    MIN_NEWS_SENTIMENT,        # +0.2
    WEIGHT_TECHNICAL,          # 0.6
    WEIGHT_NEWS,               # 0.4
    STRATEGY,                  # "swing-bullish-news"
    TIMEFRAME,                 # "swing"
)
```

The `Candidate` dataclass is the contract for card 7 (sizer +
Discord approval): `ticker, strategy, timeframe, entry_price,
technical_score (0-1), news_score (-1, +1), combined_score,
top_signal_reason, technical_signal`.

The four indicators:

* **50/200 SMA golden cross** — SMA(50) > SMA(200). Long-term trend.
* **20-day high breakout** — today's close > max of the prior 20
  closes. Price action.
* **MACD bullish crossover (12/26/9)** — MACD line > signal line.
* **RSI(14) not overbought** — RSI < 70. A precondition, not a
  signal in itself.

The AND-of-two gate: a ticker is a candidate iff
`bullish_count >= 2` AND `news_score > +0.2`. The combined score
weights are `0.6 * (bullish_count/4) + 0.4 * news_score`.

Key invariants:

* yfinance is cached for 1 hour (`PRICE_TTL_SECONDS`) — the morning
  run does not re-pull within the session.
* The price cache is keyed by `(ticker, as_of_date)` so a backtest
  re-run with a different date re-computes (no look-ahead bias).
* yfinance returns the CURRENT (partial) bar with NaN close; the
  signal drops it and uses the last fully-closed bar. This is the
  "yesterday's close" semantics the spec calls for.
* A yfinance failure for ONE ticker logs WARNING and skips — it
  never breaks the whole universe. The combined gate drops the
  empty-signal sentinel (bullish_count=0, close=0.0) automatically.
* An LLM sentiment failure falls back to 0.0 for every ticker,
  which means the news gate drops everyone. The morning run still
  completes cleanly.
* The Candidate dataclass is `frozen=True` so an operator
  reviewing the list cannot accidentally mutate it.

The demo (`scripts/demo_signals.py`) prints the top-N candidates
with their score components. Run it with `--as-of-date YYYY-MM-DD`
to backtest a specific trading day.

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
