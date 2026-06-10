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

## Card 5/6 runner — `portfoliomind.signals`

Card 6 builds a single combined signal per ticker: technical
indicators (trend + momentum + volatility) + news sentiment (card
5's `score_ticker_sentiment`). The package is split across three
modules so each piece is testable in isolation:

* `portfoliomind.signals.technicals` — pure-math SMA ratio, RSI(14),
  realized-vol regime → `TechnicalScore` in [-1, +1]. The yfinance
  fetch lives here too (`fetch_ohlcv`) and is the only network call
  in the package. All OHLCV logs are DEBUG-only (no raw OHLCV at
  INFO per the card 6 spec).
* `portfoliomind.signals.cache` — same-day idempotency layer
  wrapping `NewsCache`. Reads/writes the `technical_cache` table
  (added by the v1 → v2 migration in
  :mod:`portfoliomind.news.store`). The cache shares the same
  SQLite file as the news cache, so a single backup captures both
  tables.
* `portfoliomind.signals.combiner` — the public entry point
  consumed by card 7/8: `score_ticker(ticker) -> Signal` and
  `score_universe() -> list[Signal]`. **Never raises** — every
  failure mode is converted into a `Signal` with `combined=0.0` and
  the error string in `reasons`. Card 7/8 depend on this contract.

### Public API (card 6)

```python
from portfoliomind.signals import (
    Signal,                       # the combined output dataclass
    TechnicalScore,               # the technical-only dataclass
    score_ticker,                 # single-ticker entry, never raises
    score_universe,               # full UNIVERSE, sorted by combined desc
    compute_technical_score,      # pure-math, takes closes list
    TechnicalCache,               # same-day idempotency layer
)
```

### Combine math

* `combined = 0.6 * technical.score + 0.4 * sentiment` (weights
  in `portfoliomind.signals.combiner.WEIGHT_TECHNICAL` /
  `WEIGHT_SENTIMENT`).
* `confidence = |combined| * (1 - |technical - sentiment|)` — high
  when components agree AND the combined signal is large. A signal
  where the components contradict has near-zero confidence and
  should be filtered out before any operator-facing ping.
* `TechnicalScore.score = 0.5*trend + 0.3*momentum + 0.2*volatility`
  (weights in `portfoliomind.signals.technicals.WEIGHT_TREND` /
  `WEIGHT_MOMENTUM` / `WEIGHT_VOLATILITY`).

### Idempotency contract

A re-run of `score_universe()` in the same Bogota day returns
**identical** `Signal` objects for every ticker — the technical
cache is keyed on `(ticker, asof_date)` and the sentiment cache is
keyed on `(ticker, day)`; both are day-pinned in Bogota time. A run
after midnight Bogota triggers a fresh yfinance fetch + (if not
cached) a fresh LLM sentiment call.

### Cache schema migration (v1 → v2)

The `technical_cache` table was added in card 6. The
`portfoliomind.news.store.NewsCache` runs a forward-only migration
on open: if it sees a v1 DB, it stamps the version row to v2 and
the new table is created on the same connection. A v2+ DB that the
current code can't read is rejected with a clear `NewsCacheError`.

### Handoff to card 7

Card 7 (Discord approval) will import `score_universe`, pick the
top-N bullish + bottom-N bearish signals, and post them to the
operator's Discord home channel with approve/reject buttons. The
`confidence` field is what determines whether a signal warrants a
Discord ping — card 7 should filter out `confidence < some_threshold`
signals before posting. The default combine weights (0.6 / 0.4)
live in three module-level constants so card 7 can re-weight
without forking the package.

## Card 7 — Position sizer + Discord approval

Card 7 takes a card-6 `Signal` and turns it into a `TradeOrder`
(qty, entry, SL, TP, notional, commission_rt, r_r_ratio) or a
`RejectReason`. The flow is split across four small modules so
each piece is testable in isolation:

* `portfoliomind.signals.sizer` — the `PositionSizer` class.
  Pure-math; never raises (any failure is converted into a
  `RejectReason`). Rules: per-trade dollar cap (`xtb_per_trade_cap`,
  default $200), max open positions (default 5), commission
  rejection (round-trip > 5% of notional), 2:1 R/R floor. Stocks
  are whole-share, ETFs can be fractional. Construct via
  `PositionSizer.from_config(config)` for the standard path.
* `portfoliomind.signals.commissions` — `XTBCommissionModel`.
  Hardcoded US schedule: stocks `max($8, 0.08% × notional)`,
  ETFs 0% under $100k monthly volume, 0.08% above. `round_trip =
  2 × one_way`. Source URL in the docstring; refresh on schedule
  changes. `InstrumentType` is a `str`-valued Enum (not a free-form
  string) so the value round-trips through JSON / Sheets.
* `portfoliomind.signals.last_close` — small yfinance helper.
  `last_close(ticker) -> Optional[float]`. Returns `None` on any
  failure; the sizer converts that into a `RejectReason`. Not
  cached — the sizer wants the latest close, separate from card 6's
  day-pinned technical-cache view.
* `portfoliomind.approval` — the operator-facing approval package.
  Split into two modules:
  - `portfoliomind.approval.discord` — `post_candidates_and_collect_reactions(candidates, *, timeout_seconds=1800, ...) -> ApprovalOutcome`. Posts one
    message to the operator's home thread (configured via
    `DISCORD_HOME_CHANNEL_THREAD_ID`), adds per-candidate
    1️⃣-5️⃣ reactions, polls for ✅/❌/⏸ reactions, and treats
    timeout-as-reject for the per-candidate slots. **Never raises.**
  - `portfoliomind.approval.persist` — `persist_approved_trades(outcome, sheets, *, sheet_id, dry_run=False) -> PersistResult`. Reads existing
    `APPROVED_TRADES`, dedups on `(Ticker, signal_date, Entry Price)`,
    appends the new rows. **Never raises.**

### Public API (card 7)

```python
from portfoliomind.signals.sizer import PositionSizer, TradeOrder, RejectReason
from portfoliomind.signals.commissions import XTBCommissionModel, InstrumentType
from portfoliomind.signals.last_close import last_close
from portfoliomind.approval import (
    post_candidates_and_collect_reactions,
    persist_approved_trades,
    ApprovalOutcome,
    ApprovedTrade,
    RejectedTrade,
    WaitedTrade,
    PersistResult,
)
```

### Sizing rules (operator-approved 2026-06-08)

* Per-trade dollar cap: $200 per ticker (`xtb_per_trade_cap`).
* Max open positions: 5 (`xtb_max_open_positions`).
* Commission rejection: round-trip > 5% of position value
  (`xtb_max_commission_pct`).
* Stop-loss: 7% below entry (default; `xtb_sl_pct`).
* Take-profit: 14% above entry (default; `xtb_tp_pct`).
* Long-only. Negative `combined` already filtered by card 6.

### Approval flow (Discord interactive)

1. Build one operator-readable message (one line per candidate).
2. POST to the home thread.
3. Add 1️⃣-5️⃣ reactions (one per candidate, capped at 5).
4. Poll the message's reactions for up to 30 min
   (`APPROVAL_TIMEOUT_MIN`).
5. ✅ globally → all approved; ❌ globally → all rejected;
   ⏸ globally → all waited; per-candidate 1️⃣-5️⃣ → that one approved.
6. Anything not approved by timeout is rejected (logged to
   `AGENT_LOG`).
7. Approved subset → `APPROVED_TRADES` via `SheetsClient.append_rows`.

### Dedup key

`persist_approved_trades` keys on the triple
`(Ticker, signal_date, Entry Price)`. The `signal_date` is
recovered from the row's Approval Note (`signal_date=YYYY-MM-DD`),
not the row's wall-clock Timestamp — so a re-run with the same
data on the same day produces zero new rows.

### Failure isolation

* Discord post fails → outcome has `error` set, but the runner
  continues. `persist_approved_trades` runs anyway in case partial
  reactions are present.
* Sheets read fails → result has `error` set, no rows appended.
  A "tab not found" is treated as empty (first-ever run).
* Sheets append fails → result has `error` set, no rows appended.
* `PositionSizer.size` raises → converted to `RejectReason`. The
  runner (card 8) logs and continues with the next candidate.
* `last_close` returns `None` (yfinance down) → sizer rejects
  with "entry price unavailable".

### Iron rules (sizer + approval)

1. **Stop-loss is mandatory.** The sizer always sets `sl =
   entry * (1 - xtb_sl_pct)`. A misconfigured `xtb_sl_pct` (≤ 0 or
   ≥ 1) raises at construction time, never silently produces a
   zero-stop order.
2. **Long-only.** The sizer does not check direction (card 6
   already filters `combined > 0`). It will *not* size a short
   sell even if called with a negative `combined` — the resulting
   qty and SL/TP math would be wrong, and the upstream filter
   is the contract.
3. **Never raise.** Every public function in the card 7 surface
   (`size`, `post_candidates_and_collect_reactions`,
   `persist_approved_trades`, `last_close`) catches all
   exceptions and returns a result/error object. The strategy
   runner (card 8) depends on this.

## Card 2 runner — `portfoliomind.investingpro.runner`

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

| Job                 | Cadence                  | Bogota time | UTC cron          | Action                                                                 |
|---------------------|--------------------------|-------------|-------------------|------------------------------------------------------------------------|
| `morning_run`       | Mon–Fri (skip weekends)  | 08:30       | `30 13 * * 1-5`   | Strategy picks → operator approval → XTB execution                     |
| `refresh_returns`   | Daily                    | 16:30       | `30 21 * * *`     | yfinance lookup for every row in RETURNS_TRACKER; update + prune       |

Both triggers are pinned to `zoneinfo.ZoneInfo("America/Bogota")` —
NOT the host's local time. The production Docker image runs UTC, so
the cron expressions are the only thing keeping the schedule on
Colombia time.

**Card 8 added a third layer to `morning_run` — the strategy
runner.** The full morning pipeline is now:

1. **Card 2 (InvestingPro scrape)** — runs first, reads the AI
   Picks tab from InvestingPro.
2. **Card 8 (strategy runner, `portfoliomind.strategy_runner`)** —
   runs second: scores the universe (card 6), sizes the
   candidates (card 7), posts to Discord for operator approval
   (card 7), and persists the approved subset to
   `APPROVED_TRADES`.
3. **Card 3 (XTB execution)** — runs last, reads `APPROVED_TRADES`
   and places the orders (dry-run by default).

The strategy runner is lazy-imported with the same pattern as cards
2 and 3 — if the card-6/7 modules are not yet on the import path,
the strategy runner returns `status='not_implemented'` cleanly and
the morning job continues. Card 8 therefore ships safely ahead of
cards 6 and 7.

The morning job also consults `PORTFOLIOMIND_HOLIDAYS` (comma-separated
`YYYY-MM-DD` list) to skip configured market holidays. Default = no
holidays (skip = false for every day except Sat/Sun).

### `morning_cron` override

The `ScheduleConfig` dataclass exposes a `morning_cron` field that
overrides the default 08:30 Bogota Mon–Fri schedule with a raw
5-field cron expression in UTC. Default = `""` (empty string),
which falls through to the `morning_hour`/`morning_minute` path
in `America/Bogota` with `day_of_week='mon-fri'`. Set
`morning_cron` to a non-empty value to take over.

```python
from portfoliomind.scheduler.loop import ScheduleConfig

# Default: 08:30 Bogota Mon-Fri via America/Bogota timezone.
cfg = ScheduleConfig()

# Override: 13:30 UTC Mon-Fri (= 08:30 Bogota, expressed in UTC
# because the container runs UTC). The day_of_week is encoded in
# the cron string itself.
cfg = ScheduleConfig(morning_cron="30 13 * * 1-5")
```

The CLI exposes this as `--morning-cron`:

```bash
uv run python scripts/run_scheduler.py --daemon --morning-cron "30 13 * * 1-5"
```

The `DEFAULT_MORNING_CRON` constant is the canonical
"08:30 Bogota Mon-Fri in UTC" expression. Reference it from the
`scripts/register_cron.sh` registration script as the single
source of truth — don't duplicate the string in docs and shell
scripts.

The container is UTC; Colombia does not observe DST, so the
offset is fixed at UTC-5 year-round. Operators in other
timezones should adjust `morning_cron` accordingly and document
the local-time equivalent in their AGENTS.md.

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

### `scripts/register_cron.sh` — operator runs once

The operator runs `scripts/register_cron.sh` once after deploying
to register the scheduler under the `portfoliomind` Hermes
profile. The script prints the exact `hermes cron create` command
it would run and (with `--apply`) actually registers it.

```bash
# Dry-run: print the command without running it.
bash scripts/register_cron.sh

# Show a different cron expression.
bash scripts/register_cron.sh --morning-cron "30 13 * * 1-5"

# Actually register the cron job.
bash scripts/register_cron.sh --apply
```

The script sources the canonical cron string from
`DEFAULT_MORNING_CRON` via `python -c`, so the registration
command and the in-process scheduler always agree. Don't hardcode
the cron string in the shell — let the script import it.

### Disabling a single trigger

The cron job above registers a single daemon process that runs
both jobs. To disable a single trigger without killing the
daemon, override the `ScheduleConfig`:

```bash
# Run the daemon with no morning trigger; returns refresh still fires.
uv run python scripts/run_scheduler.py --daemon --morning-hh 99 --morning-mm 99
# (Note: APScheduler treats hour=99 as "never fire", so the morning
#  job is effectively disabled. A cleaner approach is to set
#  --morning-cron to a cron that never matches, e.g. "0 0 31 2 *".)
```

The "right" way to disable a single trigger is to use a cron
expression that never matches, e.g. `0 0 31 2 *` (Feb 31 doesn't
exist). The CLI flag `--morning-cron` takes any 5-field
expression. The returns job is disabled the same way via
`--returns-hh` / `--returns-mm` (no `--returns-cron` flag yet —
add one in a follow-up card if needed).

If the daemon is running under `hermes cron` and you need to
disable the whole pipeline (both triggers), the right knob is:

```bash
hermes cron pause portfoliomind-scheduler
```

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

## Things still in scope for future cards (not done in cards 1-8)

- Card 2: InvestingPro runner (`portfoliomind.investingpro.runner`) —
  done. Wires login + scrape + deep-dive into the morning-run seam.
- Card 3: XTB runner (`portfoliomind.xtb.runner`) — done. Reads
  `APPROVED_TRADES`, places each order (dry-run by default), writes
  `EXECUTED_ORDERS`. Honors the two-toggle `xtb_dry_run` /
  `xtb_live_confirm` gate.
- Card 5: News + LLM sentiment scoring (`portfoliomind.news.*`) —
  done. RSS ingestion + 4o-mini scoring. Output: `score_ticker_sentiment`
  in [-1, +1] per ticker per Bogota day, cached for same-day
  idempotency.
- Card 6: Combined strategy signal (`portfoliomind.signals.*`) —
  done. Technical indicators (SMA ratio, RSI, vol regime) + news
  sentiment → single `Signal(ticker, combined, technical, sentiment,
  confidence, reasons)` in [-1, +1]. Card 7 (Discord approval) and
  card 8 (morning-run wiring) consume this.
- Card 7: Position sizer + Discord interactive approval
  (`portfoliomind.signals.sizer`, `portfoliomind.signals.commissions`,
  `portfoliomind.signals.last_close`, `portfoliomind.approval.*`) —
  done. Sizer converts card-6 Signals into TradeOrders (qty, entry,
  SL, TP, notional, commission_rt, r_r_ratio) honoring the
  per-trade cap, max open positions, commission rejection (>$5% of
  notional), and 2:1 R/R floor. Discord approval posts the batch
  to the operator's home thread, collects ✅/❌/⏸ reactions (and
  per-candidate 1️⃣-5️⃣ taps), and writes the approved subset to
  `APPROVED_TRADES` with `(ticker, signal_date, entry_price)` dedup.
  XTB commission model is hardcoded in `signals.commissions` with
  the source URL in the docstring; refresh the constants on schedule
  changes.
- Card 8 (this card): Strategy runner wiring — done. The
  runner is the integration seam that will activate automatically
  once card 7 lands.
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
