# PortfolioMind v4

Multi-strategy equity investing agent focused on **long-only bullish-pattern
setups**. Pulls AI picks from InvestingPro, scans the universe for bullish
chart patterns with a probabilistic (log-odds) scoring model, vetoes setups
on negative news sentiment, sizes positions with fractional Kelly, executes
operator-mandated orders via XTB xStation 5, and tracks the full lifecycle
in a Google Sheet.

The strategy pipeline is fully wired: technical indicators + bullish
patterns + LLM news sentiment → candidate gates → fractional-Kelly sizing →
Suggestions-mandate approval → XTB execution (dry-run by default). An LLM
agent layer (`portfoliomind.agent`) carries the operating prompt and the
skill registry that lets a tool-calling model drive the whole workflow —
logging in to Google Sheets (service account), InvestingPro, and XTB on the
operator's behalf, strictly inside hard guardrails.

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
| `OPENAI_API_KEY` | yes | news sentiment (card 5) + the LLM agent loop |
| `SESSION_DIR` | no (default `./sessions`) | card 2 (Playwright cookies) |
| `SCREENSHOT_DIR` | no (default `./screenshots`) | card 3 (pre-trade screenshots) |
| `PORTFOLIOMIND_EQUITY` | no (default `10000`) | fractional-Kelly position sizer |

`GOOGLE_SERVICE_ACCOUNT_JSON` accepts either an absolute path to a JSON file
or the raw JSON contents inline. Path form is recommended.

## Usage

```bash
# 1) Create the sheet (or verify the existing one) and ensure all 12 tabs.
uv run python scripts/bootstrap_sheet.py

# 2) Smoke-test the Sheets integration: write one timestamped row to each tab.
uv run python scripts/dry_run.py

# 3) Run the primary LLM agent loop (dry-run by default — see the two-toggle gate).
uv run python scripts/run_agent.py

# 4) Run the INDEPENDENT validator agent — re-checks the proposed trades and
#    brings you an APPROVE / FLAG / REJECT report for the final call.
uv run python scripts/run_validator.py

# 5) Backtest the bullish-pattern strategy (whole universe, or named tickers).
uv run python scripts/backtest.py
uv run python scripts/backtest.py AAPL MSFT NVDA --period 5y
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
| 💡 Suggestions | The operator's standing mandate: tickers the agent MAY buy, with allocation caps | Operator-edited |

## The bullish-pattern strategy

The morning strategy run (`portfoliomind.signals.combined.score_universe`)
qualifies long-only candidates through four gates:

1. **Bullish-tech gate** — positive technical score (SMA trend + RSI
   momentum + vol regime).
2. **Pattern gate** — `portfoliomind.signals.patterns` scans for golden
   cross, 55-day breakout, RSI oversold recovery, MACD bull cross, higher
   lows, pullback bounce, and uptrend stack. Detected patterns are folded
   into a posterior probability of upside `p_bullish` via shrunk log-odds
   aggregation; the gate requires `p_bullish ≥ 0.55`.
3. **Positive-news gate** — LLM news sentiment must not be negative.
4. **Strength gate** — the blended score (0.40 technical + 0.35 patterns
   + 0.25 sentiment) must clear 0.15.

Qualified candidates are sized by `portfoliomind.signals.sizer.PositionSizer`
(quarter-Kelly on `p_bullish` at 2:1 reward:risk, 3σ vol-anchored stops,
hard 10% per-position cap), then matched against the **💡 Suggestions**
mandate: a trade is approved only when its ticker has an `ACTIVE` `BUY`
row, clamped to the row's `Max Allocation ($)`. Approved trades flow to
✅ Approved Trades and the XTB executor (dry-run unless the operator
enables the two-toggle live gate).

## The LLM agent

`portfoliomind.agent` ships the operating prompt
(`AGENT_SYSTEM_PROMPT` — mission, authorized accounts, probabilistic
reasoning rules, hard guardrails) and a 10-skill registry
(`connect_google_sheets`, `login_investingpro`, `login_xtb`,
`read_suggestions`, `scan_bullish_patterns`, `analyze_news`,
`score_universe`, `propose_trades`, `execute_approved_trades`,
`log_action`) exposed as OpenAI function-calling tools.
`scripts/run_agent.py` runs the tool loop. Guardrails the prompt and the
code both enforce: long-only, mandate-only, SL/TP mandatory, the agent
cannot enable live trading itself, secrets never logged, every decision
audited to 🗒️ Agent Log.

## Backtesting

`portfoliomind.backtest` walk-forward replays each ticker's historical
closes through the same pattern detector and vol-anchored stops the live
strategy uses, with no look-ahead. The headline metric is the
**calibration gap** — how far the model's claimed `p_bullish` sat above
the realized win rate — which is the empirical check on the pattern
hit-rate priors. It also reports win rate, expectancy, profit factor, max
drawdown, and a per-pattern win-rate breakdown so you can see which setups
actually pay. `scripts/backtest.py` runs a single ticker or sweeps the
whole universe. The engine is pure Python over a list of closes (`fetch`
injected), so it is hermetic in tests.

## The independent validator agent

`scripts/run_validator.py` runs a **second, separate agent** that executes
*after* the primary agent has done the news + technical analysis and
proposed sized trades. It is a deliberate second set of eyes with
separation of duties:

- It re-derives the evidence instead of trusting the primary scores:
  re-checks news sentiment, runs a backtest to confirm the pattern has a
  positive out-of-sample edge, and checks reward:risk and concentration
  (`portfoliomind.validation`).
- It produces a per-trade **APPROVE / FLAG / REJECT** verdict. Hard checks
  (broken SL/TP, negative news, negative historical edge, over the
  concentration cap) reject; soft concerns (thin backtest sample, weak
  R:R, overconfident probability vs. backtest) flag.
- It **cannot execute** — its skill registry has no execution skill by
  design. It records every verdict to 🗒️ Agent Log (REJECTs also to
  🚫 Disqualified) and presents the report to you for the final go/no-go.
  The validator advises; you decide.

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
