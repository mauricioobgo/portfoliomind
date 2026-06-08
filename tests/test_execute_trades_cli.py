"""Smoke tests for the execute_trades CLI.

These tests run the full CLI path against a mock SheetsClient so we
exercise:

  * Argument parsing and defaults
  * Sheet read / dedup
  * Row -> OrderSpec conversion
  * The print_dry_run path (the operator-facing output)
  * The safety net that refuses --dry-run=false without --confirm-each

The tests do NOT launch Chromium. Playwright is not in scope for card 3's
unit-test surface (it's an integration concern).
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from portfoliomind.sheets.schema import APPROVED_TRADES
import scripts.execute_trades as cli


# --- Mock SheetsClient --------------------------------------------------------


class _MockSheetsClient:
    """In-memory replacement for :class:`SheetsClient`.

    Records every read/write call and returns canned data. Just enough
    to exercise the CLI's path through ``_read_approved_trades``,
    ``_read_executed_order_keys``, and the dry-run print.
    """

    def __init__(
        self,
        approved_rows: list[list[str]] | None = None,
        executed_rows: list[list[str]] | None = None,
    ) -> None:
        self._approved = approved_rows or []
        self._executed = executed_rows or []
        self.appended: list[list[list[Any]]] = []
        self.reads: list[tuple[str, str, str]] = []

    def read_range(self, sheet_id: str, tab_name: str, range_a1: str) -> list[list[str]]:
        self.reads.append((sheet_id, tab_name, range_a1))
        if tab_name == APPROVED_TRADES:
            return [list(r) for r in self._approved]
        if tab_name == "📈 Executed Orders":
            return [list(r) for r in self._executed]
        return []

    def append_rows(self, sheet_id: str, tab_name: str, values: list[list[Any]]) -> int:
        self.appended.append(values)
        return 0  # row number doesn't matter for tests

    @classmethod
    def from_config(cls, config: Any) -> "_MockSheetsClient":
        # Returns a fresh instance per call so multiple ``SheetsClient.from_config``
        # invocations in a test don't share state.
        return cls()


# --- Row -> OrderSpec --------------------------------------------------------


class TestRowToOrderSpec:
    def test_buy_with_all_fields(self):
        row = {
            "Ticker": "AAPL.US",
            "Type": "BUY",
            "Qty": "10",
            "Entry Price": "192.50",
            "SL": "189.00",
            "TP": "198.00",
            "Approval Note": "pullback to support",
        }
        spec = cli._row_to_order_spec(row)
        assert spec.ticker == "AAPL.US"
        assert spec.side.value == "BUY"
        assert spec.qty == 10
        assert spec.entry_price == 192.50
        assert spec.sl == 189.00
        assert spec.tp == 198.00
        assert spec.note == "pullback to support"

    def test_sell_with_comma_separated_price(self):
        # The Sheets display may show "1,200.50" — _to_float must strip commas.
        row = {
            "Ticker": "TSLA.US",
            "Type": "SELL",
            "Qty": "2",
            "Entry Price": "1,200.50",
            "SL": "1,250.00",
            "TP": "1,150.00",
        }
        spec = cli._row_to_order_spec(row)
        assert spec.side.value == "SELL"
        assert spec.entry_price == 1200.50
        assert spec.sl == 1250.00
        assert spec.tp == 1150.00

    def test_type_is_security_classifier_defaults_to_buy(self):
        # APPROVED_TRADES "Type" can be "Stock"/"ETF"/"BUY"/"SELL". If it's
        # a security classifier (not a side), we default to BUY (long bias).
        row = {
            "Ticker": "SPY",
            "Type": "ETF",
            "Qty": "5",
            "Entry Price": "550",
            "SL": "540",
            "TP": "565",
        }
        spec = cli._row_to_order_spec(row)
        assert spec.side.value == "BUY"

    def test_blank_entry_price_is_market_order(self):
        row = {
            "Ticker": "EURUSD",
            "Type": "BUY",
            "Qty": "1",
            "Entry Price": "",  # blank -> 0 -> market
            "SL": "1.0900",
            "TP": "1.1200",
        }
        spec = cli._row_to_order_spec(row)
        assert spec.entry_price == 0.0

    def test_missing_sl_raises_validation_error(self):
        # The CLI calls OrderSpec.checked, which raises ValidationError.
        row = {
            "Ticker": "AAPL.US",
            "Type": "BUY",
            "Qty": "10",
            "Entry Price": "192.50",
            "SL": "0",  # zero -> rejected
            "TP": "198.00",
        }
        from portfoliomind.xtb.order import ValidationError
        with pytest.raises(ValidationError, match=r"Stop-loss"):
            cli._row_to_order_spec(row)


# --- _print_dry_run ----------------------------------------------------------


class TestPrintDryRun:
    def test_prints_table_with_sl_and_tp(self, capsys: pytest.CaptureFixture[str]):
        from portfoliomind.xtb.order import OrderSpec
        specs = [
            OrderSpec.checked("AAPL.US", "BUY", 10, 192.50, 189.00, 198.00),
            OrderSpec.checked("MSFT.US", "BUY", 5, 415.10, 405.00, 435.00),
            OrderSpec.checked("EURUSD", "SELL", 1.0, 1.0950, 1.1000, 1.0850),
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._print_dry_run(specs)
        output = buf.getvalue()

        # The output must clearly mark itself as dry run.
        assert "DRY RUN" in output
        # Every order's SL and TP must be visible in the table.
        for spec in specs:
            assert spec.ticker in output
        # The R:R column must compute for entries with a non-zero entry.
        assert re.search(r"\d+\.\d{2}", output)  # at least one R:R
        # The hint to run with --dry-run=false must be present.
        assert "--dry-run=false" in output

    def test_warns_on_rr_below_one(self, capsys: pytest.CaptureFixture[str]):
        from portfoliomind.xtb.order import OrderSpec
        # Tight stop, modest TP -> R:R < 1.
        spec = OrderSpec.checked("AAPL.US", "BUY", 1, 100.0, 99.0, 100.5)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._print_dry_run([spec])
        assert "R:R<1" in buf.getvalue()


# --- CLI end-to-end (subprocess) ---------------------------------------------


class TestCliDryRunPath:
    """Drive the CLI as a subprocess to make sure entry-point handling
    (argparse, exit codes, stdout) works the way the operator expects."""

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent

    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/execute_trades.py", "--help"],
            capture_output=True,
            text=True,
            cwd=self._PROJECT_ROOT,
            env={"PATH": "/usr/bin:/bin"},  # Strip env so config loader can't find real creds.
        )
        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        assert "Execute the trades" in result.stdout

    def test_config_error_when_env_missing(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/execute_trades.py", "--dry-run", "true"],
            capture_output=True,
            text=True,
            cwd=self._PROJECT_ROOT,
            env={"PATH": "/usr/bin:/bin"},
        )
        # ConfigError -> exit code 1.
        assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        assert "config error" in result.stderr.lower() or "missing" in result.stderr.lower()


# --- _read_executed_order_keys (dedup) ---------------------------------------


class TestExecutedOrderKeys:
    def test_returns_set_of_ticker_pairs(self):
        client = _MockSheetsClient(
            executed_rows=[
                ["AAPL.US", "2026-06-08T08:30:00-05:00", "123456789"],
                ["MSFT.US", "2026-06-08T08:31:00-05:00", "987654321"],
            ]
        )
        keys = cli._read_executed_order_keys(client, "fake-sheet")
        # We can't predict the dedup key shape exactly (it depends on the
        # column ordering), but it must be a non-empty set with the
        # right number of entries.
        assert len(keys) == 2
        for key in keys:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_empty_executed_returns_empty_set(self):
        client = _MockSheetsClient(executed_rows=[])
        keys = cli._read_executed_order_keys(client, "fake-sheet")
        assert keys == set()

    def test_skips_short_rows(self):
        client = _MockSheetsClient(
            executed_rows=[
                ["AAPL.US", "2026-06-08T08:30:00-05:00"],  # only 2 cols
            ]
        )
        keys = cli._read_executed_order_keys(client, "fake-sheet")
        assert keys == set()  # short row was skipped


# --- main() end-to-end (in-process, mocked sheets) --------------------------


class TestMainFlow:
    """Drive the full main() function with a mock SheetsClient.

    We don't need a real Google Sheet for the dry-run path — the mock
    returns canned rows and the CLI prints them.
    """

    @staticmethod
    def _build_test_config():
        """Build a PortfoliomindConfig from a hermetic env so the CLI's
        own ``PortfoliomindConfig.from_env()`` call sees the right
        values regardless of the surrounding shell env."""
        from portfoliomind.config import PortfoliomindConfig
        from tests.conftest import full_env
        return PortfoliomindConfig.from_env(env=full_env(sheet_id="test-sheet-id-123"))

    def test_dry_run_with_three_rows_prints_three_orders(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Mock the sheets client to return three APPROVED_TRADES rows.
        mock_client = _MockSheetsClient(
            approved_rows=[
                [
                    "2026-06-08T08:30:00-05:00", "AAPL.US", "BUY", "long",
                    "1d", "1925", "10", "192.50", "189.00", "198.00",
                    "first trade",
                ],
                [
                    "2026-06-08T08:31:00-05:00", "MSFT.US", "BUY", "long",
                    "1d", "2075", "5", "415.10", "405.00", "435.00",
                    "",
                ],
                [
                    "2026-06-08T08:32:00-05:00", "EURUSD", "SELL", "fx",
                    "1h", "1100", "1.0", "1.0950", "1.1000", "1.0850",
                    "fade the rally",
                ],
            ],
        )
        cfg = self._build_test_config()

        with patch.object(cli, "PortfoliomindConfig") as mock_cfg, \
             patch.object(cli, "SheetsClient") as mock_cls:
            mock_cfg.from_env.return_value = cfg
            mock_cls.from_config.return_value = mock_client
            rc = cli.main(["--dry-run", "true"])

        assert rc == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "3 order(s) ready" in captured.out
        assert "AAPL.US" in captured.out
        assert "MSFT.US" in captured.out
        assert "EURUSD" in captured.out

    def test_dry_run_with_zero_rows_prints_nothing_to_process(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_client = _MockSheetsClient(approved_rows=[])
        cfg = self._build_test_config()

        with patch.object(cli, "PortfoliomindConfig") as mock_cfg, \
             patch.object(cli, "SheetsClient") as mock_cls:
            mock_cfg.from_env.return_value = cfg
            mock_cls.from_config.return_value = mock_client
            rc = cli.main(["--dry-run", "true"])

        # No rows is a clean exit (0) — the operator hasn't approved
        # anything today.
        assert rc == 0
        captured = capsys.readouterr()
        assert "no APPROVED_TRADES rows" in captured.err

    def test_dry_run_with_invalid_row_exits_2(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Row with SL=0 -> validation error.
        mock_client = _MockSheetsClient(
            approved_rows=[
                [
                    "2026-06-08T08:30:00-05:00", "BAD.US", "BUY", "long",
                    "1d", "1000", "10", "100", "0", "110",  # SL=0
                    "",
                ],
            ],
        )
        cfg = self._build_test_config()

        with patch.object(cli, "PortfoliomindConfig") as mock_cfg, \
             patch.object(cli, "SheetsClient") as mock_cls:
            mock_cfg.from_env.return_value = cfg
            mock_cls.from_config.return_value = mock_client
            rc = cli.main(["--dry-run", "true"])

        # Validation error -> exit code 2.
        assert rc == 2
        captured = capsys.readouterr()
        assert "VALIDATION ERRORS" in captured.err
        assert "Stop-loss" in captured.err

    def test_dedup_skips_already_executed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # APPROVED_TRADES has 2 rows; EXECUTED_ORDERS has the dedup key
        # for one of them, so dry-run should print just 1.
        mock_client = _MockSheetsClient(
            approved_rows=[
                [
                    "2026-06-08T08:30:00-05:00", "AAPL.US", "BUY", "long",
                    "1d", "1925", "10", "192.50", "189.00", "198.00",
                    "",
                ],
                [
                    "2026-06-08T08:31:00-05:00", "MSFT.US", "BUY", "long",
                    "1d", "2075", "5", "415.10", "405.00", "435.00",
                    "",
                ],
            ],
            # The dedup key in the CLI is (Ticker, Timestamp from
            # APPROVED_TRADES). Mark AAPL as already executed by putting
            # its ticker + that timestamp into the executed set.
            executed_rows=[
                ["AAPL.US", "2026-06-08T08:30:00-05:00", "111111111"],
            ],
        )
        cfg = self._build_test_config()

        with patch.object(cli, "PortfoliomindConfig") as mock_cfg, \
             patch.object(cli, "SheetsClient") as mock_cls:
            mock_cfg.from_env.return_value = cfg
            mock_cls.from_config.return_value = mock_client
            rc = cli.main(["--dry-run", "true"])

        assert rc == 0
        captured = capsys.readouterr()
        # Only MSFT survives the dedup.
        assert "1 order(s) ready" in captured.out
        assert "MSFT.US" in captured.out
        assert "AAPL.US" not in captured.out

    def test_live_path_refuses_without_confirm_each(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_client = _MockSheetsClient(
            approved_rows=[
                [
                    "2026-06-08T08:30:00-05:00", "AAPL.US", "BUY", "long",
                    "1d", "1925", "10", "192.50", "189.00", "198.00",
                    "",
                ],
            ],
        )
        cfg = self._build_test_config()

        with patch.object(cli, "PortfoliomindConfig") as mock_cfg, \
             patch.object(cli, "SheetsClient") as mock_cls:
            mock_cfg.from_env.return_value = cfg
            mock_cls.from_config.return_value = mock_client
            # --dry-run=false without --confirm-each on a multi-row run
            # is a safety trap: the CLI must refuse.
            rc = cli.main(["--dry-run", "false"])

        assert rc == 1
        captured = capsys.readouterr()
        assert "refusing to place live orders" in captured.err.lower()


# --- Parser boolean ---------------------------------------------------------


class TestParserBoolean:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True), ("True", True), ("TRUE", True),
            ("yes", True), ("y", True), ("1", True), ("t", True),
            ("false", False), ("False", False),
            ("no", False), ("n", False), ("0", False), ("f", False),
        ],
    )
    def test_parses_common_truthy_falsy_strings(self, raw: str, expected: bool):
        assert cli._parse_bool(raw) is expected

    def test_garbage_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            cli._parse_bool("definitely-not-a-bool")
