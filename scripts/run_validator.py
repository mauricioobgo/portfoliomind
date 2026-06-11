#!/usr/bin/env python
"""Run the INDEPENDENT validation agent.

This is a separate agent from ``scripts/run_agent.py``. The primary
agent finds and proposes trades; this agent runs afterward and
independently double-checks each one — re-running the news analysis,
backtesting the pattern's historical edge, and checking reward:risk
and concentration — then presents an APPROVE / FLAG / REJECT report to
the user for the final go/no-go.

It has no execution skill by design: separation of duties means the
validator advises, the user decides.

Usage:

    uv run python scripts/run_validator.py
    uv run python scripts/run_validator.py --goal "Validate only the NVDA proposal"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfoliomind.agent import (
    VALIDATOR_MODEL,
    VALIDATOR_SYSTEM_PROMPT,
    invoke_validator_skill,
    validator_tools,
)
from portfoliomind.config import PortfoliomindConfig
from portfoliomind.logging_setup import get_logger, setup_logging

log = get_logger("scripts.run_validator")

DEFAULT_GOAL = (
    "Read the proposed trades, independently validate each one (backtest the "
    "edge, re-check the news, check reward:risk and concentration), record the "
    "verdicts, and give me a clear APPROVE / FLAG / REJECT summary with the "
    "decisive reasons. Then ask me which trades to proceed with."
)


def run_validator_loop(*, goal: str, max_rounds: int, model: str) -> int:
    from openai import OpenAI

    config = PortfoliomindConfig.from_env()
    client = OpenAI(api_key=config.openai_api_key)
    tools = validator_tools()
    messages: list[dict] = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    for round_no in range(1, max_rounds + 1):
        response = client.chat.completions.create(model=model, messages=messages, tools=tools)
        msg = response.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])] or None,
            }
        )
        if not msg.tool_calls:
            print(msg.content or "(validator finished without a report)")
            return 0
        for tc in msg.tool_calls:
            log.info("validator_round=%d skill=%s", round_no, tc.function.name)
            result = invoke_validator_skill(tc.function.name, tc.function.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)}
            )

    log.error("validator hit the round budget (%d) without finishing", max_rounds)
    return 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--model", default=VALIDATOR_MODEL)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    return run_validator_loop(goal=args.goal, max_rounds=args.max_rounds, model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
