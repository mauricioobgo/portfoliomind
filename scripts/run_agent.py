#!/usr/bin/env python
"""Run the PortfolioMind LLM agent loop.

The agent gets the operating prompt (mission, accounts, guardrails)
and the skill registry as OpenAI function-calling tools, then drives
the morning workflow autonomously: connect to Google Sheets, read the
operator's Suggestions mandate, scan the universe for bullish
patterns + news, size and propose trades, and (dry-run by default)
hand the approved batch to the XTB executor.

Usage:

    uv run python scripts/run_agent.py
    uv run python scripts/run_agent.py --goal "Review NVDA only and tell me if it qualifies"
    uv run python scripts/run_agent.py --max-rounds 12 --log-level DEBUG

The two-toggle live gate applies regardless of what the model asks
for: with ``xtb_dry_run=True`` (the default) no real order is ever
placed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfoliomind.agent import AGENT_MODEL, build_system_prompt, invoke_skill, to_openai_tools
from portfoliomind.config import PortfoliomindConfig
from portfoliomind.logging_setup import get_logger, setup_logging

log = get_logger("scripts.run_agent")

DEFAULT_GOAL = (
    "Run the morning workflow: connect to the sheet, read the suggestions "
    "mandate, scan for bullish setups, propose and persist approved trades, "
    "then execute (dry-run unless live mode is enabled). Log every decision."
)


def run_agent_loop(*, goal: str, max_rounds: int, model: str) -> int:
    """Drive the OpenAI tool-use loop until the model finishes or rounds run out.

    Returns a process exit code (0 = clean finish, 4 = round budget hit).
    """
    from openai import OpenAI

    config = PortfoliomindConfig.from_env()
    client = OpenAI(api_key=config.openai_api_key)
    tools = to_openai_tools()
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": goal},
    ]

    for round_no in range(1, max_rounds + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])] or None,
            }
        )

        if not msg.tool_calls:
            # The model is done — its final message is the run report.
            print(msg.content or "(agent finished without a report)")
            return 0

        for tc in msg.tool_calls:
            name = tc.function.name
            log.info("agent_round=%d skill=%s", round_no, name)
            result = invoke_skill(name, tc.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                }
            )

    log.error("agent hit the round budget (%d) without finishing", max_rounds)
    return 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="What to ask the agent to do")
    parser.add_argument("--max-rounds", type=int, default=16, help="Tool-loop round budget")
    parser.add_argument("--model", default=AGENT_MODEL, help="OpenAI model for the agent loop")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    return run_agent_loop(goal=args.goal, max_rounds=args.max_rounds, model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
