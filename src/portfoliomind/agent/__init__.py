"""LLM agent layer: the operating prompt + the skill registry.

Public surface::

    from portfoliomind.agent import (
        AGENT_MODEL,
        AGENT_SYSTEM_PROMPT,
        build_system_prompt,
        SKILLS,
        to_openai_tools,
        invoke_skill,
    )

The prompt (:mod:`portfoliomind.agent.prompt`) tells the LLM what it
is, which accounts it may log in to (Google Sheets via service
account, InvestingPro, XTB xStation), and the hard guardrails. The
skills (:mod:`portfoliomind.agent.skills`) are the only actions the
LLM can take — each wires into an existing, tested PortfolioMind
module. The agent loop lives in ``scripts/run_agent.py``.
"""

from __future__ import annotations

from .prompt import AGENT_MODEL, AGENT_SYSTEM_PROMPT, build_system_prompt
from .skills import SKILLS, AgentSkill, get_skill, invoke_skill, to_openai_tools
from .validator import (
    VALIDATOR_MODEL,
    VALIDATOR_SKILLS,
    VALIDATOR_SYSTEM_PROMPT,
    invoke_validator_skill,
    validator_tools,
)

__all__ = [
    "AGENT_MODEL",
    "AGENT_SYSTEM_PROMPT",
    "build_system_prompt",
    "AgentSkill",
    "SKILLS",
    "get_skill",
    "invoke_skill",
    "to_openai_tools",
    # Independent validation agent (card 10)
    "VALIDATOR_MODEL",
    "VALIDATOR_SYSTEM_PROMPT",
    "VALIDATOR_SKILLS",
    "validator_tools",
    "invoke_validator_skill",
]
