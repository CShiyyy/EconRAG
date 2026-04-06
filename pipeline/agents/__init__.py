"""Agents package — Agent A (local LLM) and Agent B (deterministic quant)."""

from pipeline.agents.agent_a import run_agent_a
from pipeline.agents.agent_b import run_agent_b

__all__ = ["run_agent_a", "run_agent_b"]
