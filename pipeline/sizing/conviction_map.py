"""Conviction map (deprecated).

Historically this module mapped Agent C's `narrative_score` (0–10) to a 0.5×–1.5×
multiplier applied to Agent B's mean-variance `target_weight`. As of branch
`rationale-model-agent-b`, weights are 100% deterministic from Agent B; LLMs
(Agents A and C) only annotate via rationale and never scale numbers.

The module is kept as an empty shim so any straggling imports fail loudly
at import time rather than silently re-introducing the multiplier.
"""
