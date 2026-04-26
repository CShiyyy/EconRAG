"""Conviction multiplier for the Position Sizing Engine.

Maps Agent C narrative_score (0–10) to a weight multiplier applied to
Agent B's mean-variance target_weight. A score of 5 is neutral (1.0×).
"""
from __future__ import annotations


def narrative_multiplier(score: float) -> float:
    """Map Agent C narrative_score (0–10) to a weight multiplier [0.5, 1.5].

    score=0  → 0.5× (strong negative narrative dampens Agent B weight by half)
    score=5  → 1.0× (neutral — Agent B weight unchanged)
    score=10 → 1.5× (strong positive narrative amplifies Agent B weight by 50%)
    """
    return 0.5 + max(0.0, min(10.0, score)) / 10.0
