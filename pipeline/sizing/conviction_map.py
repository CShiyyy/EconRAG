"""Conviction weight mapping for the Position Sizing Engine.

Maps Agent C's categorical conviction sub-scores and numeric confidence values
to numeric weights. Supports a 5-level categorical scale (very_strong, strong,
moderate, weak, very_weak) plus a numeric confidence modifier (0.0–10.0).
"""

from __future__ import annotations

VALID_SCORES = ("very_strong", "strong", "moderate", "weak", "very_weak")

# Level -> base score used in the weighted-sum formula.
_LEVEL_SCORE: dict[str, float] = {
    "very_strong": 1.0,
    "strong":      0.8,
    "moderate":    0.5,
    "weak":        0.3,
    "very_weak":   0.1,
}

# Weights for each conviction dimension in the formula.
# narrative_alignment carries the most weight on all run types.
_DIM_WEIGHTS = {
    "narrative_alignment": 0.40,
    "quant_support":       0.35,
    "signal_agreement":    0.25,
}


def _compute_table_weight(na: str, qs: str, sa: str) -> float:
    """Compute base conviction weight from 3 categorical sub-scores.

    Formula: weight = 0.40*na_score + 0.35*qs_score + 0.25*sa_score
    Output range: 0.10 (all very_weak) to 1.00 (all very_strong).
    """
    return round(
        _DIM_WEIGHTS["narrative_alignment"] * _LEVEL_SCORE[na]
        + _DIM_WEIGHTS["quant_support"] * _LEVEL_SCORE[qs]
        + _DIM_WEIGHTS["signal_agreement"] * _LEVEL_SCORE[sa],
        4,
    )


# Full 125-combination table generated programmatically (5^3).
CONVICTION_TABLE: dict[tuple[str, str, str], float] = {
    (na, qs, sa): _compute_table_weight(na, qs, sa)
    for na in VALID_SCORES
    for qs in VALID_SCORES
    for sa in VALID_SCORES
}

# First-run mapping: only narrative_alignment is available.
FIRST_RUN_TABLE: dict[str, float] = {
    "very_strong": 0.90,
    "strong":      0.70,
    "moderate":    0.45,
    "weak":        0.25,
    "very_weak":   0.10,
}


def apply_confidence_modifier(base_weight: float, confidence: float) -> float:
    """Scale a base conviction weight by a numeric confidence score.

    Args:
        base_weight: Base conviction weight from CONVICTION_TABLE or FIRST_RUN_TABLE.
        confidence: Numeric confidence in [0.0, 10.0]. 5.0 = neutral (no change).

    Returns:
        Adjusted weight. At confidence=0 → 70% of base. At confidence=10 → 130%.
    """
    confidence = max(0.0, min(10.0, confidence))  # clamp
    return base_weight * (0.7 + 0.06 * confidence)


def get_conviction_weight(conviction: dict, is_first_run: bool = False) -> float:
    """Map conviction sub-scores to a final numeric weight.

    Applies the categorical base weight first, then applies the confidence
    modifier for the relevant dimension(s).

    Args:
        conviction: Dict with keys:
            - narrative_alignment: very_strong|strong|moderate|weak|very_weak
            - narrative_confidence: float 0.0–10.0 (optional, defaults to 5.0)
            - quant_support: same levels, or 'n/a' on first run
            - quant_confidence: float 0.0–10.0 (optional)
            - signal_agreement: same levels, or 'n/a' on first run
            - signal_confidence: float 0.0–10.0 (optional)
        is_first_run: If True, use first-run mapping (narrative_alignment only).

    Returns:
        Final numeric conviction weight (positive float).

    Raises:
        ValueError: If required scores are invalid.
    """
    na = conviction.get("narrative_alignment", "")
    qs = conviction.get("quant_support", "")
    sa = conviction.get("signal_agreement", "")

    if is_first_run or (qs == "n/a" and sa == "n/a"):
        if na not in FIRST_RUN_TABLE:
            raise ValueError(
                f"Invalid narrative_alignment for first run: {na!r}. "
                f"Expected one of {list(FIRST_RUN_TABLE.keys())}"
            )
        base = FIRST_RUN_TABLE[na]
        confidence = float(conviction.get("narrative_confidence") or 5.0)
        return apply_confidence_modifier(base, confidence)

    key = (na, qs, sa)
    if key not in CONVICTION_TABLE:
        raise ValueError(
            f"Invalid conviction combination: {key}. "
            f"Each score must be one of {VALID_SCORES}"
        )
    base = CONVICTION_TABLE[key]

    # Blend confidence scores across all three dimensions proportionally.
    na_conf = float(conviction.get("narrative_confidence") or 5.0)
    qs_conf = float(conviction.get("quant_confidence") or 5.0)
    sa_conf = float(conviction.get("signal_confidence") or 5.0)
    blended_confidence = (
        _DIM_WEIGHTS["narrative_alignment"] * na_conf
        + _DIM_WEIGHTS["quant_support"] * qs_conf
        + _DIM_WEIGHTS["signal_agreement"] * sa_conf
    )
    return apply_confidence_modifier(base, blended_confidence)
