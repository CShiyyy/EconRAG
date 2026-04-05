"""Conviction weight mapping for the Position Sizing Engine.

Maps Agent C's categorical conviction sub-scores to numeric weights.
All 27 combinations of (narrative_alignment, quant_support, signal_agreement)
are defined, plus a first-run mapping when quant/signal data is unavailable.
"""

VALID_SCORES = ("strong", "moderate", "weak")

# All 27 combinations: (narrative_alignment, quant_support, signal_agreement) -> weight
CONVICTION_TABLE: dict[tuple[str, str, str], float] = {
    # strong narrative_alignment (9 combos)
    ("strong", "strong", "strong"): 1.00,
    ("strong", "strong", "moderate"): 0.80,
    ("strong", "strong", "weak"): 0.70,
    ("strong", "moderate", "strong"): 0.85,
    ("strong", "moderate", "moderate"): 0.60,
    ("strong", "moderate", "weak"): 0.45,
    ("strong", "weak", "strong"): 0.65,
    ("strong", "weak", "moderate"): 0.50,
    ("strong", "weak", "weak"): 0.35,
    # moderate narrative_alignment (9 combos)
    ("moderate", "strong", "strong"): 0.75,
    ("moderate", "strong", "moderate"): 0.55,
    ("moderate", "strong", "weak"): 0.45,
    ("moderate", "moderate", "strong"): 0.60,
    ("moderate", "moderate", "moderate"): 0.50,
    ("moderate", "moderate", "weak"): 0.35,
    ("moderate", "weak", "strong"): 0.45,
    ("moderate", "weak", "moderate"): 0.40,
    ("moderate", "weak", "weak"): 0.25,
    # weak narrative_alignment (9 combos)
    ("weak", "strong", "strong"): 0.50,
    ("weak", "strong", "moderate"): 0.40,
    ("weak", "strong", "weak"): 0.30,
    ("weak", "moderate", "strong"): 0.40,
    ("weak", "moderate", "moderate"): 0.30,
    ("weak", "moderate", "weak"): 0.20,
    ("weak", "weak", "strong"): 0.30,
    ("weak", "weak", "moderate"): 0.20,
    ("weak", "weak", "weak"): 0.15,
}

# First-run mapping: only narrative_alignment is available.
FIRST_RUN_TABLE: dict[str, float] = {
    "strong": 0.70,
    "moderate": 0.45,
    "weak": 0.20,
}


def get_conviction_weight(conviction: dict, is_first_run: bool = False) -> float:
    """Map conviction sub-scores to a numeric weight.

    Args:
        conviction: Dict with keys 'narrative_alignment', 'quant_support',
                    'signal_agreement'. Values are 'strong'/'moderate'/'weak'
                    (or 'n/a' for quant_support/signal_agreement on first run).
        is_first_run: If True, use first-run mapping (narrative_alignment only).

    Returns:
        Numeric conviction weight (0.0 to 1.0).

    Raises:
        ValueError: If scores are invalid or combination not found.
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
        return FIRST_RUN_TABLE[na]

    key = (na, qs, sa)
    if key not in CONVICTION_TABLE:
        raise ValueError(
            f"Invalid conviction combination: {key}. "
            f"Each score must be one of {VALID_SCORES}"
        )
    return CONVICTION_TABLE[key]
