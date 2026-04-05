"""Normalization and constraint enforcement for target weights.

Implements the 4-step constraint enforcement pipeline:
1. Cap single positions
2. Cap sector concentrations
3. Floor dust positions
4. Re-normalize to (1 - cash_floor)
"""

from collections import defaultdict


def normalize_weights(
    ticker_weights: dict[str, float],
    cash_floor: float,
) -> dict[str, float]:
    """Normalize conviction weights to target portfolio weights.

    Raw weight per ticker = conviction_weight_i / sum(all conviction_weights).
    Then scale by (1.0 - cash_floor) to reserve cash.

    Args:
        ticker_weights: {ticker: conviction_weight} for Buy/Hold tickers only.
        cash_floor: Fraction of portfolio to keep in cash (e.g. 0.05).

    Returns:
        {ticker: target_weight} summing to (1 - cash_floor).
    """
    if not ticker_weights:
        return {}

    total = sum(ticker_weights.values())
    if total == 0:
        return {}

    investable = 1.0 - cash_floor
    return {
        ticker: (w / total) * investable
        for ticker, w in ticker_weights.items()
    }


def cap_single_positions(
    weights: dict[str, float],
    max_single: float,
) -> dict[str, float]:
    """Cap any single position exceeding max_single, redistribute excess proportionally."""
    if not weights:
        return {}

    result = dict(weights)
    frozen: set[str] = set()  # tickers permanently locked at max_single

    for _ in range(10):  # iterate until stable
        newly_capped = {t: w for t, w in result.items() if w > max_single and t not in frozen}
        if not newly_capped:
            break

        excess = sum(w - max_single for w in newly_capped.values())
        for t in newly_capped:
            result[t] = max_single
            frozen.add(t)

        # Redistribute to non-frozen tickers
        free = {t: w for t, w in result.items() if t not in frozen and w > 0}
        free_total = sum(free.values())
        if free_total > 0:
            for t in free:
                result[t] += excess * (free[t] / free_total)

    return result


def cap_sector_concentrations(
    weights: dict[str, float],
    sectors: dict[str, str],
    max_sector: float,
) -> dict[str, float]:
    """Cap sector concentrations, redistribute excess proportionally to other sectors."""
    if not weights:
        return {}

    result = dict(weights)
    for _ in range(10):  # iterate until stable
        # Sum weights per sector
        sector_weights: dict[str, float] = defaultdict(float)
        for t, w in result.items():
            sector_weights[sectors[t]] += w

        breached = {s: sw for s, sw in sector_weights.items() if sw > max_sector}
        if not breached:
            break

        for sector, sector_total in breached.items():
            excess = sector_total - max_sector
            # Scale down tickers in this sector proportionally
            sector_tickers = {t: w for t, w in result.items() if sectors[t] == sector}
            scale = max_sector / sector_total
            for t in sector_tickers:
                result[t] *= scale

            # Redistribute excess to tickers in other sectors
            other = {t: w for t, w in result.items() if sectors[t] != sector and w > 0}
            other_total = sum(other.values())
            if other_total > 0:
                for t in other:
                    result[t] += excess * (other[t] / other_total)

    return result


def floor_dust_positions(
    weights: dict[str, float],
    min_size: float,
) -> dict[str, float]:
    """Zero out positions below min_size, redistribute their weight proportionally."""
    if not weights:
        return {}

    result = dict(weights)
    dust = {t: w for t, w in result.items() if 0 < w < min_size}
    if not dust:
        return result

    freed = sum(dust.values())
    for t in dust:
        result[t] = 0.0

    remaining = {t: w for t, w in result.items() if w > 0}
    remaining_total = sum(remaining.values())
    if remaining_total > 0:
        for t in remaining:
            result[t] += freed * (remaining[t] / remaining_total)

    # Remove zeroed tickers
    return {t: w for t, w in result.items() if w > 0}


def renormalize(weights: dict[str, float], cash_floor: float) -> dict[str, float]:
    """Re-normalize weights to sum to exactly (1 - cash_floor)."""
    if not weights:
        return {}

    current_total = sum(weights.values())
    if current_total == 0:
        return {}

    target = 1.0 - cash_floor
    scale = target / current_total
    return {t: w * scale for t, w in weights.items()}


def enforce_constraints(
    weights: dict[str, float],
    sectors: dict[str, str],
    constraints: dict[str, float],
) -> dict[str, float]:
    """Apply all constraints in the specified order.

    Order matters:
    1. Single position cap
    2. Sector concentration cap
    3. Dust position floor
    4. Re-normalize

    Args:
        weights: {ticker: target_weight} from normalization step.
        sectors: {ticker: sector_name} for sector constraint checks.
        constraints: {constraint_name: value} from the constraints table.

    Returns:
        {ticker: final_target_weight} after all constraints applied.
    """
    max_single = constraints["max_single_position"]
    max_sector = constraints["max_sector_concentration"]
    min_size = constraints["min_position_size"]
    cash_floor = constraints["cash_floor"]

    result = cap_single_positions(weights, max_single)
    result = cap_sector_concentrations(result, sectors, max_sector)
    result = floor_dust_positions(result, min_size)
    result = renormalize(result, cash_floor)
    return result
