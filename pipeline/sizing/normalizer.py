"""Normalization and constraint enforcement for target weights.

Primary allocation pipeline (score-based):
1. allocate_by_score   — proportional allocation with iterative single-position cap
2. cap_sector_concentrations — sector cap with score-based redistribution
3. floor_dust_positions — dust removal; residual to cash when no viable survivors

Legacy helpers (normalize_weights, cap_single_positions, renormalize) are
retained but not called by the primary enforce_constraints path.
"""

from __future__ import annotations

from collections import defaultdict


def allocate_by_score(
    scores: dict[str, float],
    cash_floor: float,
    cap: float,
) -> dict[str, float]:
    """Proportional allocation with iterative single-position cap redistribution.

    Tickers with non-positive scores receive zero weight. Each iteration clips
    over-cap tickers to exactly `cap`, removes them from the pool, and
    re-allocates the remaining budget proportionally among uncapped tickers.
    Budget undeployable after all tickers are capped stays in cash.

    Args:
        scores: {ticker: raw_score}. Non-positive scores are excluded.
        cash_floor: Fraction to keep in cash (e.g. 0.05).
        cap: Maximum weight per ticker (e.g. 0.15).

    Returns:
        {ticker: target_weight} for tickers with positive scores.
    """
    investable = 1.0 - cash_floor
    pool = {t: s for t, s in scores.items() if s > 0}
    if not pool:
        return {}

    # If raw scores already fit within investable budget and each is within cap,
    # use them directly — don't inflate small scores to fill unused capacity.
    _eps = 1e-9
    raw_total = sum(pool.values())
    if raw_total <= investable + _eps and all(s <= cap + _eps for s in pool.values()):
        return {t: min(s, cap) for t, s in pool.items()}

    result: dict[str, float] = {}
    remaining = investable

    while pool and remaining > 1e-9:
        total = sum(pool.values())
        tentative = {t: remaining * s / total for t, s in pool.items()}
        newly_capped = [t for t, w in tentative.items() if w > cap]

        if not newly_capped:
            result.update(tentative)
            break

        for t in newly_capped:
            result[t] = cap
            remaining -= cap
            del pool[t]

    return result


def cap_sector_concentrations(
    weights: dict[str, float],
    sectors: dict[str, str],
    max_sector: float,
    scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Cap sector concentrations; redistribute freed budget by score to other sectors.

    Args:
        weights: {ticker: current_weight}.
        sectors: {ticker: sector_name}.
        max_sector: Maximum fraction per sector (e.g. 0.25).
        scores: Optional {ticker: raw_score} for proportional redistribution.
            Falls back to weight-proportional when scores are absent or zero.

    Returns:
        {ticker: adjusted_weight} with all sectors ≤ max_sector.
    """
    if not weights:
        return {}

    result = dict(weights)
    for _ in range(10):
        sector_weights: dict[str, float] = defaultdict(float)
        for t, w in result.items():
            sector_weights[sectors[t]] += w

        breached = {s: sw for s, sw in sector_weights.items() if sw > max_sector}
        if not breached:
            break

        for sector, sector_total in breached.items():
            excess = sector_total - max_sector
            scale = max_sector / sector_total
            for t in result:
                if sectors[t] == sector:
                    result[t] *= scale

            other = {t: w for t, w in result.items() if sectors[t] != sector and w > 0}
            if not other:
                continue

            if scores:
                other_s = {t: scores[t] for t in other if scores.get(t, 0.0) > 0}
                denom = sum(other_s.values())
                if denom > 0:
                    for t, s in other_s.items():
                        result[t] += excess * s / denom
                    continue

            # Fallback: redistribute by current weight
            other_total = sum(other.values())
            if other_total > 0:
                for t in other:
                    result[t] += excess * other[t] / other_total

    return result


def floor_dust_positions(
    weights: dict[str, float],
    min_size: float,
    scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Drop positions below min_size and redistribute freed budget to survivors by score.

    If no survivor has a positive score (or no survivors exist), freed budget
    stays in cash rather than forcing an invalid redistribution.

    Args:
        weights: {ticker: current_weight}.
        min_size: Minimum viable position size (e.g. 0.02).
        scores: Optional {ticker: raw_score} for proportional redistribution.

    Returns:
        {ticker: weight} with all weights ≥ min_size.
    """
    if not weights:
        return {}

    result = dict(weights)
    dust = {t: w for t, w in result.items() if 0 < w < min_size}
    if not dust:
        return {t: w for t, w in result.items() if w > 0}

    freed = sum(dust.values())
    for t in dust:
        result[t] = 0.0

    survivors = {t: w for t, w in result.items() if w >= min_size}
    if not survivors:
        return {}

    if scores:
        s_scores = {t: scores[t] for t in survivors if scores.get(t, 0.0) > 0}
        denom = sum(s_scores.values())
        if denom > 0:
            for t, s in s_scores.items():
                result[t] += freed * s / denom
            return {t: w for t, w in result.items() if w > 0}

    # Fallback: redistribute by current weight
    total_w = sum(survivors.values())
    if total_w > 0:
        for t in survivors:
            result[t] += freed * survivors[t] / total_w

    return {t: w for t, w in result.items() if w > 0}


def enforce_constraints(
    scores: dict[str, float],
    sectors: dict[str, str],
    constraints: dict[str, float],
) -> dict[str, float]:
    """Apply all constraints using score-based iterative allocation.

    Pipeline:
    1. allocate_by_score  — proportional + single-position cap
    2. cap_sector_concentrations — sector cap with score redistribution
    3. Hard clip any ticker pushed over max_single by sector redistribution
    4. floor_dust_positions — dust removal; residual to cash

    Args:
        scores: {ticker: raw_score} from engine (Agent B × Agent A tilt).
        sectors: {ticker: sector_name}.
        constraints: {constraint_name: value} from the constraints table.

    Returns:
        {ticker: final_target_weight}.
    """
    max_single = constraints["max_single_position"]
    max_sector = constraints["max_sector_concentration"]
    min_size = constraints["min_position_size"]
    cash_floor = constraints["cash_floor"]

    result = allocate_by_score(scores, cash_floor, max_single)
    result = cap_sector_concentrations(result, sectors, max_sector, scores=scores)
    result = {t: min(w, max_single) for t, w in result.items()}
    result = floor_dust_positions(result, min_size, scores=scores)
    return result


# ── Legacy helpers (not called by primary path) ──────────────────────────────

def normalize_weights(
    ticker_weights: dict[str, float],
    cash_floor: float,
) -> dict[str, float]:
    """Normalize conviction weights to target portfolio weights (legacy)."""
    if not ticker_weights:
        return {}
    total = sum(ticker_weights.values())
    if total == 0:
        return {}
    investable = 1.0 - cash_floor
    return {ticker: (w / total) * investable for ticker, w in ticker_weights.items()}


def cap_single_positions(
    weights: dict[str, float],
    max_single: float,
) -> dict[str, float]:
    """Cap single positions with weight-proportional redistribution (legacy)."""
    if not weights:
        return {}
    result = dict(weights)
    frozen: set[str] = set()
    for _ in range(10):
        newly_capped = {t: w for t, w in result.items() if w > max_single and t not in frozen}
        if not newly_capped:
            break
        excess = sum(w - max_single for w in newly_capped.values())
        for t in newly_capped:
            result[t] = max_single
            frozen.add(t)
        free = {t: w for t, w in result.items() if t not in frozen and w > 0}
        free_total = sum(free.values())
        if free_total > 0:
            for t in free:
                result[t] += excess * (free[t] / free_total)
    return result


def renormalize(weights: dict[str, float], cash_floor: float) -> dict[str, float]:
    """Re-normalize weights to sum to exactly (1 - cash_floor) (legacy)."""
    if not weights:
        return {}
    current_total = sum(weights.values())
    if current_total == 0:
        return {}
    target = 1.0 - cash_floor
    scale = target / current_total
    return {t: w * scale for t, w in weights.items()}
