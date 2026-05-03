"""Tests for the score-based allocation pipeline in normalizer.py."""

import pytest
from pipeline.sizing.normalizer import (
    allocate_by_score,
    cap_sector_concentrations,
    floor_dust_positions,
    enforce_constraints,
)


# ── allocate_by_score ─────────────────────────────────────────────────────────

def test_worked_example():
    """Matches the canonical worked example from the design spec."""
    scores = {
        "A": 25, "B": 15, "C": 10, "D": 5,
        "E": 4,  "F": 4,  "G": 4,  "H": 2,
        "I": 2,  "J": 1,  "K": 0,  "L": -1,
    }
    result = allocate_by_score(scores, cash_floor=0.0, cap=0.15)

    expected = {
        "A": 0.15, "B": 0.15, "C": 0.15,
        "D": 0.125, "E": 0.10, "F": 0.10, "G": 0.10,
        "H": 0.05, "I": 0.05, "J": 0.025,
    }
    assert set(result.keys()) == set(expected.keys()), f"Unexpected tickers: {result.keys()}"
    for t, w in expected.items():
        assert abs(result[t] - w) < 1e-3, f"{t}: expected {w:.4f} got {result[t]:.4f}"


def test_all_non_positive_scores_returns_empty():
    scores = {"A": 0.0, "B": -1.0, "C": -0.5}
    result = allocate_by_score(scores, cash_floor=0.05, cap=0.15)
    assert result == {}


def test_single_positive_below_cap():
    """One ticker with a score, no cap hit — gets the full investable budget."""
    result = allocate_by_score({"A": 1.0}, cash_floor=0.05, cap=1.0)
    assert abs(result["A"] - 0.95) < 1e-9


def test_single_positive_above_cap():
    """One ticker capped — gets max_single, remainder stays in cash."""
    result = allocate_by_score({"A": 1.0}, cash_floor=0.0, cap=0.15)
    assert abs(result["A"] - 0.15) < 1e-9
    assert len(result) == 1


def test_two_equal_scores_no_cap():
    result = allocate_by_score({"A": 1.0, "B": 1.0}, cash_floor=0.0, cap=0.60)
    assert abs(result["A"] - 0.50) < 1e-9
    assert abs(result["B"] - 0.50) < 1e-9


def test_sum_equals_investable_when_uncapped():
    """When no cap is hit, weights must sum to exactly 1 - cash_floor."""
    scores = {"A": 3.0, "B": 2.0, "C": 1.0}
    result = allocate_by_score(scores, cash_floor=0.05, cap=0.50)
    assert abs(sum(result.values()) - 0.95) < 1e-9


def test_sum_less_than_investable_when_all_capped():
    """When breadth is too low to fill the budget, sum < 1 - cash_floor."""
    # 3 tickers × 0.15 cap = 0.45 max, cash_floor=0.05 → target=0.95, gap=0.50
    scores = {"A": 1.0, "B": 1.0, "C": 1.0}
    result = allocate_by_score(scores, cash_floor=0.05, cap=0.15)
    assert abs(sum(result.values()) - 0.45) < 1e-9
    for w in result.values():
        assert abs(w - 0.15) < 1e-9


def test_proportional_by_score():
    """Two uncapped tickers: weights should be proportional to scores."""
    scores = {"A": 3.0, "B": 1.0}
    result = allocate_by_score(scores, cash_floor=0.0, cap=0.90)
    assert abs(result["A"] / result["B"] - 3.0) < 1e-6


# ── cap_sector_concentrations ─────────────────────────────────────────────────

def test_sector_cap_redistributes_by_score():
    """Excess from over-cap sector flows to under-cap sector proportional to score."""
    weights = {"A": 0.20, "B": 0.20, "C": 0.05, "D": 0.05}
    sectors = {"A": "Tech", "B": "Tech", "C": "Finance", "D": "Finance"}
    scores  = {"A": 5.0,   "B": 5.0,   "C": 3.0,       "D": 1.0}
    result = cap_sector_concentrations(weights, sectors, max_sector=0.30, scores=scores)

    tech_total = result["A"] + result["B"]
    assert tech_total <= 0.30 + 1e-9, f"Tech sector still over cap: {tech_total:.4f}"

    # Finance tickers receive excess proportional to their scores (3:1)
    finance_excess_A = result["C"] - 0.05
    finance_excess_B = result["D"] - 0.05
    if finance_excess_A + finance_excess_B > 0:
        ratio = finance_excess_A / (finance_excess_A + finance_excess_B)
        assert abs(ratio - 0.75) < 1e-3, f"Score-proportional redistribution failed: {ratio:.4f}"


def test_sector_cap_no_breach_unchanged():
    weights = {"A": 0.10, "B": 0.10, "C": 0.10}
    sectors = {"A": "Tech", "B": "Tech", "C": "Finance"}
    result = cap_sector_concentrations(weights, sectors, max_sector=0.30)
    for t in weights:
        assert abs(result[t] - weights[t]) < 1e-9


# ── floor_dust_positions ──────────────────────────────────────────────────────

def test_dust_removed_and_redistributed_by_score():
    weights = {"A": 0.50, "B": 0.40, "C": 0.01}   # C is dust
    scores  = {"A": 2.0,  "B": 1.0,  "C": 0.5}
    result = floor_dust_positions(weights, min_size=0.02, scores=scores)

    assert "C" not in result or result.get("C", 0) == 0
    assert abs((result.get("A", 0) + result.get("B", 0)) - 0.91) < 1e-9
    # A should receive 2/3 of the freed weight, B should receive 1/3
    freed = 0.01
    assert abs(result["A"] - (0.50 + freed * 2/3)) < 1e-6
    assert abs(result["B"] - (0.40 + freed * 1/3)) < 1e-6


def test_dust_no_viable_survivors_returns_empty():
    """If all weights are dust, return empty (freed stays as cash)."""
    weights = {"A": 0.01, "B": 0.01}
    result = floor_dust_positions(weights, min_size=0.02)
    assert result == {}


def test_dust_no_dust_unchanged():
    weights = {"A": 0.50, "B": 0.45}
    result = floor_dust_positions(weights, min_size=0.02)
    assert result == weights


# ── enforce_constraints (integration) ────────────────────────────────────────

def test_enforce_constraints_worked_example():
    """enforce_constraints with the canonical worked example (cash_floor=0.05)."""
    scores = {
        "A": 25, "B": 15, "C": 10, "D": 5,
        "E": 4,  "F": 4,  "G": 4,  "H": 2,
        "I": 2,  "J": 1,  "K": 0,  "L": -1,
    }
    sectors = {t: "Tech" if t in ("A", "B", "C") else "Finance" for t in scores}
    constraints = {
        "max_single_position": 0.15,
        "max_sector_concentration": 0.50,
        "min_position_size": 0.02,
        "cash_floor": 0.05,
    }
    result = enforce_constraints(scores, sectors, constraints)

    for w in result.values():
        assert w <= 0.15 + 1e-9, f"Single-position cap violated: {w:.6f}"

    tech = sum(w for t, w in result.items() if sectors[t] == "Tech")
    assert tech <= 0.50 + 1e-9, f"Sector cap violated: tech={tech:.4f}"

    for w in result.values():
        assert w >= 0.02 - 1e-9, f"Dust floor violated: {w:.6f}"

    assert "K" not in result or result.get("K", 0) == 0
    assert "L" not in result or result.get("L", 0) == 0


def test_enforce_constraints_all_non_positive():
    scores = {"A": 0.0, "B": -1.0}
    sectors = {"A": "Tech", "B": "Tech"}
    constraints = {
        "max_single_position": 0.15,
        "max_sector_concentration": 0.50,
        "min_position_size": 0.02,
        "cash_floor": 0.05,
    }
    result = enforce_constraints(scores, sectors, constraints)
    assert result == {}


def test_enforce_constraints_single_cap_hard():
    """Single ticker with a huge score: capped at max_single, rest stays cash."""
    scores = {"A": 100.0}
    sectors = {"A": "Tech"}
    constraints = {
        "max_single_position": 0.15,
        "max_sector_concentration": 0.50,
        "min_position_size": 0.02,
        "cash_floor": 0.05,
    }
    result = enforce_constraints(scores, sectors, constraints)
    assert abs(result["A"] - 0.15) < 1e-9


def test_enforce_constraints_single_position_cap_never_exceeded():
    """Sector redistribution must not push a ticker above max_single_position."""
    scores = {"A": 10, "B": 10, "C": 9, "D": 1}
    sectors = {"A": "Tech", "B": "Tech", "C": "Finance", "D": "Finance"}
    constraints = {
        "max_single_position": 0.20,
        "max_sector_concentration": 0.35,
        "min_position_size": 0.02,
        "cash_floor": 0.05,
    }
    result = enforce_constraints(scores, sectors, constraints)
    for t, w in result.items():
        assert w <= 0.20 + 1e-9, f"{t} exceeds single-position cap: {w:.6f}"
