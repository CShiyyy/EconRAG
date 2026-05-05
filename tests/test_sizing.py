"""Comprehensive tests for the Position Sizing Engine (Phase 2)."""

import json
import math

import pytest

from pipeline.sizing.normalizer import (
    cap_sector_concentrations,
    cap_single_positions,
    enforce_constraints,
    floor_dust_positions,
    normalize_weights,
    renormalize,
)
from pipeline.sizing.trade_builder import (
    compute_trade_list,
    store_computed_targets,
    update_cost_basis,
)
from pipeline.sizing.engine import run_sizing_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_account(conn, cash=100_000.0, universe="djia30"):
    """Insert a minimal account row for testing."""
    from datetime import datetime, timezone

    conn.execute(
        "INSERT INTO account (account_id, cash_balance, initial_cash, universe, initialized_at) "
        "VALUES (1, ?, ?, ?, ?)",
        (cash, cash, universe, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _insert_holding(conn, ticker, shares, cost_basis, sector):
    conn.execute(
        "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) VALUES (?, ?, ?, ?)",
        (ticker, shares, cost_basis, sector),
    )
    conn.commit()


def _insert_watchlist(conn, entries):
    """entries: list of (ticker, name, sector)."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    for ticker, name, sector in entries:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker, company_name, sector, added_at) "
            "VALUES (?, ?, ?, ?)",
            (ticker, name, sector, now),
        )
    conn.commit()


def _insert_constraints(conn):
    """Insert default constraints."""
    from pipeline.config import DEFAULT_CONSTRAINTS

    for name, (value, desc) in DEFAULT_CONSTRAINTS.items():
        conn.execute(
            "INSERT OR IGNORE INTO constraints (constraint_name, value, description) VALUES (?, ?, ?)",
            (name, value, desc),
        )
    conn.commit()


def _insert_run_log(conn, run_type="pre_open"):
    """Insert a run_log row and return the run_id."""
    from datetime import datetime, timezone

    cursor = conn.execute(
        "INSERT INTO run_log (timestamp, run_type) VALUES (?, ?)",
        (datetime.now(timezone.utc).isoformat(), run_type),
    )
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Test: Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_five_tickers_sum_to_investable(self):
        """5 tickers with known weights should sum to (1 - cash_floor)."""
        weights = {"A": 1.0, "B": 0.85, "C": 0.60, "D": 0.50, "E": 0.30}
        cash_floor = 0.05
        result = normalize_weights(weights, cash_floor)

        assert len(result) == 5
        assert math.isclose(sum(result.values()), 1.0 - cash_floor, abs_tol=1e-9)

    def test_empty_weights(self):
        assert normalize_weights({}, 0.05) == {}

    def test_proportions_preserved(self):
        """Relative proportions should be maintained."""
        weights = {"A": 1.0, "B": 0.50}
        result = normalize_weights(weights, 0.0)
        ratio = result["A"] / result["B"]
        assert math.isclose(ratio, 2.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Test: Single Position Cap
# ---------------------------------------------------------------------------

class TestSinglePositionCap:
    def test_cap_exceeded_redistributed(self):
        """Ticker exceeding max_single_position is capped, excess redistributed."""
        # A exceeds cap; B-E are small enough to absorb excess without exceeding cap
        weights = {"A": 0.40, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.05}
        max_single = 0.15
        result = cap_single_positions(weights, max_single)

        assert result["A"] <= max_single + 1e-9
        # B-E should have absorbed A's excess and no one else hits cap
        for t in ("B", "C", "D", "E"):
            assert result[t] > weights[t]
        # Total is preserved
        assert math.isclose(sum(result.values()), sum(weights.values()), abs_tol=1e-6)

    def test_cap_all_exceed(self):
        """When all tickers exceed cap, all get capped (total may shrink — renormalize fixes this)."""
        weights = {"A": 0.50, "B": 0.25, "C": 0.20}
        max_single = 0.15
        result = cap_single_positions(weights, max_single)

        for w in result.values():
            assert w <= max_single + 1e-9

    def test_no_cap_needed(self):
        weights = {"A": 0.10, "B": 0.10, "C": 0.10}
        result = cap_single_positions(weights, 0.15)
        assert result == weights


# ---------------------------------------------------------------------------
# Test: Sector Concentration Cap
# ---------------------------------------------------------------------------

class TestSectorCap:
    def test_sector_cap_exceeded(self):
        """4 tickers in same sector exceeding max are capped, excess redistributed."""
        weights = {"A": 0.15, "B": 0.12, "C": 0.10, "D": 0.08, "E": 0.10}
        sectors = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Tech", "E": "Energy"}
        max_sector = 0.35

        result = cap_sector_concentrations(weights, sectors, max_sector)

        tech_total = sum(result[t] for t in ("A", "B", "C", "D"))
        assert tech_total <= max_sector + 1e-9
        assert math.isclose(sum(result.values()), sum(weights.values()), abs_tol=1e-9)

    def test_no_sector_breach(self):
        weights = {"A": 0.10, "B": 0.10, "C": 0.10}
        sectors = {"A": "Tech", "B": "Finance", "C": "Energy"}
        result = cap_sector_concentrations(weights, sectors, 0.35)
        for t in weights:
            assert math.isclose(result[t], weights[t], abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Test: Dust Position Floor
# ---------------------------------------------------------------------------

class TestDustFloor:
    def test_dust_zeroed_and_redistributed(self):
        """Ticker below min_position_size is removed, weight redistributed."""
        weights = {"A": 0.40, "B": 0.35, "C": 0.01}
        min_size = 0.02
        result = floor_dust_positions(weights, min_size)

        assert "C" not in result
        assert math.isclose(sum(result.values()), sum(weights.values()), abs_tol=1e-9)

    def test_no_dust(self):
        weights = {"A": 0.40, "B": 0.35}
        result = floor_dust_positions(weights, 0.02)
        assert result == weights


# ---------------------------------------------------------------------------
# Test: Full Constraint Enforcement
# ---------------------------------------------------------------------------

class TestEnforceConstraints:
    def test_full_pipeline_order(self):
        """Verify constraints are applied and result is valid."""
        weights = {"A": 0.30, "B": 0.25, "C": 0.20, "D": 0.15, "E": 0.05}
        sectors = {"A": "Tech", "B": "Tech", "C": "Finance", "D": "Energy", "E": "Energy"}
        constraints = {
            "max_single_position": 0.25,
            "max_sector_concentration": 0.40,
            "min_position_size": 0.02,
            "cash_floor": 0.05,
        }
        result = enforce_constraints(weights, sectors, constraints)

        # Dust removed — no positions below min_position_size
        for w in result.values():
            assert w >= constraints["min_position_size"] - 1e-9

        # Sum to investable fraction
        assert math.isclose(sum(result.values()), 0.95, abs_tol=1e-6)

        # At least some tickers remain
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Test: Trade List Computation
# ---------------------------------------------------------------------------

class TestTradeList:
    def test_correct_share_deltas(self):
        """Given current holdings and target weights, verify correct deltas."""
        target_weights = {"AAPL": 0.30, "MSFT": 0.20}
        current_holdings = {
            "AAPL": {"shares": 10, "cost_basis_per_share": 150.0, "sector": "Tech"},
        }
        total_value = 100_000.0
        fill_prices = {"AAPL": 150.0, "MSFT": 300.0}
        actions = {"AAPL": "Hold", "MSFT": "Buy"}

        trades = compute_trade_list(
            target_weights, current_holdings, total_value, fill_prices, actions,
            cash_floor=0.05, cash_balance=98_500.0,
        )

        # AAPL: target $30k, current $1500, delta = +$28500 = 190 shares
        aapl_trades = [t for t in trades if t["ticker"] == "AAPL"]
        assert len(aapl_trades) == 1
        assert aapl_trades[0]["action"] == "Buy"
        assert math.isclose(aapl_trades[0]["shares"], 190.0, abs_tol=0.01)

        # MSFT: target $20k, current $0, delta = +$20000 = ~66.67 shares
        msft_trades = [t for t in trades if t["ticker"] == "MSFT"]
        assert len(msft_trades) == 1
        assert msft_trades[0]["action"] == "Buy"
        assert math.isclose(msft_trades[0]["shares"], 66.667, abs_tol=0.01)

    def test_exit_liquidates(self):
        """Exit action sells all shares."""
        target_weights = {}
        current_holdings = {
            "AAPL": {"shares": 100, "cost_basis_per_share": 150.0, "sector": "Tech"},
        }
        actions = {"AAPL": "Exit"}
        fill_prices = {"AAPL": 160.0}

        trades = compute_trade_list(
            target_weights, current_holdings, 100_000.0, fill_prices, actions,
            cash_floor=0.05, cash_balance=84_000.0,
        )

        assert len(trades) == 1
        assert trades[0]["action"] == "Exit"
        assert trades[0]["shares"] == 100

    def test_cash_floor_violation_raises(self):
        """Engine refuses trades that would violate cash floor."""
        target_weights = {"AAPL": 0.95}
        current_holdings = {}
        fill_prices = {"AAPL": 150.0}
        actions = {"AAPL": "Buy"}

        with pytest.raises(ValueError, match="cash floor"):
            compute_trade_list(
                target_weights, current_holdings, 100_000.0, fill_prices, actions,
                cash_floor=0.05, cash_balance=100.0,  # not enough cash
            )


# ---------------------------------------------------------------------------
# Test: Cost Basis Accounting
# ---------------------------------------------------------------------------

class TestCostBasis:
    def test_buy_new_position(self, db_conn):
        """New buy sets cost basis to fill price."""
        _init_account(db_conn)
        result = update_cost_basis(db_conn, "AAPL", "Buy", 100, 150.0, "Tech")
        assert result is None

        row = db_conn.execute("SELECT * FROM holdings WHERE ticker = 'AAPL'").fetchone()
        assert row["shares"] == 100
        assert row["cost_basis_per_share"] == 150.0

    def test_buy_add_to_existing(self, db_conn):
        """Adding to existing position uses weighted average cost basis."""
        _init_account(db_conn)
        _insert_holding(db_conn, "AAPL", 100, 150.0, "Tech")

        update_cost_basis(db_conn, "AAPL", "Buy", 50, 180.0, "Tech")

        row = db_conn.execute("SELECT * FROM holdings WHERE ticker = 'AAPL'").fetchone()
        assert row["shares"] == 150
        expected_basis = (100 * 150.0 + 50 * 180.0) / 150
        assert math.isclose(row["cost_basis_per_share"], expected_basis, abs_tol=0.01)

    def test_trim_cost_basis_unchanged(self, db_conn):
        """Trim reduces shares but cost basis stays the same."""
        _init_account(db_conn)
        _insert_holding(db_conn, "AAPL", 100, 150.0, "Tech")

        realized = update_cost_basis(db_conn, "AAPL", "Trim", 30, 180.0, "Tech")

        row = db_conn.execute("SELECT * FROM holdings WHERE ticker = 'AAPL'").fetchone()
        assert row["shares"] == 70
        assert row["cost_basis_per_share"] == 150.0
        assert math.isclose(realized, (180.0 - 150.0) * 30, abs_tol=0.01)

    def test_exit_removes_position(self, db_conn):
        """Exit removes the holdings row and returns realized P&L."""
        _init_account(db_conn)
        _insert_holding(db_conn, "AAPL", 100, 150.0, "Tech")

        realized = update_cost_basis(db_conn, "AAPL", "Exit", 100, 180.0, "Tech")

        row = db_conn.execute("SELECT * FROM holdings WHERE ticker = 'AAPL'").fetchone()
        assert row is None
        assert math.isclose(realized, (180.0 - 150.0) * 100, abs_tol=0.01)

    def test_exit_no_position_raises(self, db_conn):
        _init_account(db_conn)
        with pytest.raises(ValueError, match="no existing position"):
            update_cost_basis(db_conn, "AAPL", "Exit", 100, 180.0, "Tech")


# ---------------------------------------------------------------------------
# Test: Computed Targets Storage
# ---------------------------------------------------------------------------

class TestComputedTargets:
    def test_store_and_retrieve(self, db_conn):
        _init_account(db_conn)
        run_id = _insert_run_log(db_conn)
        data = {"AAPL": {"target_weight": 0.30, "conviction_weight": 0.85, "action": "Buy"}}
        target_id = store_computed_targets(db_conn, run_id, data)

        row = db_conn.execute(
            "SELECT * FROM computed_targets WHERE target_id = ?", (target_id,)
        ).fetchone()
        assert row is not None
        assert json.loads(row["per_ticker_json"]) == data


# ---------------------------------------------------------------------------
# Test: Full Engine — weights sourced exclusively from Agent B
# ---------------------------------------------------------------------------

class TestEngineWeightSourcing:
    """Branch `rationale-model-agent-b`: weights are 100% deterministic from
    Agent B's optimizer. Agent C output cannot scale them."""

    def test_zero_agent_b_weight_stays_zero(self, db_conn):
        """A ticker with Agent B weight=0 gets weight=0, period — no LLM upside."""
        _init_account(db_conn, cash=100_000.0)
        _insert_constraints(db_conn)
        _insert_watchlist(db_conn, [
            ("AAPL", "Apple", "Information Technology"),
            ("MSFT", "Microsoft", "Information Technology"),
            ("JPM", "JPMorgan", "Financials"),
        ])
        run_id = _insert_run_log(db_conn)

        agent_b_output = {
            "per_ticker": {
                "AAPL": {"target_weight": 0.15, "volatility_30d": 0.25, "health_score": "normal", "flags": []},
                "MSFT": {"target_weight": 0.10, "volatility_30d": 0.22, "health_score": "normal", "flags": []},
                "JPM":  {"target_weight": 0.0,  "volatility_30d": 0.20, "health_score": "normal", "flags": []},
            }
        }
        agent_c_output = {
            "AAPL": {"action": "Buy"},
            "MSFT": {"action": "Buy"},
            "JPM":  {"action": "Buy"},  # bullish narrative cannot revive a zero B weight
        }
        fill_prices = {"AAPL": 150.0, "MSFT": 300.0, "JPM": 200.0}

        result = run_sizing_engine(
            db_conn, run_id, agent_c_output, fill_prices,
            agent_b_output=agent_b_output,
        )

        assert result["target_weights"].get("JPM", 0.0) == 0.0
        assert result["target_weights"].get("AAPL", 0.0) > 0.0
        assert result["target_weights"].get("MSFT", 0.0) > 0.0

    def test_agent_c_action_does_not_scale_weights(self, db_conn):
        """Two tickers with identical Agent B target_weights get identical
        normalized weights regardless of Agent C action labels — narrative
        no longer modulates."""
        _init_account(db_conn, cash=100_000.0)
        _insert_constraints(db_conn)
        _insert_watchlist(db_conn, [
            ("AAPL", "Apple", "Information Technology"),
            ("MSFT", "Microsoft", "Information Technology"),
        ])
        run_id = _insert_run_log(db_conn)

        agent_b_output = {
            "per_ticker": {
                "AAPL": {"target_weight": 0.10, "volatility_30d": 0.25, "health_score": "normal", "flags": []},
                "MSFT": {"target_weight": 0.10, "volatility_30d": 0.22, "health_score": "normal", "flags": []},
            }
        }
        agent_c_output = {
            "AAPL": {"action": "Buy"},   # different label
            "MSFT": {"action": "Hold"},  # different label
        }
        fill_prices = {"AAPL": 150.0, "MSFT": 300.0}

        result = run_sizing_engine(
            db_conn, run_id, agent_c_output, fill_prices,
            agent_b_output=agent_b_output,
        )

        # Equal raw scores (both = b_weight = 0.10) survive as equal targets.
        assert math.isclose(
            result["target_weights"]["AAPL"],
            result["target_weights"]["MSFT"],
            abs_tol=1e-9,
        )

    def test_raw_score_equals_agent_b_target_weight(self, db_conn):
        """Engine surfaces raw_score = agent_b.target_weight in per_ticker_data."""
        _init_account(db_conn, cash=100_000.0)
        _insert_constraints(db_conn)
        _insert_watchlist(db_conn, [
            ("AAPL", "Apple", "Information Technology"),
            ("MSFT", "Microsoft", "Information Technology"),
        ])
        run_id = _insert_run_log(db_conn)

        agent_b_output = {
            "per_ticker": {
                "AAPL": {"target_weight": 0.12, "volatility_30d": 0.25, "health_score": "normal", "flags": []},
                "MSFT": {"target_weight": 0.07, "volatility_30d": 0.22, "health_score": "normal", "flags": []},
            }
        }
        agent_c_output = {
            "AAPL": {"action": "Buy"},
            "MSFT": {"action": "Hold"},
        }
        fill_prices = {"AAPL": 150.0, "MSFT": 300.0}

        result = run_sizing_engine(
            db_conn, run_id, agent_c_output, fill_prices,
            agent_b_output=agent_b_output,
        )

        assert math.isclose(result["per_ticker_data"]["AAPL"]["raw_score"], 0.12, abs_tol=1e-9)
        assert math.isclose(result["per_ticker_data"]["MSFT"]["raw_score"], 0.07, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Test: Full Engine — all Exit
# ---------------------------------------------------------------------------

class TestEngineAllExit:
    def test_all_exit_returns_to_cash(self, db_conn):
        """All Exit actions liquidate positions and return to cash."""
        _init_account(db_conn, cash=10_000.0)
        _insert_constraints(db_conn)
        _insert_watchlist(db_conn, [
            ("AAPL", "Apple", "Information Technology"),
            ("MSFT", "Microsoft", "Information Technology"),
        ])
        _insert_holding(db_conn, "AAPL", 100, 140.0, "Information Technology")
        _insert_holding(db_conn, "MSFT", 50, 280.0, "Information Technology")
        run_id = _insert_run_log(db_conn)

        agent_c_output = {
            "AAPL": {"action": "Exit"},
            "MSFT": {"action": "Exit"},
        }
        fill_prices = {"AAPL": 150.0, "MSFT": 300.0}

        result = run_sizing_engine(db_conn, run_id, agent_c_output, fill_prices, mutate_portfolio=True)

        # All positions should be gone
        holdings = db_conn.execute("SELECT * FROM holdings").fetchall()
        assert len(holdings) == 0

        # Cash should be initial + proceeds
        account = db_conn.execute("SELECT cash_balance FROM account WHERE account_id = 1").fetchone()
        expected_cash = 10_000.0 + (100 * 150.0) + (50 * 300.0)
        assert math.isclose(account["cash_balance"], expected_cash, abs_tol=0.01)

        # All trades should be Exit
        for trade in result["trade_list"]:
            assert trade["action"] == "Exit"
