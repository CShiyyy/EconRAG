import json

import pytest

from pipeline.config import DEFAULT_CONSTRAINTS
from pipeline.db.helpers import (
    get_account,
    get_constraints,
    get_derived_weights,
    get_portfolio_value,
    get_previous_computed_target,
    get_watchlist,
    is_first_run,
)
from pipeline.db.init import initialize
from tests.conftest import MOCK_WATCHLIST


# ---------------------------------------------------------------------------
# Unit tests (mock scraper, no network)
# ---------------------------------------------------------------------------


def test_account_row_created(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    acct = get_account(db_conn)
    assert acct["cash_balance"] == 100_000.0
    assert acct["initial_cash"] == 100_000.0
    assert acct["universe"] == "djia30"


def test_default_constraints(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    constraints = get_constraints(db_conn)
    assert len(constraints) == 4
    for name, (default_val, _) in DEFAULT_CONSTRAINTS.items():
        assert constraints[name] == pytest.approx(default_val)


def test_constraint_overrides(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0, constraint_overrides={"cash_floor": 0.10})
    constraints = get_constraints(db_conn)
    assert constraints["cash_floor"] == pytest.approx(0.10)
    assert constraints["max_single_position"] == pytest.approx(0.15)  # unchanged


def test_watchlist_populated(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    watchlist = get_watchlist(db_conn)
    assert len(watchlist) == len(MOCK_WATCHLIST)
    tickers = {w["ticker"] for w in watchlist}
    expected_tickers = {t for t, _, _ in MOCK_WATCHLIST}
    assert tickers == expected_tickers


def test_canonical_entities_companies(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    rows = db_conn.execute(
        "SELECT * FROM canonical_entities WHERE entity_type = 'COMPANY'"
    ).fetchall()
    entity_ids = {row["canonical_id"] for row in rows}
    expected = {t for t, _, _ in MOCK_WATCHLIST}
    assert entity_ids == expected


def test_canonical_entities_sectors(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    rows = db_conn.execute(
        "SELECT * FROM canonical_entities WHERE entity_type = 'SECTOR'"
    ).fetchall()
    unique_sectors = {s for _, _, s in MOCK_WATCHLIST}
    assert len(rows) == len(unique_sectors)


def test_canonical_entity_aliases(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    row = db_conn.execute(
        "SELECT aliases FROM canonical_entities WHERE canonical_id = 'MSFT'"
    ).fetchone()
    aliases = json.loads(row["aliases"])
    assert "Microsoft Corporation" in aliases
    assert "Microsoft" in aliases  # suffix stripped


def test_is_first_run_true(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    assert is_first_run(db_conn) is True


def test_double_init_raises(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    with pytest.raises(RuntimeError, match="already initialized"):
        initialize(db_conn, "djia30", 100_000.0)


def test_invalid_universe_raises(db_conn):
    with pytest.raises(ValueError, match="Unknown universe"):
        initialize(db_conn, "invalid", 100_000.0)


def test_invalid_cash_raises(db_conn):
    with pytest.raises(ValueError, match="positive"):
        initialize(db_conn, "djia30", -100.0)


def test_zero_cash_raises(db_conn):
    with pytest.raises(ValueError, match="positive"):
        initialize(db_conn, "djia30", 0)


def test_invalid_constraint_override_key_raises(db_conn, mock_scraper):
    with pytest.raises(ValueError, match="Unknown constraint"):
        initialize(db_conn, "djia30", 100_000.0, constraint_overrides={"bogus": 0.5})


def test_get_account(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 50_000.0)
    acct = get_account(db_conn)
    assert acct["account_id"] == 1
    assert acct["cash_balance"] == 50_000.0
    assert "initialized_at" in acct


def test_get_constraints(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    c = get_constraints(db_conn)
    assert set(c.keys()) == set(DEFAULT_CONSTRAINTS.keys())


def test_get_watchlist(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    wl = get_watchlist(db_conn)
    assert all("ticker" in w and "company_name" in w and "sector" in w for w in wl)


def test_get_portfolio_value_cash_only(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    assert get_portfolio_value(db_conn) == pytest.approx(100_000.0)


def test_get_derived_weights_empty(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    assert get_derived_weights(db_conn) == {}


def test_get_previous_computed_target_none(db_conn, mock_scraper):
    initialize(db_conn, "djia30", 100_000.0)
    assert get_previous_computed_target(db_conn) is None


# ---------------------------------------------------------------------------
# Network tests (actual Wikipedia scrape)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_initialize_djia30(db_conn):
    initialize(db_conn, "djia30", 100_000.0)
    wl = get_watchlist(db_conn)
    assert 25 <= len(wl) <= 35


@pytest.mark.network
def test_initialize_nasdaq100(db_conn):
    initialize(db_conn, "nasdaq100", 100_000.0)
    wl = get_watchlist(db_conn)
    assert 95 <= len(wl) <= 110


@pytest.mark.network
def test_initialize_sp500(db_conn):
    initialize(db_conn, "sp500", 100_000.0)
    wl = get_watchlist(db_conn)
    assert 490 <= len(wl) <= 520
