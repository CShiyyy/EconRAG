import pytest

from pipeline.db.connection import get_connection
from pipeline.db.schema import create_tables


MOCK_WATCHLIST = [
    ("AAPL", "Apple Inc.", "Information Technology"),
    ("MSFT", "Microsoft Corporation", "Information Technology"),
    ("JPM", "JPMorgan Chase & Co.", "Financials"),
    ("JNJ", "Johnson & Johnson", "Health Care"),
    ("XOM", "Exxon Mobil Corporation", "Energy"),
    ("PG", "Procter & Gamble Co.", "Consumer Staples"),
    ("NVDA", "NVIDIA Corporation", "Information Technology"),
    ("UNH", "UnitedHealth Group Incorporated", "Health Care"),
]


@pytest.fixture
def db_conn(tmp_path):
    """File-based SQLite connection with all tables created."""
    conn = get_connection(tmp_path / "test.db")
    create_tables(conn)
    yield conn
    conn.close()


@pytest.fixture
def empty_conn(tmp_path):
    """File-based SQLite connection with NO tables created."""
    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def mock_scraper(monkeypatch):
    """Patch scrape_universe to return MOCK_WATCHLIST without network."""
    monkeypatch.setattr(
        "pipeline.db.init.scrape_universe",
        lambda universe: MOCK_WATCHLIST,
    )
