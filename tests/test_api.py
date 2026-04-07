"""Tests for the FastAPI backend (Phase 8)."""

import json
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.deps import RUN_STATUS, get_db
from backend.main import app
from pipeline.db.connection import get_connection
from pipeline.db.schema import create_tables

MOCK_WATCHLIST = [
    ("AAPL", "Apple Inc.", "Information Technology"),
    ("MSFT", "Microsoft Corporation", "Information Technology"),
    ("JPM", "JPMorgan Chase & Co.", "Financials"),
    ("JNJ", "Johnson & Johnson", "Health Care"),
    ("XOM", "Exxon Mobil Corporation", "Energy"),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with a fresh test database and mocked scraper."""
    db_path = tmp_path / "test.db"

    def override_get_db():
        conn = get_connection(db_path)
        try:
            yield conn
        finally:
            conn.close()

    monkeypatch.setattr(
        "pipeline.db.init.scrape_universe",
        lambda universe: MOCK_WATCHLIST,
    )
    # Patch the scraper at its source (used by watchlist refresh via local import)
    monkeypatch.setattr(
        "pipeline.scrapers.watchlist.scrape_universe",
        lambda universe: MOCK_WATCHLIST,
    )

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
    RUN_STATUS.clear()


@pytest.fixture
def initialized_client(client):
    """Client with an initialized system (djia30, $100k)."""
    resp = client.post(
        "/api/init",
        json={"universe": "djia30", "starting_cash": 100000.0},
    )
    assert resp.status_code == 200
    return client


# --- Init ---


class TestInit:
    def test_status_before_init(self, client):
        resp = client.get("/api/init/status")
        assert resp.status_code == 200
        assert resp.json()["initialized"] is False

    def test_init_success(self, client):
        resp = client.post(
            "/api/init",
            json={"universe": "djia30", "starting_cash": 100000.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cash_balance"] == 100000.0
        assert data["universe"] == "djia30"

    def test_status_after_init(self, initialized_client):
        resp = initialized_client.get("/api/init/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["initialized"] is True
        assert data["account"]["universe"] == "djia30"

    def test_init_duplicate_409(self, initialized_client):
        resp = initialized_client.post(
            "/api/init",
            json={"universe": "djia30", "starting_cash": 50000.0},
        )
        assert resp.status_code == 409

    def test_init_invalid_universe(self, client):
        resp = client.post(
            "/api/init",
            json={"universe": "invalid", "starting_cash": 100000.0},
        )
        assert resp.status_code == 422  # pydantic validation

    def test_init_negative_cash(self, client):
        resp = client.post(
            "/api/init",
            json={"universe": "djia30", "starting_cash": -100.0},
        )
        assert resp.status_code == 422

    def test_init_with_constraint_overrides(self, client):
        resp = client.post(
            "/api/init",
            json={
                "universe": "djia30",
                "starting_cash": 100000.0,
                "constraints": {"cash_floor": 0.10},
            },
        )
        assert resp.status_code == 200
        # Verify override applied
        constraints = client.get("/api/constraints").json()["constraints"]
        cf = next(c for c in constraints if c["constraint_name"] == "cash_floor")
        assert cf["value"] == 0.10


# --- Portfolio ---


class TestPortfolio:
    def test_portfolio_cash_only(self, initialized_client):
        resp = initialized_client.get("/api/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_value"] == 100000.0
        assert data["cash"] == 100000.0
        assert data["holdings"] == []
        assert data["weights"] == {}

    def test_portfolio_not_initialized(self, client):
        resp = client.get("/api/portfolio")
        assert resp.status_code == 404

    def test_snapshots_empty(self, initialized_client):
        resp = initialized_client.get("/api/portfolio/snapshots")
        assert resp.status_code == 200
        assert resp.json()["snapshots"] == []


# --- Watchlist ---


class TestWatchlist:
    def test_watchlist_populated(self, initialized_client):
        resp = initialized_client.get("/api/watchlist")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == len(MOCK_WATCHLIST)
        tickers = {i["ticker"] for i in items}
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_watchlist_not_initialized(self, client):
        resp = client.get("/api/watchlist")
        assert resp.status_code == 404

    def test_watchlist_refresh_no_new(self, initialized_client):
        resp = initialized_client.post("/api/watchlist/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["added"] == []
        assert data["total"] == len(MOCK_WATCHLIST)


# --- Constraints ---


class TestConstraints:
    def test_get_constraints(self, initialized_client):
        resp = initialized_client.get("/api/constraints")
        assert resp.status_code == 200
        constraints = resp.json()["constraints"]
        assert len(constraints) == 4
        names = {c["constraint_name"] for c in constraints}
        assert names == {
            "cash_floor",
            "max_single_position",
            "max_sector_concentration",
            "min_position_size",
        }

    def test_patch_constraints(self, initialized_client):
        resp = initialized_client.patch(
            "/api/constraints",
            json={"constraints": {"cash_floor": 0.10, "max_single_position": 0.20}},
        )
        assert resp.status_code == 200
        constraints = resp.json()["constraints"]
        cf = next(c for c in constraints if c["constraint_name"] == "cash_floor")
        msp = next(
            c for c in constraints if c["constraint_name"] == "max_single_position"
        )
        assert cf["value"] == 0.10
        assert msp["value"] == 0.20

    def test_patch_invalid_key(self, initialized_client):
        resp = initialized_client.patch(
            "/api/constraints",
            json={"constraints": {"nonexistent": 0.5}},
        )
        assert resp.status_code == 400


# --- Recommendations ---


class TestRecommendations:
    def test_empty_recommendations(self, initialized_client):
        resp = initialized_client.get("/api/recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["skip"] == 0
        assert data["limit"] == 20

    def test_recommendations_by_run_empty(self, initialized_client):
        resp = initialized_client.get("/api/recommendations/999")
        assert resp.status_code == 200
        assert resp.json()["items"] == []


# --- Runs ---


class TestRuns:
    def test_empty_runs(self, initialized_client):
        resp = initialized_client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_run_detail_not_found(self, initialized_client):
        resp = initialized_client.get("/api/runs/999")
        assert resp.status_code == 404

    def test_trigger_returns_id(self, initialized_client):
        with patch(
            "pipeline.orchestration.graph.run_pipeline",
            new_callable=AsyncMock,
            return_value={"run_id": 1},
        ):
            resp = initialized_client.post(
                "/api/runs/trigger",
                json={"run_type": "post_close"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "trigger_id" in data
            assert data["status"] == "running"

    def test_trigger_status_not_found(self, initialized_client):
        resp = initialized_client.get("/api/runs/trigger/9999/status")
        assert resp.status_code == 404

    def test_trigger_invalid_run_type(self, initialized_client):
        resp = initialized_client.post(
            "/api/runs/trigger",
            json={"run_type": "invalid"},
        )
        assert resp.status_code == 422


# --- Standing Events ---


class TestStandingEvents:
    def test_list_empty(self, initialized_client):
        resp = initialized_client.get("/api/standing-events")
        assert resp.status_code == 200
        assert resp.json()["items"] == []

    def test_create_and_list(self, initialized_client):
        create_resp = initialized_client.post(
            "/api/standing-events",
            json={
                "canonical_id": "event:tariff_war",
                "category": "trade_policy",
                "summary": "US-China tariff escalation",
                "affected_tickers": ["AAPL", "MSFT"],
            },
        )
        assert create_resp.status_code == 201
        event = create_resp.json()
        assert event["status"] == "active"
        assert event["promotion_source"] == "manual"
        assert event["affected_tickers"] == ["AAPL", "MSFT"]

        list_resp = initialized_client.get("/api/standing-events")
        assert len(list_resp.json()["items"]) == 1

    def test_patch_resolve(self, initialized_client):
        # Create
        create_resp = initialized_client.post(
            "/api/standing-events",
            json={
                "canonical_id": "event:fed_rate",
                "category": "monetary_policy",
                "summary": "Fed rate decision",
                "affected_tickers": ["JPM"],
            },
        )
        standing_id = create_resp.json()["standing_id"]

        # Resolve
        patch_resp = initialized_client.patch(
            f"/api/standing-events/{standing_id}",
            json={"status": "resolved"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["status"] == "resolved"
        assert patch_resp.json()["resolved_at"] is not None

    def test_patch_update_summary(self, initialized_client):
        create_resp = initialized_client.post(
            "/api/standing-events",
            json={
                "canonical_id": "event:oil_shock",
                "category": "geopolitical",
                "summary": "Original summary",
                "affected_tickers": ["XOM"],
            },
        )
        standing_id = create_resp.json()["standing_id"]

        patch_resp = initialized_client.patch(
            f"/api/standing-events/{standing_id}",
            json={"summary": "Updated summary", "affected_tickers": ["XOM", "JPM"]},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["summary"] == "Updated summary"
        assert patch_resp.json()["affected_tickers"] == ["XOM", "JPM"]

    def test_patch_not_found(self, initialized_client):
        resp = initialized_client.patch(
            "/api/standing-events/9999",
            json={"summary": "nope"},
        )
        assert resp.status_code == 404

    def test_filter_by_status(self, initialized_client):
        # Create two events
        initialized_client.post(
            "/api/standing-events",
            json={
                "canonical_id": "event:a",
                "category": "other",
                "summary": "A",
                "affected_tickers": [],
            },
        )
        create2 = initialized_client.post(
            "/api/standing-events",
            json={
                "canonical_id": "event:b",
                "category": "other",
                "summary": "B",
                "affected_tickers": [],
            },
        )
        # Resolve one
        initialized_client.patch(
            f"/api/standing-events/{create2.json()['standing_id']}",
            json={"status": "resolved"},
        )

        active = initialized_client.get("/api/standing-events?status=active")
        assert len(active.json()["items"]) == 1
        assert active.json()["items"][0]["summary"] == "A"

        resolved = initialized_client.get("/api/standing-events?status=resolved")
        assert len(resolved.json()["items"]) == 1
        assert resolved.json()["items"][0]["summary"] == "B"
