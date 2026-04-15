"""Tests for Agent C — Cloud LLM Portfolio Synthesizer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from pipeline.agents.agent_c import (
    _validate_output,
    _validate_per_ticker,
    _validate_standing_actions,
    _degraded_output,
    _sanitize_json_text,
    run_agent_c,
)
from pipeline.agents.cloud_client import CloudAuthError, CloudTimeoutError
from pipeline.db.helpers import get_previous_assessment, store_recommendations


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "NVDA"]


def _valid_assessment_output(tickers=None):
    tickers = tickers or TICKERS
    per_ticker = {}
    for t in tickers:
        per_ticker[t] = {
            "action": "assessment",
            "conviction": {
                "narrative_alignment": "strong",
                "quant_support": "moderate",
                "signal_agreement": "strong",
            },
            "rationale": f"End-of-day assessment for {t}.",
            "key_risk_factors": ["market volatility"],
            "notable_change": None,
        }
    return {"per_ticker": per_ticker}


def _valid_decision_output(tickers=None):
    tickers = tickers or TICKERS
    per_ticker = {}
    actions = ["Buy", "Hold"]
    for i, t in enumerate(tickers):
        per_ticker[t] = {
            "action": actions[i % len(actions)],
            "conviction": {
                "narrative_alignment": "strong",
                "quant_support": "moderate",
                "signal_agreement": "strong",
            },
            "rationale": f"Decision reasoning for {t}.",
            "key_risk_factors": ["sector risk"],
        }
    return {"per_ticker": per_ticker}


def _valid_first_run_output(tickers=None):
    tickers = tickers or TICKERS
    per_ticker = {}
    for t in tickers:
        per_ticker[t] = {
            "action": "Buy",
            "conviction": {
                "narrative_alignment": "strong",
                "quant_support": "n/a",
                "signal_agreement": "n/a",
            },
            "rationale": f"Compelling initial position for {t}.",
            "key_risk_factors": ["new position risk"],
        }
    return {"per_ticker": per_ticker}


def _valid_decision_with_standing_actions():
    output = _valid_decision_output()
    output["standing_event_actions"] = {
        "promote_to_standing": [
            {
                "canonical_id": "tariff_war_2026",
                "category": "trade_policy",
                "summary": "US-China tariff escalation impacting semiconductors",
                "affected_tickers": ["NVDA", "AAPL"],
            }
        ],
        "recommend_resolution": [
            {
                "standing_id": 1,
                "reason": "Fed pivot completed, rate hike cycle ended",
            }
        ],
    }
    return output


def _agent_a_output():
    return {
        "macro_overview": "Markets driven by AI and rate expectations.",
        "standing_context_assessment": {},
        "per_ticker": {
            "AAPL": {
                "sentiment": "bullish",
                "confidence": "strong",
                "key_catalysts": ["earnings beat"],
                "narrative": "Strong performance expected.",
            },
            "NVDA": {
                "sentiment": "bullish",
                "confidence": "moderate",
                "key_catalysts": ["AI demand"],
                "narrative": "AI tailwind continues.",
            },
        },
    }


def _agent_b_output():
    return {
        "portfolio_level": {
            "total_value": 100000.0,
            "cash_pct": 0.08,
            "portfolio_volatility_30d": 0.18,
            "max_drawdown_30d": -0.04,
            "overall_status": "normal",
        },
        "per_ticker": {
            "AAPL": {
                "current_weight": 0.12,
                "previous_target_weight": 0.10,
                "drift": 0.02,
                "volatility_30d": 0.25,
                "sector": "Information Technology",
                "health_score": "normal",
                "flags": [],
            },
            "NVDA": {
                "current_weight": 0.15,
                "previous_target_weight": 0.14,
                "drift": 0.01,
                "volatility_30d": 0.35,
                "sector": "Information Technology",
                "health_score": "normal",
                "flags": [],
            },
        },
        "sector_concentrations": {
            "Information Technology": {"weight": 0.27, "limit": 0.35, "status": "normal"},
        },
        "constraint_violations": [],
    }


def _source_health():
    return [
        {"source": "yfinance", "status": "success", "items_fetched": 30},
        {"source": "newsdata", "status": "success", "items_fetched": 15},
    ]


# ---------------------------------------------------------------------------
# DB seed helpers
# ---------------------------------------------------------------------------

def _seed_run_log(conn, run_type="pre_open") -> int:
    cur = conn.execute(
        "INSERT INTO run_log (timestamp, run_type) VALUES ('2026-04-06T08:30:00', ?)",
        (run_type,),
    )
    conn.commit()
    return cur.lastrowid


def _seed_account(conn, cash=100000.0):
    conn.execute(
        "INSERT INTO account (account_id, cash_balance, initial_cash, universe, initialized_at) "
        "VALUES (1, ?, ?, 'djia30', '2026-04-06T00:00:00')",
        (cash, cash),
    )
    conn.commit()


def _seed_constraints(conn):
    for name, val in [
        ("cash_floor", 0.05),
        ("max_single_position", 0.15),
        ("max_sector_concentration", 0.35),
        ("min_position_size", 0.02),
    ]:
        conn.execute(
            "INSERT INTO constraints (constraint_name, value, description) VALUES (?, ?, '')",
            (name, val),
        )
    conn.commit()


def _seed_watchlist(conn):
    for ticker, name, sector in [
        ("AAPL", "Apple Inc.", "Information Technology"),
        ("NVDA", "NVIDIA Corporation", "Information Technology"),
        ("JPM", "JPMorgan Chase", "Financials"),
    ]:
        conn.execute(
            "INSERT INTO watchlist (ticker, company_name, sector, added_at) VALUES (?, ?, ?, '2026-04-06')",
            (ticker, name, sector),
        )
    conn.commit()


def _seed_holdings(conn):
    for ticker, shares, cost, sector in [
        ("AAPL", 50, 150.0, "Information Technology"),
        ("NVDA", 30, 800.0, "Information Technology"),
    ]:
        conn.execute(
            "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) "
            "VALUES (?, ?, ?, ?)",
            (ticker, shares, cost, sector),
        )
    conn.commit()


def _seed_computed_target(conn, run_id):
    """Seed a computed_targets row so is_first_run() returns False."""
    conn.execute(
        "INSERT INTO computed_targets (run_id, timestamp, per_ticker_json) VALUES (?, '2026-04-06', ?)",
        (run_id, json.dumps({"AAPL": {"target_weight": 0.10}, "NVDA": {"target_weight": 0.14}})),
    )
    conn.commit()


def _seed_full_state(conn):
    """Seed a full DB state for non-first-run tests."""
    run_id = _seed_run_log(conn)
    _seed_account(conn)
    _seed_constraints(conn)
    _seed_watchlist(conn)
    _seed_holdings(conn)
    _seed_computed_target(conn, run_id)
    return run_id


def _make_mock_client(response_data=None, side_effect=None):
    """Create a mock CloudLLMClient."""
    client = AsyncMock()
    if side_effect is not None:
        client.generate = AsyncMock(side_effect=side_effect)
    elif response_data is not None:
        client.generate = AsyncMock(return_value=json.dumps(response_data))
    return client


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_assessment(self):
        assert _validate_output(_valid_assessment_output(), "assessment") is True

    def test_valid_decision(self):
        assert _validate_output(_valid_decision_output(), "decision") is True

    def test_valid_first_run(self):
        assert _validate_output(_valid_first_run_output(), "first_run") is True

    def test_valid_decision_with_standing_actions(self):
        assert _validate_output(_valid_decision_with_standing_actions(), "decision") is True

    def test_assessment_rejects_trade_actions(self):
        output = _valid_decision_output()  # has Buy/Hold actions
        assert _validate_output(output, "assessment") is False

    def test_decision_rejects_assessment_action(self):
        output = _valid_assessment_output()  # has assessment actions
        assert _validate_output(output, "decision") is False

    def test_first_run_rejects_non_na_quant(self):
        output = _valid_decision_output()  # has non-n/a quant_support
        assert _validate_output(output, "first_run") is False

    def test_rejects_invalid_action(self):
        output = _valid_decision_output()
        output["per_ticker"]["AAPL"]["action"] = "InvalidAction"
        assert _validate_output(output, "decision") is False

    def test_rejects_invalid_conviction(self):
        output = _valid_assessment_output()
        output["per_ticker"]["AAPL"]["conviction"]["narrative_alignment"] = "very_strong"
        assert _validate_output(output, "assessment") is False

    def test_rejects_missing_rationale(self):
        output = _valid_assessment_output()
        output["per_ticker"]["AAPL"]["rationale"] = ""
        assert _validate_output(output, "assessment") is False

    def test_rejects_empty_per_ticker(self):
        assert _validate_output({"per_ticker": {}}, "assessment") is False

    def test_rejects_non_dict(self):
        assert _validate_output("not a dict", "assessment") is False


class TestStandingActionValidation:
    def test_valid_promote(self):
        actions = {
            "promote_to_standing": [
                {
                    "canonical_id": "test_event",
                    "category": "geopolitical",
                    "summary": "Test event",
                    "affected_tickers": ["AAPL"],
                }
            ],
            "recommend_resolution": [],
        }
        assert _validate_standing_actions(actions) is True

    def test_invalid_category(self):
        actions = {
            "promote_to_standing": [
                {
                    "canonical_id": "test",
                    "category": "invalid_category",
                    "summary": "Test",
                    "affected_tickers": [],
                }
            ],
            "recommend_resolution": [],
        }
        assert _validate_standing_actions(actions) is False

    def test_invalid_resolution_id(self):
        actions = {
            "promote_to_standing": [],
            "recommend_resolution": [
                {"standing_id": "not_an_int", "reason": "test"}
            ],
        }
        assert _validate_standing_actions(actions) is False


class TestDegradedOutput:
    def test_assessment_degraded(self):
        result = _degraded_output(["AAPL"], "assessment")
        entry = result["per_ticker"]["AAPL"]
        assert entry["action"] == "assessment"
        assert entry["conviction"]["narrative_alignment"] == "weak"
        assert "failed" in entry["rationale"].lower()

    def test_decision_degraded(self):
        result = _degraded_output(["AAPL"], "decision")
        entry = result["per_ticker"]["AAPL"]
        assert entry["action"] == "Hold"
        assert entry["conviction"]["quant_support"] == "weak"

    def test_first_run_degraded(self):
        result = _degraded_output(["AAPL"], "first_run")
        entry = result["per_ticker"]["AAPL"]
        assert entry["action"] == "Hold"
        assert entry["conviction"]["quant_support"] == "n/a"
        assert entry["conviction"]["signal_agreement"] == "n/a"


# ---------------------------------------------------------------------------
# Integration tests (with mock cloud client)
# ---------------------------------------------------------------------------

class TestAssessmentMode:
    @pytest.mark.asyncio
    async def test_assessment_output_structure(self, db_conn):
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "post_close")
        output = _valid_assessment_output()
        client = _make_mock_client(response_data=output)

        result = await run_agent_c(
            db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
            run_id, "post_close", is_first_run=False, client=client,
        )

        assert "per_ticker" in result
        for entry in result["per_ticker"].values():
            assert entry["action"] == "assessment"
            assert entry["conviction"]["narrative_alignment"] in {"strong", "moderate", "weak"}

    @pytest.mark.asyncio
    async def test_assessment_stored_to_agent_outputs(self, db_conn):
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "post_close")
        client = _make_mock_client(response_data=_valid_assessment_output())

        await run_agent_c(
            db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
            run_id, "post_close", is_first_run=False, client=client,
        )

        row = db_conn.execute(
            "SELECT * FROM agent_outputs WHERE run_id = ? AND agent = 'C'", (run_id,)
        ).fetchone()
        assert row is not None
        assert json.loads(row["output_blob"])["per_ticker"] is not None

    @pytest.mark.asyncio
    async def test_assessment_stored_to_recommendations(self, db_conn):
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "post_close")
        client = _make_mock_client(response_data=_valid_assessment_output())

        await run_agent_c(
            db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
            run_id, "post_close", is_first_run=False, client=client,
        )

        rows = db_conn.execute(
            "SELECT * FROM recommendations WHERE run_id = ?", (run_id,)
        ).fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r["action"] == "assessment"


class TestDecisionMode:
    @pytest.mark.asyncio
    async def test_decision_output_structure(self, db_conn):
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "pre_open")
        client = _make_mock_client(response_data=_valid_decision_output())

        result = await run_agent_c(
            db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
            run_id, "pre_open", is_first_run=False, client=client,
        )

        assert "per_ticker" in result
        for entry in result["per_ticker"].values():
            assert entry["action"] in {"Buy", "Hold", "Trim", "Exit"}

    @pytest.mark.asyncio
    async def test_decision_with_standing_actions(self, db_conn):
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "pre_open")
        output = _valid_decision_with_standing_actions()
        client = _make_mock_client(response_data=output)

        result = await run_agent_c(
            db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
            run_id, "pre_open", is_first_run=False, client=client,
        )

        actions = result.get("standing_event_actions", {})
        assert len(actions.get("promote_to_standing", [])) == 1
        assert actions["promote_to_standing"][0]["canonical_id"] == "tariff_war_2026"
        assert len(actions.get("recommend_resolution", [])) == 1
        assert actions["recommend_resolution"][0]["standing_id"] == 1


class TestFirstRunMode:
    @pytest.mark.asyncio
    async def test_first_run_output(self, db_conn):
        _seed_account(db_conn)
        _seed_constraints(db_conn)
        _seed_watchlist(db_conn)
        run_id = _seed_run_log(db_conn)
        # No computed_targets → is_first_run=True
        output = _valid_first_run_output(["AAPL", "NVDA", "JPM"])
        client = _make_mock_client(response_data=output)

        result = await run_agent_c(
            db_conn, _agent_a_output(), None, _source_health(),
            run_id, "pre_open", is_first_run=True, client=client,
        )

        for entry in result["per_ticker"].values():
            assert entry["conviction"]["quant_support"] == "n/a"
            assert entry["conviction"]["signal_agreement"] == "n/a"

    @pytest.mark.asyncio
    async def test_first_run_recommends_from_watchlist(self, db_conn):
        _seed_account(db_conn)
        _seed_constraints(db_conn)
        _seed_watchlist(db_conn)
        run_id = _seed_run_log(db_conn)
        output = _valid_first_run_output(["AAPL", "NVDA"])
        client = _make_mock_client(response_data=output)

        result = await run_agent_c(
            db_conn, _agent_a_output(), None, _source_health(),
            run_id, "pre_open", is_first_run=True, client=client,
        )

        assert len(result["per_ticker"]) > 0
        for ticker in result["per_ticker"]:
            assert ticker in {"AAPL", "NVDA", "JPM"}


class TestRetryAndDegradedOutput:
    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self, db_conn):
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "post_close")

        valid_output = _valid_assessment_output()
        client = _make_mock_client(side_effect=[
            "not valid json {{{",  # first attempt fails
            json.dumps(valid_output),  # retry succeeds
        ])

        result = await run_agent_c(
            db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
            run_id, "post_close", is_first_run=False, client=client,
        )

        assert result["per_ticker"]["AAPL"]["action"] == "assessment"
        assert client.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_degraded_on_double_failure(self, db_conn):
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "post_close")

        client = _make_mock_client(side_effect=[
            "bad json",
            "still bad json",
        ])

        result = await run_agent_c(
            db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
            run_id, "post_close", is_first_run=False, client=client,
        )

        for entry in result["per_ticker"].values():
            assert entry["conviction"]["narrative_alignment"] == "weak"
            assert "failed" in entry["rationale"].lower()

    @pytest.mark.asyncio
    async def test_degraded_on_timeout(self, db_conn):
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "post_close")

        client = _make_mock_client(side_effect=[
            CloudTimeoutError("timeout"),
            CloudTimeoutError("timeout again"),
        ])

        result = await run_agent_c(
            db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
            run_id, "post_close", is_first_run=False, client=client,
        )

        for entry in result["per_ticker"].values():
            assert "failed" in entry["rationale"].lower()


class TestAPIErrors:
    @pytest.mark.asyncio
    async def test_auth_error_raises(self, db_conn):
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "post_close")

        client = _make_mock_client(side_effect=[
            CloudAuthError("401 Unauthorized"),
        ])

        with pytest.raises(CloudAuthError):
            await run_agent_c(
                db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
                run_id, "post_close", is_first_run=False, client=client,
            )


# ---------------------------------------------------------------------------
# DB helper tests
# ---------------------------------------------------------------------------

class TestStoreRecommendations:
    def test_stores_per_ticker_entries(self, db_conn):
        run_id = _seed_run_log(db_conn)
        per_ticker = _valid_assessment_output()["per_ticker"]
        ids = store_recommendations(db_conn, run_id, per_ticker)

        assert len(ids) == 2
        rows = db_conn.execute("SELECT * FROM recommendations WHERE run_id = ?", (run_id,)).fetchall()
        assert len(rows) == 2
        tickers = {r["ticker"] for r in rows}
        assert tickers == {"AAPL", "NVDA"}

    def test_stores_conviction_as_json(self, db_conn):
        run_id = _seed_run_log(db_conn)
        per_ticker = _valid_assessment_output()["per_ticker"]
        store_recommendations(db_conn, run_id, per_ticker)

        row = db_conn.execute(
            "SELECT conviction_scores FROM recommendations WHERE ticker = 'AAPL' AND run_id = ?",
            (run_id,),
        ).fetchone()
        scores = json.loads(row["conviction_scores"])
        assert scores["narrative_alignment"] == "strong"


class TestGetPreviousAssessment:
    def test_returns_none_when_no_assessments(self, db_conn):
        assert get_previous_assessment(db_conn) is None

    def test_returns_latest_assessment(self, db_conn):
        run_id = _seed_run_log(db_conn, "post_close")
        per_ticker = _valid_assessment_output()["per_ticker"]
        store_recommendations(db_conn, run_id, per_ticker)

        result = get_previous_assessment(db_conn)
        assert result is not None
        assert "AAPL" in result
        assert "conviction" in result["AAPL"]
        assert result["AAPL"]["conviction"]["narrative_alignment"] == "strong"


# ---------------------------------------------------------------------------
# Regression tests — JSON sanitizer and empty per_ticker retry
# ---------------------------------------------------------------------------

class TestSanitizeJsonText:
    def test_strips_markdown_fences(self):
        raw = "```json\n{\"a\": 1}\n```"
        result = json.loads(_sanitize_json_text(raw))
        assert result == {"a": 1}

    def test_strips_plain_fences(self):
        raw = "```\n{\"a\": 1}\n```"
        result = json.loads(_sanitize_json_text(raw))
        assert result == {"a": 1}

    def test_extracts_json_after_prose(self):
        raw = 'Here is the JSON:\n{"a": 1}'
        result = json.loads(_sanitize_json_text(raw))
        assert result == {"a": 1}

    def test_passthrough_clean_json(self):
        raw = '{"a": 1}'
        result = json.loads(_sanitize_json_text(raw))
        assert result == {"a": 1}

    def test_extracts_first_balanced_object_ignores_trailing_garbage(self):
        # Verifies string-aware brace counting: the "}" inside the string
        # value must not close the outer object prematurely.
        raw = '{"per_ticker": {"NVDA": {"nested": "}"}}} trailing garbage'
        result = json.loads(_sanitize_json_text(raw))
        assert result["per_ticker"]["NVDA"]["nested"] == "}"


class TestRetryOnEmptyPerTicker:
    @pytest.mark.asyncio
    async def test_retry_on_empty_per_ticker(self, db_conn):
        """When the LLM returns valid JSON with per_ticker:{}, the retry
        corrective prompt fires and the second attempt with real content succeeds."""
        _seed_full_state(db_conn)
        run_id = _seed_run_log(db_conn, "post_close")

        valid_output = _valid_assessment_output()
        client = _make_mock_client(side_effect=[
            '{"per_ticker": {}}',        # attempt 1: valid JSON but fails validation
            json.dumps(valid_output),    # attempt 2: succeeds
        ])

        result = await run_agent_c(
            db_conn, _agent_a_output(), _agent_b_output(), _source_health(),
            run_id, "post_close", is_first_run=False, client=client,
        )

        assert result["per_ticker"]["AAPL"]["action"] == "assessment"
        assert client.generate.call_count == 2
        # The corrective message appended on attempt 1 should reference the failure reason
        second_call_messages = client.generate.call_args_list[1][0][0]
        corrective = next(
            (m for m in second_call_messages if m.get("role") == "user"
             and "rejected" in m.get("content", "")),
            None,
        )
        assert corrective is not None, "Expected a corrective user message on retry"
        assert "per_ticker" in corrective["content"]
