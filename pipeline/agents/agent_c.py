"""Agent C — Cloud LLM Portfolio Synthesizer.

Operates in three modes:
  - Assessment (post-close): conviction scores only, no trade actions.
  - Decision (pre-open): trade actions + conviction scores.
  - First-run: initial position recommendations from cash.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from pipeline.agents.cloud_client import CloudAuthError, CloudLLMClient, CloudTimeoutError, create_cloud_client
from pipeline.db.helpers import (
    get_account,
    get_active_standing_events,
    get_constraints,
    get_derived_weights,
    get_holdings,
    get_previous_assessment,
    get_watchlist,
    store_agent_output,
    store_recommendations,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"Hold", "Buy", "Trim", "Exit", "assessment"}
VALID_CONVICTION = {"strong", "moderate", "weak"}
VALID_CONVICTION_NA = {"strong", "moderate", "weak", "n/a"}
VALID_CATEGORIES = {
    "geopolitical", "monetary_policy", "regulatory",
    "trade_policy", "sector_crisis", "other",
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_agent_a_context(agent_a_output: dict) -> str:
    """Format Agent A output into a readable context block."""
    parts = []
    if agent_a_output.get("macro_overview"):
        parts.append(f"MACRO OVERVIEW:\n{agent_a_output['macro_overview']}")

    standing = agent_a_output.get("standing_context_assessment", {})
    if standing:
        parts.append("STANDING EVENT ASSESSMENT:")
        for cid, info in standing.items():
            parts.append(
                f"  [{cid}] relevant={info.get('still_relevant')}, "
                f"impact={info.get('current_impact')}, "
                f"tickers={info.get('affected_tickers_update', [])}"
            )

    per_ticker = agent_a_output.get("per_ticker", {})
    if per_ticker:
        parts.append("PER-TICKER SENTIMENT:")
        for ticker, data in per_ticker.items():
            parts.append(
                f"  {ticker}: sentiment={data.get('sentiment')}, "
                f"confidence={data.get('confidence')}, "
                f"catalysts={data.get('key_catalysts', [])}"
            )
            if data.get("narrative"):
                parts.append(f"    narrative: {data['narrative']}")

    return "\n".join(parts)


def _format_agent_b_context(agent_b_output: dict) -> str:
    """Format Agent B output into a readable context block."""
    parts = []
    pl = agent_b_output.get("portfolio_level", {})
    parts.append(
        f"PORTFOLIO HEALTH:\n"
        f"  total_value={pl.get('total_value')}, cash_pct={pl.get('cash_pct')}, "
        f"volatility_30d={pl.get('portfolio_volatility_30d')}, "
        f"max_drawdown_30d={pl.get('max_drawdown_30d')}, "
        f"status={pl.get('overall_status')}"
    )

    per_ticker = agent_b_output.get("per_ticker", {})
    if per_ticker:
        parts.append("PER-TICKER QUANT METRICS:")
        for ticker, data in per_ticker.items():
            parts.append(
                f"  {ticker}: weight={data.get('current_weight')}, "
                f"drift={data.get('drift')}, vol={data.get('volatility_30d')}, "
                f"health={data.get('health_score')}, flags={data.get('flags', [])}"
            )

    violations = agent_b_output.get("constraint_violations", [])
    if violations:
        parts.append("CONSTRAINT VIOLATIONS:")
        for v in violations:
            parts.append(f"  {v.get('detail', str(v))}")

    return "\n".join(parts)


def _format_holdings_context(holdings: list[dict], weights: dict[str, float]) -> str:
    """Format current holdings into context."""
    if not holdings:
        return "CURRENT HOLDINGS: None (100% cash)"
    lines = ["CURRENT HOLDINGS:"]
    for h in holdings:
        ticker = h["ticker"]
        w = weights.get(ticker, 0.0)
        lines.append(
            f"  {ticker}: shares={h['shares']}, "
            f"cost_basis={h['cost_basis_per_share']}, "
            f"sector={h['sector']}, weight={w:.4f}"
        )
    return "\n".join(lines)


def _format_standing_context(events: list[dict]) -> str:
    """Format active standing events into context."""
    if not events:
        return "ACTIVE STANDING EVENTS: None"
    lines = ["ACTIVE STANDING EVENTS:"]
    for ev in events:
        lines.append(
            f"  [{ev.get('canonical_id')}] category={ev.get('category')}, "
            f"summary={ev.get('summary')}, "
            f"tickers={ev.get('affected_tickers', [])}"
        )
    return "\n".join(lines)


def _format_source_health(source_health: list[dict]) -> str:
    """Format source health status."""
    if not source_health:
        return "SOURCE HEALTH: No data"
    lines = ["SOURCE HEALTH:"]
    for s in source_health:
        lines.append(
            f"  {s.get('source')}: status={s.get('status')}, "
            f"items={s.get('items_fetched', 0)}"
        )
    return "\n".join(lines)


_ASSESSMENT_SYSTEM = """\
You are the lead portfolio manager performing an end-of-day assessment. \
Evaluate the portfolio state against the full day's data.

Produce ONLY valid JSON matching this exact schema:
{
  "per_ticker": {
    "TICKER": {
      "action": "assessment",
      "conviction": {
        "narrative_alignment": "strong | moderate | weak",
        "quant_support": "strong | moderate | weak",
        "signal_agreement": "strong | moderate | weak"
      },
      "rationale": "End-of-day assessment reasoning",
      "key_risk_factors": ["risk1", "risk2"],
      "notable_change": "Description of significant changes or null"
    }
  },
  "standing_event_actions": {
    "promote_to_standing": [],
    "recommend_resolution": []
  }
}

Rules:
- Every held ticker MUST have an entry with action "assessment".
- conviction sub-scores must be one of: "strong", "moderate", "weak".
- standing_event_actions is optional — omit or leave arrays empty if no actions warranted.
- Respond with ONLY valid JSON, no additional text."""

_DECISION_SYSTEM = """\
You are the lead portfolio manager making pre-open trading decisions. \
Cross-reference the sentiment analysis against quantitative metrics and the \
overnight assessment to produce concrete trade recommendations.

Produce ONLY valid JSON matching this exact schema:
{
  "per_ticker": {
    "TICKER": {
      "action": "Buy | Hold | Trim | Exit",
      "conviction": {
        "narrative_alignment": "strong | moderate | weak",
        "quant_support": "strong | moderate | weak",
        "signal_agreement": "strong | moderate | weak"
      },
      "rationale": "Narrative and quant reasoning for this decision",
      "key_risk_factors": ["risk1", "risk2"]
    }
  },
  "standing_event_actions": {
    "promote_to_standing": [
      {
        "canonical_id": "event_name",
        "category": "geopolitical | monetary_policy | regulatory | trade_policy | sector_crisis | other",
        "summary": "Description of the condition",
        "affected_tickers": ["TICKER1"]
      }
    ],
    "recommend_resolution": [
      {
        "standing_id": 123,
        "reason": "Why this standing event should be resolved"
      }
    ]
  }
}

Rules:
- Every held ticker and any new Buy recommendations MUST have an entry.
- action must be one of: "Buy", "Hold", "Trim", "Exit".
- conviction sub-scores must be one of: "strong", "moderate", "weak".
- standing_event_actions is optional — omit or leave arrays empty if no actions warranted.
- Respond with ONLY valid JSON, no additional text."""

_FIRST_RUN_SYSTEM = """\
You are the lead portfolio manager selecting initial positions for a new portfolio. \
The portfolio is currently 100% cash. Evaluate the watchlist tickers based on \
narrative analysis and recommend initial positions.

Produce ONLY valid JSON matching this exact schema:
{
  "per_ticker": {
    "TICKER": {
      "action": "Buy | Hold",
      "conviction": {
        "narrative_alignment": "strong | moderate | weak",
        "quant_support": "n/a",
        "signal_agreement": "n/a"
      },
      "rationale": "Why this ticker is compelling for the initial portfolio",
      "key_risk_factors": ["risk1", "risk2"]
    }
  }
}

Rules:
- Select tickers from the provided watchlist that you find compelling.
- action must be "Buy" for recommended positions, "Hold" is acceptable for borderline picks.
- quant_support and signal_agreement MUST be "n/a" (no quantitative data available).
- narrative_alignment must be one of: "strong", "moderate", "weak".
- No standing_event_actions on first run.
- Respond with ONLY valid JSON, no additional text."""


def _build_assessment_prompt(
    agent_a_output: dict,
    agent_b_output: dict,
    source_health: list[dict],
    holdings: list[dict],
    weights: dict[str, float],
    constraints: dict[str, float],
    standing_events: list[dict],
) -> list[dict]:
    """Build messages for assessment mode (post-close)."""
    user_parts = [
        _format_agent_a_context(agent_a_output),
        _format_agent_b_context(agent_b_output),
        _format_source_health(source_health),
        _format_holdings_context(holdings, weights),
        _format_standing_context(standing_events),
        f"CONSTRAINTS: {json.dumps(constraints)}",
    ]
    return [
        {"role": "system", "content": _ASSESSMENT_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _build_decision_prompt(
    agent_a_output: dict,
    agent_b_output: dict,
    source_health: list[dict],
    holdings: list[dict],
    weights: dict[str, float],
    constraints: dict[str, float],
    standing_events: list[dict],
    previous_assessment: dict | None,
) -> list[dict]:
    """Build messages for decision mode (pre-open)."""
    user_parts = [
        _format_agent_a_context(agent_a_output),
        _format_agent_b_context(agent_b_output),
        _format_source_health(source_health),
        _format_holdings_context(holdings, weights),
        _format_standing_context(standing_events),
        f"CONSTRAINTS: {json.dumps(constraints)}",
    ]
    if previous_assessment:
        user_parts.append(
            f"PREVIOUS POST-CLOSE ASSESSMENT:\n{json.dumps(previous_assessment, indent=2)}"
        )
    return [
        {"role": "system", "content": _DECISION_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _build_first_run_prompt(
    agent_a_output: dict,
    source_health: list[dict],
    constraints: dict[str, float],
    watchlist: list[dict],
) -> list[dict]:
    """Build messages for first-run mode."""
    tickers_info = "\n".join(
        f"  {w['ticker']}: {w['company_name']} ({w['sector']})"
        for w in watchlist
    )
    user_parts = [
        _format_agent_a_context(agent_a_output),
        _format_source_health(source_health),
        f"CONSTRAINTS: {json.dumps(constraints)}",
        f"WATCHLIST (available tickers):\n{tickers_info}",
    ]
    return [
        {"role": "system", "content": _FIRST_RUN_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_per_ticker(
    per_ticker: dict,
    mode: str,
) -> bool:
    """Validate the per_ticker section of Agent C output.

    Args:
        per_ticker: The per_ticker dict from the parsed output.
        mode: One of "assessment", "decision", "first_run".
    """
    if not isinstance(per_ticker, dict) or not per_ticker:
        return False

    for ticker, entry in per_ticker.items():
        if not isinstance(entry, dict):
            return False

        action = entry.get("action")
        if action not in VALID_ACTIONS:
            return False

        # Mode-specific action checks
        if mode == "assessment" and action != "assessment":
            return False
        if mode in ("decision", "first_run") and action == "assessment":
            return False

        conviction = entry.get("conviction")
        if not isinstance(conviction, dict):
            return False

        na = conviction.get("narrative_alignment")
        if na not in VALID_CONVICTION:
            return False

        qs = conviction.get("quant_support")
        sa = conviction.get("signal_agreement")

        if mode == "first_run":
            if qs != "n/a" or sa != "n/a":
                return False
        else:
            if qs not in VALID_CONVICTION_NA:
                return False
            if sa not in VALID_CONVICTION_NA:
                return False

        if not isinstance(entry.get("rationale"), str) or not entry["rationale"]:
            return False

        if not isinstance(entry.get("key_risk_factors"), list):
            return False

    return True


def _validate_standing_actions(standing_actions: dict) -> bool:
    """Validate standing_event_actions if present."""
    if not isinstance(standing_actions, dict):
        return False

    promotions = standing_actions.get("promote_to_standing", [])
    if not isinstance(promotions, list):
        return False
    for p in promotions:
        if not isinstance(p, dict):
            return False
        if not isinstance(p.get("canonical_id"), str):
            return False
        if p.get("category") not in VALID_CATEGORIES:
            return False
        if not isinstance(p.get("summary"), str) or not p["summary"]:
            return False
        if not isinstance(p.get("affected_tickers"), list):
            return False

    resolutions = standing_actions.get("recommend_resolution", [])
    if not isinstance(resolutions, list):
        return False
    for r in resolutions:
        if not isinstance(r, dict):
            return False
        if not isinstance(r.get("standing_id"), int):
            return False
        if not isinstance(r.get("reason"), str):
            return False

    return True


def _validate_output(data: dict, mode: str) -> bool:
    """Full output validation."""
    if not isinstance(data, dict):
        return False

    per_ticker = data.get("per_ticker")
    if not _validate_per_ticker(per_ticker, mode):
        return False

    standing_actions = data.get("standing_event_actions")
    if standing_actions is not None:
        if not _validate_standing_actions(standing_actions):
            return False

    return True


# ---------------------------------------------------------------------------
# Degraded output
# ---------------------------------------------------------------------------

def _degraded_output(tickers: list[str], mode: str) -> dict:
    """Produce a safe fallback output when cloud LLM fails."""
    per_ticker = {}
    for ticker in tickers:
        if mode == "first_run":
            per_ticker[ticker] = {
                "action": "Hold",
                "conviction": {
                    "narrative_alignment": "weak",
                    "quant_support": "n/a",
                    "signal_agreement": "n/a",
                },
                "rationale": "Cloud LLM extraction failed — using degraded output.",
                "key_risk_factors": [],
            }
        elif mode == "assessment":
            per_ticker[ticker] = {
                "action": "assessment",
                "conviction": {
                    "narrative_alignment": "weak",
                    "quant_support": "weak",
                    "signal_agreement": "weak",
                },
                "rationale": "Cloud LLM extraction failed — using degraded output.",
                "key_risk_factors": [],
                "notable_change": None,
            }
        else:  # decision
            per_ticker[ticker] = {
                "action": "Hold",
                "conviction": {
                    "narrative_alignment": "weak",
                    "quant_support": "weak",
                    "signal_agreement": "weak",
                },
                "rationale": "Cloud LLM extraction failed — using degraded output.",
                "key_risk_factors": [],
            }
    return {"per_ticker": per_ticker}


# ---------------------------------------------------------------------------
# Cloud call with retry
# ---------------------------------------------------------------------------

async def _call_cloud(
    client: CloudLLMClient,
    messages: list[dict],
    mode: str,
) -> dict | None:
    """Call cloud LLM, parse JSON, validate. Retry once on failure.

    Returns validated dict or None if both attempts fail.
    """
    for attempt in range(2):
        try:
            raw = await client.generate(messages, json_mode=True)
            data = json.loads(raw)
            if _validate_output(data, mode):
                return data
            logger.warning("Cloud response failed validation (attempt %d)", attempt + 1)
        except CloudAuthError:
            raise
        except (json.JSONDecodeError, CloudTimeoutError):
            logger.warning("Cloud call failed (attempt %d)", attempt + 1)
        except Exception:
            logger.exception("Unexpected error calling cloud LLM (attempt %d)", attempt + 1)

        if attempt == 0:
            # Add corrective prompt for retry
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON or did not match "
                        "the requested schema. Please respond with ONLY valid JSON "
                        "matching the schema exactly."
                    ),
                },
            ]

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_agent_c(
    conn: sqlite3.Connection,
    agent_a_output: dict,
    agent_b_output: dict | None,
    source_health: list[dict],
    run_id: int,
    run_type: str,
    is_first_run: bool,
    client: CloudLLMClient | None = None,
) -> dict:
    """Run Agent C: cloud LLM synthesis of agent outputs into recommendations.

    Args:
        conn: SQLite connection.
        agent_a_output: Agent A's structured JSON output.
        agent_b_output: Agent B's quant JSON, or None on first run.
        source_health: Per-source ingestion status list.
        run_id: Current pipeline run ID.
        run_type: "pre_open" or "post_close".
        is_first_run: True when computed_targets table is empty.
        client: Optional injectable CloudLLMClient (for testing).

    Returns:
        Parsed output dict with per_ticker and optional standing_event_actions.
    """
    if client is None:
        client = create_cloud_client()

    # Gather portfolio state
    constraints = get_constraints(conn)
    standing_events = get_active_standing_events(conn)

    # Determine mode and build prompt
    if is_first_run:
        mode = "first_run"
        watchlist = get_watchlist(conn)
        tickers = [w["ticker"] for w in watchlist]
        messages = _build_first_run_prompt(
            agent_a_output, source_health, constraints, watchlist,
        )
    elif run_type == "post_close":
        mode = "assessment"

        def _price_stub(ticker: str) -> float:
            # During assessment, we need weights but prices come from market data.
            # Use a simple query for the latest snapshot prices.
            return 1.0  # Weights are informational; exact prices not critical here

        holdings = get_holdings(conn)
        weights = get_derived_weights(conn) if not holdings else {}
        # For assessment, we need holdings but weights are best-effort
        tickers = [h["ticker"] for h in holdings] if holdings else []
        messages = _build_assessment_prompt(
            agent_a_output, agent_b_output or {}, source_health,
            holdings, weights, constraints, standing_events,
        )
    else:  # pre_open decision
        mode = "decision"
        holdings = get_holdings(conn)
        weights = {}  # Weights need price_fn; will be approximate
        tickers = [h["ticker"] for h in holdings] if holdings else []
        previous_assessment = get_previous_assessment(conn)
        messages = _build_decision_prompt(
            agent_a_output, agent_b_output or {}, source_health,
            holdings, weights, constraints, standing_events,
            previous_assessment,
        )

    # Call cloud LLM
    result = await _call_cloud(client, messages, mode)

    if result is None:
        logger.error("Cloud LLM failed after retries — producing degraded output")
        result = _degraded_output(tickers, mode)

    # Store raw output
    store_agent_output(conn, run_id, "C", result)

    # Store per-ticker recommendations
    store_recommendations(conn, run_id, result["per_ticker"])

    return result
