"""Agent C — Cloud LLM Portfolio Synthesizer.

Operates in three modes:
  - Assessment (post-close): conviction scores only, no trade actions.
  - Decision (pre-open): trade actions + conviction scores.
  - First-run: initial position recommendations from cash.
"""

from __future__ import annotations

import json
import logging
import re
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
- You MUST produce a per_ticker entry for Every ticker held in CURRENT HOLDINGS. Do not omit any held ticker.
- Do NOT include tickers not listed in CURRENT HOLDINGS. Use ONLY the exact ticker symbols provided.
- conviction sub-scores must be one of: "strong", "moderate", "weak".
- Even when sentiment data is thin or degraded, produce your best assessment — never return an empty per_ticker.
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
- You MUST produce a per_ticker entry for Every ticker held in CURRENT HOLDINGS, plus any new tickers recommended for purchase. Do not omit any held ticker or newly recommended buy.
- Do NOT include tickers not listed in CURRENT HOLDINGS. Use ONLY the exact ticker symbols provided.
- Use action="Buy" for tickers with positive evidence, action="Hold" for tickers with insufficient or degraded evidence, action="Trim" for tickers with negative evidence and action="Exit" for tickers with highly negative evidence.
- Even when sentiment data is thin or degraded, produce your best assessment — never return an empty per_ticker.
- narrative_alignment must be one of: "strong", "moderate", "weak".
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
- You MUST produce a per_ticker entry for EVERY ticker listed in the WATCHLIST section. Do not omit any ticker.
- Do NOT include any tickers that are not in the WATCHLIST. Use ONLY the exact ticker symbols provided.
- Use action="Buy" for tickers with positive evidence, action="Hold" for tickers with insufficient or degraded evidence.
- Even when sentiment data is thin or degraded, produce your best assessment — never return an empty per_ticker.
- narrative_alignment must be one of: "strong", "moderate", "weak".
- No standing_event_actions on first run.
- Respond with ONLY valid JSON, no additional text."""

# quant_support and signal_agreement MUST be "n/a" (no quantitative data available).

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
    ticker_list = ", ".join(w["ticker"] for w in watchlist)
    user_parts = [
        _format_agent_a_context(agent_a_output),
        _format_source_health(source_health),
        f"CONSTRAINTS: {json.dumps(constraints)}",
        f"WATCHLIST (available tickers):\n{tickers_info}",
        f"REQUIRED: Your per_ticker response MUST contain an entry for each of these tickers: {ticker_list}",
    ]
    return [
        {"role": "system", "content": _FIRST_RUN_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _remap_ticker(ticker: str, known: set[str]) -> str | None:
    """Try to map an unknown ticker to a known canonical ticker.

    Handles common LLM hallucination patterns:
    - Trailing share-class suffix: GOOGL -> GOOG, BRKA -> BRK-A
    - Dot vs dash separator: BRK.B -> BRK-B
    - Exact match (pass-through).
    Returns the canonical ticker, or None if no match found.
    """
    if ticker in known:
        return ticker

    # Dot-to-dash normalization (LLMs write BRK.B, yfinance uses BRK-B)
    dot_to_dash = ticker.replace(".", "-")
    if dot_to_dash in known:
        return dot_to_dash

    # Strip trailing share-class letter (GOOGL -> GOOG, GOGL -> GOG would be wrong,
    # so only strip if the base is already known)
    if len(ticker) > 1 and ticker[-1].isalpha():
        base = ticker[:-1]
        if base in known:
            return base
        # Also try dot-to-dash on the base (e.g., BRK.A -> BRK-A after stripping nothing)
        base_dashed = base.replace(".", "-")
        if base_dashed in known:
            return base_dashed

    return None


def _normalize_response(data: dict, known_tickers: set[str] | None = None) -> dict:
    """Normalize LLM response to fix common casing/type deviations before validation.

    Handles: action casing, conviction value casing, key_risk_factors as string,
    and ticker key remapping (e.g., GOOGL -> GOOG).
    """
    per_ticker = data.get("per_ticker")
    if not isinstance(per_ticker, dict):
        return data

    # Remap ticker keys when known_tickers is provided
    if known_tickers:
        remapped: dict = {}
        for ticker, entry in per_ticker.items():
            canonical = _remap_ticker(ticker, known_tickers)
            if canonical is None:
                logger.warning("Dropping unknown ticker from Agent C output: %r", ticker)
                continue
            if canonical != ticker:
                logger.info("Remapped Agent C ticker %r -> %r", ticker, canonical)
            remapped[canonical] = entry
        per_ticker = remapped
        data["per_ticker"] = per_ticker

    for entry in per_ticker.values():
        if not isinstance(entry, dict):
            continue

        # Normalize action: title-case trade actions, keep "assessment" lowercase
        action = entry.get("action")
        if isinstance(action, str):
            lower = action.lower()
            if lower == "assessment":
                entry["action"] = "assessment"
            elif lower in ("hold", "buy", "trim", "exit"):
                entry["action"] = lower.capitalize()

        # Normalize conviction values to lowercase
        conviction = entry.get("conviction")
        if isinstance(conviction, dict):
            for key in ("narrative_alignment", "quant_support", "signal_agreement"):
                val = conviction.get(key)
                if isinstance(val, str):
                    conviction[key] = val.lower()

        # Coerce key_risk_factors from string to list
        krf = entry.get("key_risk_factors")
        if isinstance(krf, str):
            entry["key_risk_factors"] = [krf] if krf else []

    return data


def _validate_per_ticker(
    per_ticker: dict,
    mode: str,
) -> tuple[bool, str]:
    """Validate the per_ticker section of Agent C output.

    Returns (is_valid, error_message). error_message is empty string on success.
    """
    if not isinstance(per_ticker, dict) or not per_ticker:
        return False, "per_ticker is missing or empty"

    for ticker, entry in per_ticker.items():
        if not isinstance(entry, dict):
            return False, f"ticker {ticker}: entry is not a dict"

        action = entry.get("action")
        if action not in VALID_ACTIONS:
            return False, f"ticker {ticker}: action {action!r} not in {VALID_ACTIONS}"

        # Mode-specific action checks
        if mode == "assessment" and action != "assessment":
            return False, f"ticker {ticker}: assessment mode requires action='assessment', got {action!r}"
        if mode in ("decision", "first_run") and action == "assessment":
            return False, f"ticker {ticker}: {mode} mode cannot use action='assessment'"

        conviction = entry.get("conviction")
        if not isinstance(conviction, dict):
            return False, f"ticker {ticker}: conviction is not a dict"

        na = conviction.get("narrative_alignment")
        if na not in VALID_CONVICTION:
            return False, f"ticker {ticker}: narrative_alignment {na!r} not in {VALID_CONVICTION}"

        qs = conviction.get("quant_support")
        sa = conviction.get("signal_agreement")

        if mode == "first_run":
            if qs != "n/a" or sa != "n/a":
                return False, f"ticker {ticker}: first_run mode requires quant_support/signal_agreement='n/a', got {qs!r}/{sa!r}"
        else:
            if qs not in VALID_CONVICTION_NA:
                return False, f"ticker {ticker}: quant_support {qs!r} not in {VALID_CONVICTION_NA}"
            if sa not in VALID_CONVICTION_NA:
                return False, f"ticker {ticker}: signal_agreement {sa!r} not in {VALID_CONVICTION_NA}"

        if not isinstance(entry.get("rationale"), str) or not entry["rationale"]:
            return False, f"ticker {ticker}: rationale is missing or empty"

        if not isinstance(entry.get("key_risk_factors"), list):
            return False, f"ticker {ticker}: key_risk_factors is not a list"

    return True, ""


def _validate_standing_actions(standing_actions: dict) -> tuple[bool, str]:
    """Validate standing_event_actions if present. Returns (is_valid, error_message)."""
    if not isinstance(standing_actions, dict):
        return False, "standing_event_actions is not a dict"

    promotions = standing_actions.get("promote_to_standing", [])
    if not isinstance(promotions, list):
        return False, "promote_to_standing is not a list"
    for p in promotions:
        if not isinstance(p, dict):
            return False, "promote_to_standing entry is not a dict"
        if not isinstance(p.get("canonical_id"), str):
            return False, "promote_to_standing entry missing canonical_id string"
        if p.get("category") not in VALID_CATEGORIES:
            return False, f"promote_to_standing entry category {p.get('category')!r} not in {VALID_CATEGORIES}"
        if not isinstance(p.get("summary"), str) or not p["summary"]:
            return False, "promote_to_standing entry missing summary"
        if not isinstance(p.get("affected_tickers"), list):
            return False, "promote_to_standing entry affected_tickers is not a list"

    resolutions = standing_actions.get("recommend_resolution", [])
    if not isinstance(resolutions, list):
        return False, "recommend_resolution is not a list"
    for r in resolutions:
        if not isinstance(r, dict):
            return False, "recommend_resolution entry is not a dict"
        if not isinstance(r.get("standing_id"), int):
            return False, "recommend_resolution entry missing standing_id int"
        if not isinstance(r.get("reason"), str):
            return False, "recommend_resolution entry missing reason string"

    return True, ""


def _validate_output(data: dict, mode: str) -> tuple[bool, str]:
    """Full output validation. Returns (is_valid, error_message)."""
    if not isinstance(data, dict):
        return False, "response is not a dict"

    per_ticker = data.get("per_ticker")
    valid, err = _validate_per_ticker(per_ticker, mode)
    if not valid:
        return False, err

    standing_actions = data.get("standing_event_actions")
    if standing_actions is not None:
        valid, err = _validate_standing_actions(standing_actions)
        if not valid:
            return False, err

    return True, ""


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

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _sanitize_json_text(raw: str) -> str:
    """Strip markdown fences and extract the first balanced JSON object.

    Handles the common LLM habit of wrapping JSON in ```json ... ``` or
    prefixing with prose like "Here is the JSON:".
    """
    if not raw:
        return raw
    text = _CODE_FENCE_RE.sub("", raw).strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text  # unbalanced — let json.loads raise a clear error


def _build_response_schema(known_tickers: set[str], mode: str) -> dict:
    """Build a JSON Schema that constrains the LLM to produce all required tickers.

    Using schema-constrained generation (Gemini response_schema / Ollama format)
    prevents the model from returning an empty or missing per_ticker dict.
    """
    if mode == "assessment":
        valid_actions = ["assessment"]
        conviction_enum = ["strong", "moderate", "weak"]
    elif mode == "first_run":
        valid_actions = ["Buy", "Hold"]
        conviction_enum = ["strong", "moderate", "weak", "n/a"]
    else:  # decision
        valid_actions = ["Buy", "Hold", "Trim", "Exit"]
        conviction_enum = ["strong", "moderate", "weak"]

    entry_schema: dict = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": valid_actions},
            "conviction": {
                "type": "object",
                "properties": {
                    "narrative_alignment": {"type": "string", "enum": conviction_enum},
                    "quant_support": {"type": "string", "enum": conviction_enum},
                    "signal_agreement": {"type": "string", "enum": conviction_enum},
                },
                "required": ["narrative_alignment", "quant_support", "signal_agreement"],
            },
            "rationale": {"type": "string"},
            "key_risk_factors": {"type": "array", "items": {"type": "string"}},
            "notable_change": {"type": "string"},
        },
        "required": ["action", "conviction", "rationale", "key_risk_factors"],
    }

    sorted_tickers = sorted(known_tickers)
    return {
        "type": "object",
        "properties": {
            "per_ticker": {
                "type": "object",
                "properties": {t: entry_schema for t in sorted_tickers},
                "required": sorted_tickers,
            },
        },
        "required": ["per_ticker"],
    }


async def _call_cloud(
    client: CloudLLMClient,
    messages: list[dict],
    mode: str,
    known_tickers: set[str] | None = None,
    schema: dict | None = None,
) -> dict | None:
    """Call cloud LLM, parse JSON, validate. Retry once on failure.

    Returns validated dict or None if both attempts fail.
    """
    last_failure_detail: str = ""
    last_raw_preview: str = ""
    for attempt in range(2):
        try:
            raw = await client.generate(messages, json_mode=True, response_schema=schema)
            last_raw_preview = (raw or "")[:500]
            data = json.loads(_sanitize_json_text(raw))
            data = _normalize_response(data, known_tickers)
            valid, err = _validate_output(data, mode)
            if valid:
                return data
            last_failure_detail = f"Schema validation failed: {err}"
            logger.warning(
                "Cloud response failed validation (attempt %d): %s",
                attempt + 1,
                err,
            )
        except CloudAuthError:
            raise
        except json.JSONDecodeError as exc:
            last_failure_detail = (
                f"Response was not valid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}"
            )
            logger.warning("Cloud call returned invalid JSON (attempt %d): %s", attempt + 1, exc)
        except CloudTimeoutError as exc:
            last_failure_detail = f"Timeout: {exc}"
            logger.warning("Cloud call timed out (attempt %d)", attempt + 1)
        except Exception as exc:
            last_failure_detail = f"Unexpected error: {type(exc).__name__}: {exc}"
            logger.exception("Unexpected error calling cloud LLM (attempt %d)", attempt + 1)

        if attempt == 0:
            ticker_list = sorted(known_tickers) if known_tickers else []
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was rejected.\n"
                        f"Reason: {last_failure_detail}\n"
                        f"Your previous response started with: {last_raw_preview!r}\n\n"
                        "You MUST respond with ONLY a single JSON object — no markdown, "
                        "no code fences, no prose. The response MUST include a non-empty "
                        f"`per_ticker` object containing an entry for every ticker in: "
                        f"{ticker_list}. Start your response with '{{'."
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
        weights = get_derived_weights(conn, price_fn=_price_stub) if holdings else {}
        # For assessment, weights are best-effort (price stub returns 1.0)
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

    # Call cloud LLM with schema-constrained generation
    known_tickers = set(tickers)
    schema = _build_response_schema(known_tickers, mode) if known_tickers else None
    result = await _call_cloud(client, messages, mode, known_tickers=known_tickers, schema=schema)

    if result is None:
        logger.error("Cloud LLM failed after retries — producing degraded output")
        result = _degraded_output(tickers, mode)

    # Store raw output
    store_agent_output(conn, run_id, "C", result)

    # Store per-ticker recommendations
    store_recommendations(conn, run_id, result["per_ticker"])

    return result
