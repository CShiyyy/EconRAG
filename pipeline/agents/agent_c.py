"""Agent C — Cloud LLM Portfolio Synthesizer.

Operates in decision mode: produces per-ticker narrative_score (0-10) which
the sizing engine uses as a multiplier on Agent B's quantitative weights.
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


def _format_agent_b_weights_context(agent_b_output: dict) -> str:
    """Format Agent B target weights for the unified prompt."""
    per_ticker = (agent_b_output or {}).get("per_ticker", {})
    if not per_ticker:
        return "QUANTITATIVE WEIGHTS: No data available."
    lines = ["QUANTITATIVE OPTIMIZER TARGET WEIGHTS (Second Tower multi-factor):"]
    for ticker, data in sorted(per_ticker.items()):
        tw = data.get("target_weight", 0.0)
        vol = data.get("volatility_30d", 0.0)
        health = data.get("health_score", "")
        lines.append(
            f"  {ticker}: target_weight={tw:.4f}, vol_30d={vol:.4f}, health={health}"
        )
    return "\n".join(lines)


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


_SYSTEM = """\
You are a narrative moderator for a quantitative portfolio manager. \
A Second Tower mean-variance optimizer (Agent B) has already computed target weights \
for each ticker based on multi-factor signals (momentum, value, quality, risk). \
Your job is to moderate those weights using qualitative narrative evidence from Agent A.

For each ticker produce a JSON entry with:
  - narrative_score: float 0–10 (see scale below)
  - action: "Buy" | "Hold" | "Trim" | "Exit" (for display only — does not affect weights)
  - rationale: 7–10 sentences citing Agent A narrative evidence for this ticker, \
comparing its narrative to peers in the watchlist, referencing Agent B's target_weight, \
and assessing the primary risk to the thesis
  - key_risk_factors: list of 2–3 specific risks

NARRATIVE SCORE SCALE:
  8–10 → multiplier 1.30–1.50× — strong positive: clear bullish catalysts, high confidence
  6–7  → multiplier 1.10–1.20× — mild positive: net positive, mixed or thin evidence
  5    → multiplier 1.00×       — neutral: no directional signal, Agent B weight unchanged
  3–4  → multiplier 0.80–0.90× — mild negative: headwinds or deteriorating narrative
  0–2  → multiplier 0.50–0.70× — strong negative: clear bearish catalysts

RULES:
- You MUST produce a per_ticker entry for EVERY ticker in the WATCHLIST plus every ticker \
in CURRENT HOLDINGS.
- Use ONLY the exact ticker symbols provided. Do NOT invent tickers.
- Use the FULL 0–10 range to differentiate tickers. Do NOT cluster all scores near 5.
- A ticker with Agent B target_weight=0 will receive zero allocation regardless of score; \
still score it honestly for the record.
- "Trim" and "Exit" are valid ONLY for tickers that appear in CURRENT HOLDINGS.
- Never emit "Trim" or "Exit" for a ticker not in CURRENT HOLDINGS.
- standing_event_actions is optional — omit or leave arrays empty if none warranted.
- Respond with ONLY valid JSON, no additional text, no markdown fences.

OUTPUT SCHEMA:
{
  "per_ticker": {
    "TICKER": {
      "narrative_score": 7.5,
      "action": "Buy | Hold | Trim | Exit",
      "rationale": "7-10 sentences...",
      "key_risk_factors": ["risk1", "risk2"]
    }
  },
  "standing_event_actions": {
    "promote_to_standing": [
      {"canonical_id": "...", "category": "...", "summary": "...", "affected_tickers": [...]}
    ],
    "recommend_resolution": [
      {"standing_id": 123, "reason": "..."}
    ]
  }
}

EXAMPLE (2 tickers):
{
  "per_ticker": {
    "NVDA": {
      "narrative_score": 8.5,
      "action": "Buy",
      "rationale": "NVDA commands the strongest narrative in the watchlist. Agent A highlights accelerating data centre order flow and multiple sell-side upgrades overnight, with high confidence in the bullish read. Compared to other Information Technology names in the watchlist, NVDA has the most specific near-term catalyst. Agent B assigns a 14% target weight reflecting strong multi-factor momentum and quality scores. The narrative reinforces rather than contradicts the quant signal — both point to continued outperformance. Supply chain risk is the primary concern given Taiwan concentration. Export restriction escalation remains a tail risk but is not the current consensus expectation. Overall the combined evidence strongly supports increasing exposure.",
      "key_risk_factors": ["export restriction escalation", "Taiwan supply chain concentration"]
    },
    "KO": {
      "narrative_score": 4.5,
      "action": "Hold",
      "rationale": "KO narrative is mildly negative. Agent A finds limited recent coverage with the most recent theme being volume softness in emerging markets and FX headwinds. Among Consumer Staples names in the watchlist, KO ranks below average on narrative momentum. Agent B assigns a 0% target weight reflecting weak quantitative signals. The narrative is consistent with the quant view — neither provides a reason to initiate. Confidence in the narrative read is moderate given thin coverage. The stock offers defensiveness but no near-term catalyst. A score of 4.5 reflects mild negative narrative without a clear bear thesis.",
      "key_risk_factors": ["EM volume softness", "FX headwinds on repatriation"]
    }
  }
}"""


def _build_prompt(
    agent_a_output: dict,
    agent_b_output: dict | None,
    source_health: list[dict],
    holdings: list[dict],
    weights: dict[str, float],
    constraints: dict[str, float],
    standing_events: list[dict],
    previous_assessment: dict | None,
    watchlist: list[dict],
) -> list[dict]:
    """Build messages for the unified decision prompt."""
    tickers_info = "\n".join(
        f"  {w['ticker']}: {w['company_name']} ({w['sector']})"
        for w in watchlist
    )
    held_tickers = [h["ticker"] for h in holdings]
    universe = sorted({w["ticker"] for w in watchlist} | set(held_tickers))
    ticker_list = ", ".join(universe)

    user_parts = [
        _format_agent_a_context(agent_a_output),
        _format_agent_b_weights_context(agent_b_output or {}),
        _format_source_health(source_health),
        _format_holdings_context(holdings, weights),
        _format_standing_context(standing_events),
        f"CONSTRAINTS: {json.dumps(constraints)}",
        f"WATCHLIST (candidate universe):\n{tickers_info}",
        (
            "REQUIRED: Your per_ticker response MUST contain an entry for each of "
            f"these tickers: {ticker_list}. Trim/Exit are valid only for tickers "
            f"in CURRENT HOLDINGS ({', '.join(held_tickers) if held_tickers else 'none'})."
        ),
    ]
    if previous_assessment:
        user_parts.append(
            f"PREVIOUS ASSESSMENT:\n{json.dumps(previous_assessment, indent=2)}"
        )
    return [
        {"role": "system", "content": _SYSTEM},
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

    Handles: action casing, narrative_score coercion, key_risk_factors as string,
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

        # Normalize action: title-case trade actions
        action = entry.get("action")
        if isinstance(action, str):
            lower = action.lower()
            if lower in ("hold", "buy", "trim", "exit"):
                entry["action"] = lower.capitalize()

        # Coerce narrative_score to float in [0, 10]; default 5.0 (neutral).
        ns = entry.get("narrative_score")
        if ns is not None:
            try:
                entry["narrative_score"] = max(0.0, min(10.0, float(ns)))
            except (TypeError, ValueError):
                entry["narrative_score"] = 5.0
        else:
            entry["narrative_score"] = 5.0

        # Coerce key_risk_factors from string to list
        krf = entry.get("key_risk_factors")
        if isinstance(krf, str):
            entry["key_risk_factors"] = [krf] if krf else []

    return data


def _validate_per_ticker(
    per_ticker: dict,
    mode: str,
) -> tuple[bool, str]:
    """Validate the per_ticker section of Agent C output."""
    if not isinstance(per_ticker, dict) or not per_ticker:
        return False, "per_ticker is missing or empty"

    for ticker, entry in per_ticker.items():
        if not isinstance(entry, dict):
            return False, f"ticker {ticker}: entry is not a dict"

        action = entry.get("action")
        if mode == "assessment":
            if action != "assessment":
                return False, f"ticker {ticker}: assessment mode requires action='assessment', got {action!r}"
        else:
            if action not in {"Buy", "Hold", "Trim", "Exit"}:
                return False, f"ticker {ticker}: action {action!r} must be one of Buy/Hold/Trim/Exit"

        if mode != "assessment":
            ns = entry.get("narrative_score")
            if ns is None or not isinstance(ns, (int, float)):
                return False, f"ticker {ticker}: narrative_score missing or not numeric"
            if not (0.0 <= float(ns) <= 10.0):
                return False, f"ticker {ticker}: narrative_score {ns} out of range [0, 10]"

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
        if mode == "assessment":
            per_ticker[ticker] = {
                "action": "assessment",
                "conviction": {
                    "narrative_alignment": "weak",
                    "narrative_confidence": 2.0,
                    "quant_support": "weak",
                    "quant_confidence": 2.0,
                    "signal_agreement": "weak",
                    "signal_confidence": 2.0,
                },
                "rationale": "Cloud LLM extraction failed — using degraded output.",
                "key_risk_factors": [],
                "notable_change": None,
            }
        else:
            per_ticker[ticker] = {
                "narrative_score": 5.0,
                "action": "Hold",
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
    """Build JSON Schema to constrain LLM output."""
    if mode == "assessment":
        entry_schema: dict = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["assessment"]},
                "conviction": {
                    "type": "object",
                    "properties": {
                        "narrative_alignment": {"type": "string"},
                        "narrative_confidence": {"type": "number", "minimum": 0, "maximum": 10},
                        "quant_support": {"type": "string"},
                        "quant_confidence": {"type": "number", "minimum": 0, "maximum": 10},
                        "signal_agreement": {"type": "string"},
                        "signal_confidence": {"type": "number", "minimum": 0, "maximum": 10},
                    },
                    "required": [
                        "narrative_alignment", "narrative_confidence",
                        "quant_support", "quant_confidence",
                        "signal_agreement", "signal_confidence",
                    ],
                },
                "rationale": {"type": "string"},
                "key_risk_factors": {"type": "array", "items": {"type": "string"}},
                "notable_change": {"type": "string"},
            },
            "required": ["action", "conviction", "rationale", "key_risk_factors"],
        }
    else:
        entry_schema = {
            "type": "object",
            "properties": {
                "narrative_score": {"type": "number", "minimum": 0, "maximum": 10},
                "action": {"type": "string", "enum": ["Buy", "Hold", "Trim", "Exit"]},
                "rationale": {"type": "string"},
                "key_risk_factors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["narrative_score", "action", "rationale", "key_risk_factors"],
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
        agent_b_output: Agent B's quant JSON, or None if Second Tower could not produce weights.
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

    mode = "decision"
    holdings = get_holdings(conn)
    watchlist = get_watchlist(conn)
    weights: dict[str, float] = {}
    held_tickers = {h["ticker"] for h in holdings}
    watchlist_tickers = {w["ticker"] for w in watchlist}
    tickers = sorted(watchlist_tickers | held_tickers)
    previous_assessment = get_previous_assessment(conn)
    messages = _build_prompt(
        agent_a_output, agent_b_output, source_health,
        holdings, weights, constraints, standing_events,
        previous_assessment, watchlist,
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

    # Store per-ticker recommendations (pass quant_assessment so Agent B metrics land in key_quant_metrics)
    store_recommendations(conn, run_id, result["per_ticker"], quant_assessment=agent_b_output)

    return result
