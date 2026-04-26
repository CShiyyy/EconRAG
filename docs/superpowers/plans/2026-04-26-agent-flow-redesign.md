# Agent Flow Redesign — Narrative Modulation Model

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 27-combination conviction table and first-run/decision split with a single unified flow where Agent B weight is the gate and Agent C's narrative_score (0–10) acts as a 0.5–1.5× multiplier on that weight.

**Architecture:** Agent B produces quantitative target weights (always, including first run). Agent C reads Agent A narratives and produces a per-ticker `narrative_score` that multiplies Agent B's weight. Zero Agent B weight × any score = zero final weight, so sparsity is preserved.

**Tech Stack:** Python (pipeline), SQLite (data store), React/JSX (frontend)

---

## File Map

| File | Change |
|------|--------|
| `pipeline/sizing/conviction_map.py` | Replace all legacy functions with single `narrative_multiplier()` |
| `tests/test_sizing.py` | Replace `TestConvictionMap` class; update engine tests to new Agent C output format |
| `pipeline/agents/agent_c.py` | Unified prompt+schema, `narrative_score` output, remove first_run mode |
| `pipeline/sizing/engine.py` | New scoring formula using `narrative_multiplier`; remove first_run branch |
| `pipeline/db/helpers.py` | `store_recommendations` stores `narrative_score`+`multiplier` blob |
| `frontend/src/components/conviction/ConvictionScores.jsx` | Replace 3 conviction badges with score→multiplier display |

---

## Task 1: Replace `conviction_map.py`

**Files:**
- Modify: `pipeline/sizing/conviction_map.py`
- Modify: `tests/test_sizing.py`

- [ ] **Step 1: Write failing tests for `narrative_multiplier`**

Replace the entire `TestConvictionMap` class in `tests/test_sizing.py` with:

```python
# At top of file, update imports:
from pipeline.sizing.conviction_map import narrative_multiplier

# Replace TestConvictionMap class with:
class TestNarrativeMultiplier:
    def test_score_5_is_neutral(self):
        assert narrative_multiplier(5.0) == 1.0

    def test_score_10_is_max(self):
        assert abs(narrative_multiplier(10.0) - 1.5) < 1e-9

    def test_score_0_is_min(self):
        assert abs(narrative_multiplier(0.0) - 0.5) < 1e-9

    def test_score_above_10_clamped(self):
        assert abs(narrative_multiplier(15.0) - 1.5) < 1e-9

    def test_score_below_0_clamped(self):
        assert abs(narrative_multiplier(-5.0) - 0.5) < 1e-9

    def test_score_7_5(self):
        assert abs(narrative_multiplier(7.5) - 1.25) < 1e-9

    def test_score_2_5(self):
        assert abs(narrative_multiplier(2.5) - 0.75) < 1e-9
```

Also remove the now-dead imports from the top of `tests/test_sizing.py`:
```python
# Remove these lines:
from pipeline.sizing.conviction_map import (
    CONVICTION_TABLE,
    FIRST_RUN_TABLE,
    get_conviction_weight,
)
```

- [ ] **Step 2: Run failing tests**

```
pytest tests/test_sizing.py::TestNarrativeMultiplier -v
```
Expected: `ImportError` or `AttributeError` — `narrative_multiplier` does not exist yet.

- [ ] **Step 3: Rewrite `conviction_map.py`**

Replace the entire file content with:

```python
"""Conviction multiplier for the Position Sizing Engine.

Maps Agent C narrative_score (0–10) to a weight multiplier applied to
Agent B's mean-variance target_weight. A score of 5 is neutral (1.0×).
"""
from __future__ import annotations


def narrative_multiplier(score: float) -> float:
    """Map Agent C narrative_score (0–10) to a weight multiplier [0.5, 1.5].

    score=0  → 0.5× (strong negative narrative dampens Agent B weight by half)
    score=5  → 1.0× (neutral — Agent B weight unchanged)
    score=10 → 1.5× (strong positive narrative amplifies Agent B weight by 50%)
    """
    return 0.5 + max(0.0, min(10.0, score)) / 10.0
```

- [ ] **Step 4: Run tests — expect pass**

```
pytest tests/test_sizing.py::TestNarrativeMultiplier -v
```
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/sizing/conviction_map.py tests/test_sizing.py
git commit -m "refactor: replace conviction_map with narrative_multiplier"
```

---

## Task 2: Update `engine.py` scoring formula

**Files:**
- Modify: `pipeline/sizing/engine.py`
- Modify: `tests/test_sizing.py`

- [ ] **Step 1: Write failing tests for new engine behaviour**

Add this class to `tests/test_sizing.py` (replacing `TestEngineFirstRun`):

```python
class TestEngineNarrativeModulation:
    def test_zero_agent_b_weight_stays_zero_regardless_of_narrative_score(self, db_conn):
        """A ticker with Agent B weight=0 gets weight=0 even if narrative_score is 10."""
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
            "AAPL": {"action": "Buy",  "narrative_score": 8.0},
            "MSFT": {"action": "Buy",  "narrative_score": 6.0},
            "JPM":  {"action": "Buy",  "narrative_score": 10.0},  # high score, zero B weight
        }
        fill_prices = {"AAPL": 150.0, "MSFT": 300.0, "JPM": 200.0}

        result = run_sizing_engine(
            db_conn, run_id, agent_c_output, fill_prices,
            agent_b_output=agent_b_output,
        )

        assert result["target_weights"].get("JPM", 0.0) == 0.0
        assert result["target_weights"].get("AAPL", 0.0) > 0.0
        assert result["target_weights"].get("MSFT", 0.0) > 0.0

    def test_high_narrative_score_amplifies_weight(self, db_conn):
        """Ticker with narrative_score=10 gets larger weight than same B-weight at score=5."""
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
        # Same B-weight; AAPL gets high narrative score, MSFT neutral
        agent_c_output = {
            "AAPL": {"action": "Buy",  "narrative_score": 10.0},
            "MSFT": {"action": "Hold", "narrative_score": 5.0},
        }
        fill_prices = {"AAPL": 150.0, "MSFT": 300.0}

        result = run_sizing_engine(
            db_conn, run_id, agent_c_output, fill_prices,
            agent_b_output=agent_b_output,
        )

        # AAPL raw_score = 0.10 × 1.5 = 0.15; MSFT raw_score = 0.10 × 1.0 = 0.10
        # After normalisation AAPL should outweigh MSFT
        assert result["target_weights"]["AAPL"] > result["target_weights"]["MSFT"]
```

Also update `TestEngineAllExit` to use the new Agent C output format (remove old `conviction` dict):

```python
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
            "AAPL": {"action": "Exit", "narrative_score": 2.0},
            "MSFT": {"action": "Exit", "narrative_score": 2.0},
        }
        fill_prices = {"AAPL": 150.0, "MSFT": 300.0}

        result = run_sizing_engine(db_conn, run_id, agent_c_output, fill_prices)

        holdings = db_conn.execute("SELECT * FROM holdings").fetchall()
        assert len(holdings) == 0

        account = db_conn.execute("SELECT cash_balance FROM account WHERE account_id = 1").fetchone()
        expected_cash = 10_000.0 + (100 * 150.0) + (50 * 300.0)
        assert math.isclose(account["cash_balance"], expected_cash, abs_tol=0.01)

        for trade in result["trade_list"]:
            assert trade["action"] == "Exit"
```

- [ ] **Step 2: Run failing tests**

```
pytest tests/test_sizing.py::TestEngineNarrativeModulation tests/test_sizing.py::TestEngineAllExit -v
```
Expected: FAIL — engine still uses old scoring path.

- [ ] **Step 3: Update `engine.py`**

**3a.** Replace the import line at the top:
```python
# Remove:
from pipeline.sizing.conviction_map import FIRST_RUN_TABLE, raw_allocation_score, tilt_factor

# Add:
from pipeline.sizing.conviction_map import narrative_multiplier
```

**3b.** Replace the entire scoring block inside `run_sizing_engine` (lines 107–129). Find this block:

```python
    b_per_ticker = ({} if is_first_run else (agent_b_output or {})).get("per_ticker", {})
    a_per_ticker = (agent_a_output or {}).get("per_ticker", {})

    actions: dict[str, str] = {}
    raw_scores: dict[str, float] = {}

    for ticker, data in agent_c_output.items():
        actions[ticker] = data["action"]

        a_data = a_per_ticker.get(ticker, {})
        sentiment = a_data.get("sentiment", "neutral")
        confidence = a_data.get("confidence", "weak")

        b_data = b_per_ticker.get(ticker, {})
        b_weight = b_data.get("target_weight")

        if b_weight is not None and b_weight > 0:
            score = raw_allocation_score(b_weight, sentiment, confidence)
        else:
            # First run or ticker not covered by Agent B: use narrative table.
            na = data["conviction"].get("narrative_alignment", "weak")
            base = FIRST_RUN_TABLE.get(na, 0.25)
            score = max(0.0, base * tilt_factor(sentiment, confidence))

        raw_scores[ticker] = score
```

Replace with:

```python
    b_per_ticker = (agent_b_output or {}).get("per_ticker", {})

    actions: dict[str, str] = {}
    raw_scores: dict[str, float] = {}

    for ticker, data in agent_c_output.items():
        actions[ticker] = data["action"]
        b_weight = b_per_ticker.get(ticker, {}).get("target_weight", 0.0)
        narrative_score = float(data.get("narrative_score", 5.0))
        raw_scores[ticker] = narrative_multiplier(narrative_score) * b_weight
```

**3c.** Remove the implicit Hold block's conviction fallback. Find this block (around line 88–105):

```python
    omitted = [t for t in current_holdings if t not in agent_c_output]
    if omitted:
        logger.warning("Agent C omitted held tickers — applying implicit Hold: %s", omitted)
        for ticker in omitted:
            agent_c_output[ticker] = {
                "action": "Hold",
                "conviction": {
                    "narrative_alignment": "moderate",
                    "narrative_confidence": 5.0,
                    "quant_support": "n/a",
                    "quant_confidence": 0.0,
                    "signal_agreement": "n/a",
                    "signal_confidence": 0.0,
                },
                "rationale": "Implicit Hold: Agent C omitted this held ticker.",
                "key_risk_factors": [],
                "_implicit_hold": True,
            }
```

Replace with:

```python
    omitted = [t for t in current_holdings if t not in agent_c_output]
    if omitted:
        logger.warning("Agent C omitted held tickers — applying implicit Hold: %s", omitted)
        for ticker in omitted:
            agent_c_output[ticker] = {
                "action": "Hold",
                "narrative_score": 5.0,
                "rationale": "Implicit Hold: Agent C omitted this held ticker.",
                "key_risk_factors": [],
                "_implicit_hold": True,
            }
```

**3d.** Remove the now-unused `agent_a_output` parameter usage. The parameter can stay on the function signature (for compatibility) but remove `a_per_ticker` since it is no longer read.

- [ ] **Step 4: Run tests — expect pass**

```
pytest tests/test_sizing.py::TestEngineNarrativeModulation tests/test_sizing.py::TestEngineAllExit -v
```
Expected: all PASS.

- [ ] **Step 5: Run full test_sizing suite**

```
pytest tests/test_sizing.py -v
```
Expected: all tests PASS (old `TestConvictionMap` is replaced, all others still valid).

- [ ] **Step 6: Commit**

```bash
git add pipeline/sizing/engine.py tests/test_sizing.py
git commit -m "feat: engine uses narrative_multiplier × agent_b weight; remove first_run branch"
```

---

## Task 3: Rewrite `agent_c.py` — unified prompt and schema

**Files:**
- Modify: `pipeline/agents/agent_c.py`

This task has no new test (Agent C is a cloud LLM call; the existing `tests/test_agent_c.py` mocks the cloud call). Update `tests/test_agent_c.py` after implementing.

- [ ] **Step 1: Replace `_DECISION_SYSTEM` and `_FIRST_RUN_SYSTEM` with unified `_SYSTEM`**

Remove both `_DECISION_SYSTEM` and `_FIRST_RUN_SYSTEM` string constants. Add:

```python
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
      "rationale": "NVDA commands the strongest narrative in the watchlist. \
Agent A highlights accelerating data centre order flow and multiple sell-side upgrades \
overnight, with high confidence in the bullish read. Compared to other Information \
Technology names in the watchlist, NVDA has the most specific near-term catalyst. \
Agent B assigns a 14% target weight reflecting strong multi-factor momentum and quality \
scores. The narrative reinforces rather than contradicts the quant signal — both point to \
continued outperformance. Supply chain risk is the primary concern given Taiwan concentration. \
Export restriction escalation remains a tail risk but is not the current consensus expectation. \
Overall the combined evidence strongly supports increasing exposure.",
      "key_risk_factors": ["export restriction escalation", "Taiwan supply chain concentration"]
    },
    "KO": {
      "narrative_score": 4.5,
      "action": "Hold",
      "rationale": "KO narrative is mildly negative. Agent A finds limited recent coverage \
with the most recent theme being volume softness in emerging markets and FX headwinds. \
Among Consumer Staples names in the watchlist, KO ranks below average on narrative momentum. \
Agent B assigns a 0% target weight reflecting weak quantitative signals. The narrative is \
consistent with the quant view — neither provides a reason to initiate. Confidence in the \
narrative read is moderate given thin coverage. The stock offers defensiveness but no near-term \
catalyst. A score of 4.5 reflects mild negative narrative without a clear bear thesis.",
      "key_risk_factors": ["EM volume softness", "FX headwinds on repatriation"]
    }
  }
}"""
```

- [ ] **Step 2: Add `_format_agent_b_weights_context` helper**

Add this function after `_format_agent_b_context` (keep `_format_agent_b_context` — it may still be useful for assessment mode):

```python
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
```

- [ ] **Step 3: Replace `_build_decision_prompt` and `_build_first_run_prompt` with unified `_build_prompt`**

Remove both old functions. Add:

```python
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
```

- [ ] **Step 4: Update `_normalize_response`**

Replace the conviction normalization block inside `_normalize_response`. Find:

```python
        # Normalize conviction values to lowercase; clamp confidence scores.
        conviction = entry.get("conviction")
        if isinstance(conviction, dict):
            for key in ("narrative_alignment", "quant_support", "signal_agreement"):
                val = conviction.get(key)
                if isinstance(val, str):
                    conviction[key] = val.lower()
            for conf_key in ("narrative_confidence", "quant_confidence", "signal_confidence"):
                raw = conviction.get(conf_key)
                if raw is not None:
                    try:
                        conviction[conf_key] = max(0.0, min(10.0, float(raw)))
                    except (TypeError, ValueError):
                        conviction[conf_key] = 5.0
                else:
                    conviction[conf_key] = 5.0
```

Replace with:

```python
        # Coerce narrative_score to float in [0, 10]; default 5.0 (neutral).
        ns = entry.get("narrative_score")
        if ns is not None:
            try:
                entry["narrative_score"] = max(0.0, min(10.0, float(ns)))
            except (TypeError, ValueError):
                entry["narrative_score"] = 5.0
        else:
            entry["narrative_score"] = 5.0
```

Also in `_normalize_response`, remove the `"assessment"` branch from the action normalization:

```python
        # Normalize action: title-case
        action = entry.get("action")
        if isinstance(action, str):
            lower = action.lower()
            if lower in ("hold", "buy", "trim", "exit"):
                entry["action"] = lower.capitalize()
```

(Remove the `elif lower == "assessment"` branch — it was only needed for assessment mode which has its own code path.)

- [ ] **Step 5: Update `_validate_per_ticker`**

Replace the entire function:

```python
def _validate_per_ticker(per_ticker: dict, mode: str) -> tuple[bool, str]:
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
```

- [ ] **Step 6: Update `_build_response_schema`**

Replace the entire function:

```python
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
```

- [ ] **Step 7: Update `_degraded_output`**

Replace the `first_run` branch inside `_degraded_output`:

```python
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
```

- [ ] **Step 8: Update `run_agent_c` — remove first_run / decision branching**

Replace the `if is_first_run` block inside `run_agent_c`. Find:

```python
    if is_first_run:
        mode = "first_run"
        watchlist = get_watchlist(conn)
        tickers = [w["ticker"] for w in watchlist]
        messages = _build_first_run_prompt(
            agent_a_output, source_health, constraints, watchlist,
            agent_b_output=agent_b_output,
        )
    else:  # decision
        mode = "decision"
        holdings = get_holdings(conn)
        watchlist = get_watchlist(conn)
        weights = {}
        held_tickers = {h["ticker"] for h in holdings}
        watchlist_tickers = {w["ticker"] for w in watchlist}
        tickers = sorted(watchlist_tickers | held_tickers)
        previous_assessment = get_previous_assessment(conn)
        messages = _build_decision_prompt(
            agent_a_output, agent_b_output or {}, source_health,
            holdings, weights, constraints, standing_events,
            previous_assessment, watchlist,
        )
```

Replace with:

```python
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
```

- [ ] **Step 9: Remove now-dead imports from `agent_c.py`**

Remove `VALID_CONVICTION` and `VALID_CONVICTION_NA` from the module-level constants (they are no longer referenced). Keep `VALID_ACTIONS` and `VALID_CATEGORIES`.

- [ ] **Step 10: Run the agent_c tests**

```
pytest tests/test_agent_c.py -v
```
Expected: tests PASS (mocked cloud call — update any test fixture that constructs old conviction dicts to use `narrative_score` instead if tests fail).

- [ ] **Step 11: Commit**

```bash
git add pipeline/agents/agent_c.py
git commit -m "feat: agent_c unified prompt with narrative_score output"
```

---

## Task 4: Update `store_recommendations` in `helpers.py`

**Files:**
- Modify: `pipeline/db/helpers.py`

- [ ] **Step 1: Update `store_recommendations`**

Find the `cursor = conn.execute(...)` call inside `store_recommendations`. The line that writes `conviction_scores` currently reads:

```python
json.dumps(entry.get("conviction", {})),
```

And `conviction_weight` is written as `None`.

Replace those two values:

```python
        ns = float(entry.get("narrative_score", 5.0))
        multiplier = round(0.5 + max(0.0, min(10.0, ns)) / 10.0, 4)

        cursor = conn.execute(
            """INSERT INTO recommendations
               (run_id, timestamp, ticker, action, conviction_scores,
                conviction_weight, rationale, key_quant_metrics, key_risk_factors,
                requery_triggered, requery_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                ts,
                ticker,
                entry["action"],
                json.dumps({"narrative_score": ns, "multiplier": multiplier}),
                multiplier,
                entry.get("rationale", ""),
                json.dumps(quant_metrics) if quant_metrics else None,
                json.dumps(entry.get("key_risk_factors", [])),
                1 if requery_triggered else 0,
                requery_reason,
            ),
        )
```

- [ ] **Step 2: Verify with a quick smoke test**

```
python -c "
import sqlite3, json
from pipeline.db.schema import create_tables
conn = sqlite3.connect(':memory:')
conn.row_factory = sqlite3.Row
create_tables(conn)
# Minimal run_log row
conn.execute(\"INSERT INTO run_log (run_id, timestamp, run_type) VALUES (1, 'now', 'scheduled')\")
conn.commit()
from pipeline.db.helpers import store_recommendations
per_ticker = {
    'AAPL': {'action': 'Buy', 'narrative_score': 8.0, 'rationale': 'test', 'key_risk_factors': []},
    'MSFT': {'action': 'Hold', 'narrative_score': 5.0, 'rationale': 'test', 'key_risk_factors': []},
}
store_recommendations(conn, 1, per_ticker)
rows = conn.execute('SELECT ticker, conviction_scores, conviction_weight FROM recommendations').fetchall()
for r in rows:
    print(r['ticker'], json.loads(r['conviction_scores']), r['conviction_weight'])
"
```

Expected output:
```
AAPL {'narrative_score': 8.0, 'multiplier': 1.3} 1.3
MSFT {'narrative_score': 5.0, 'multiplier': 1.0} 1.0
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/db/helpers.py
git commit -m "feat: store_recommendations writes narrative_score and multiplier"
```

---

## Task 5: Update frontend conviction display

**Files:**
- Modify: `frontend/src/components/conviction/ConvictionScores.jsx`

- [ ] **Step 1: Rewrite `ConvictionScores.jsx`**

Replace the entire file:

```jsx
export default function ConvictionScores({ scores }) {
  if (!scores) return <span className="text-gray-400">—</span>;

  const ns = scores.narrative_score;
  const multiplier = scores.multiplier;

  if (ns == null) return <span className="text-gray-400">—</span>;

  const nsColor =
    ns >= 7 ? 'text-green-600' :
    ns >= 4 ? 'text-yellow-600' :
    'text-red-600';

  return (
    <span className="text-xs tabular-nums whitespace-nowrap">
      <span
        className={nsColor}
        title={`Narrative score ${ns.toFixed(1)}/10 — Agent C's qualitative assessment of narrative evidence`}
      >
        {ns.toFixed(1)}
      </span>
      <span className="text-gray-400"> → </span>
      <span
        className="text-gray-700"
        title={`Multiplier applied to Agent B's quantitative weight (0.5–1.5×)`}
      >
        {multiplier != null ? `${multiplier.toFixed(2)}×` : '—'}
      </span>
    </span>
  );
}
```

- [ ] **Step 2: Verify no broken imports**

```
grep -r "ConvictionBadge\|LABELS\|TOOLTIPS\|conviction_map" frontend/src/
```
Expected: no results (old references removed).

- [ ] **Step 3: Start dev server and visually check Recommendations page**

```
cd frontend && npm run dev
```

Open `http://localhost:5173/recommendations` and confirm:
- The "Conviction" column shows e.g. `8.5 → 1.35×` in green/yellow/red
- The "Score" sortable column sorts by `conviction_weight` (the multiplier) correctly
- Existing rows with old conviction blob show `—` gracefully (no crash)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/conviction/ConvictionScores.jsx
git commit -m "feat: conviction display shows narrative_score → multiplier"
```

---

## Self-Review Checklist

- [x] `conviction_map.py` — only `narrative_multiplier()` remains; all legacy exports removed
- [x] `engine.py` — scoring formula is `narrative_multiplier(score) × b_weight`; `is_first_run` guard removed; implicit Hold uses `narrative_score: 5.0`
- [x] `agent_c.py` — single `_SYSTEM` prompt; single `_build_prompt()`; `narrative_score` validated [0,10]; degraded fallback uses `narrative_score: 5.0`; assessment mode preserved untouched
- [x] `helpers.py` — `store_recommendations` writes `{"narrative_score": ..., "multiplier": ...}` blob and `multiplier` to `conviction_weight`
- [x] `ConvictionScores.jsx` — renders `ns → multiplier×` with colour coding; handles missing `narrative_score` gracefully
- [x] Tests updated: `TestConvictionMap` → `TestNarrativeMultiplier`; engine tests use `narrative_score` format; `TestEngineAllExit` uses `narrative_score`
- [x] No `FIRST_RUN_TABLE`, `tilt_factor`, `raw_allocation_score`, `get_conviction_weight`, `CONVICTION_TABLE` references remain in non-test production code
