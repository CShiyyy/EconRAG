# Phase 5 Design Spec: Agent A & Agent B

**Date:** 2026-04-06
**Phase:** 5 of 9
**Goal:** Both agents produce structured JSON outputs. Agent A queries LightRAG and returns per-ticker sentiment via local Ollama. Agent B computes quant metrics against computed targets.

---

## Agent A — Context Retriever (Local LLM)

### Architecture: Two-Stage Retrieval-Then-Synthesis

**Stage 1 — Context Retrieval via LightRAG:**
- Use `query_graph(rag, query, only_context=True)` to pull context without LLM generation
- Three query types:
  1. **Macro query:** "What are the dominant market narratives in the last 48 hours?" (global mode)
  2. **Standing context query:** "How do the following ongoing conditions affect the current picture?" with active standing event summaries injected (global mode)
  3. **Per-ticker query:** "What events, catalysts, or sentiment shifts affect [TICKER]?" (local mode) — append standing event summary for tickers in `affected_tickers`

**Stage 2 — Structured Synthesis via Direct Ollama:**
- Call Ollama directly (`ollama.chat()` with `format="json"`) for each query type
- System prompt enforces the exact output schema
- One call for macro overview, one for standing context assessment, one per tracked ticker

### Output Schema
```json
{
  "macro_overview": "200-word narrative summary",
  "standing_context_assessment": {
    "<canonical_id>": {
      "still_relevant": true,
      "current_impact": "...",
      "affected_tickers_update": ["TICKER1", "TICKER2"]
    }
  },
  "per_ticker": {
    "TICKER": {
      "sentiment": "bullish | bearish | neutral",
      "confidence": "strong | moderate | weak",
      "key_catalysts": ["catalyst1", "catalyst2"],
      "narrative": "100-150 word summary"
    }
  }
}
```

### Output Validation
- Parse JSON response from Ollama
- Validate: `sentiment` in `{bullish, bearish, neutral}`, `confidence` in `{strong, moderate, weak}`, required fields present, `key_catalysts` is a list
- On parse/validation failure: retry once with corrective prompt
- On second failure: produce degraded entry `{sentiment: "neutral", confidence: "weak", key_catalysts: [], narrative: "Extraction failed"}` and log warning

### Re-Query Mode
- Accept `flagged_tickers: list[str]` parameter
- For each flagged ticker, re-run LightRAG query with broadened search terms (add sector name, competitor names)
- Merge updated entries into existing output (replace only affected tickers)
- Does not re-run macro or standing context queries

### Storage
- Serialize full output JSON to `agent_outputs` table with `agent='A'`

### Key Dependencies
- `pipeline/knowledge/lightrag_config.py` — `rag_session()` for LightRAG lifecycle
- `pipeline/knowledge/graph_ops.py` — `query_graph()` for context retrieval
- `pipeline/db/helpers.py` — standing events query (new helper needed)
- `pipeline/config.py` — `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `ollama` package — direct LLM calls

---

## Agent B — Quant Analyst (Deterministic Script)

### Architecture: Pure Python, No LLM

**First-run guard:** If `is_first_run(conn)` returns `True`, return `None` immediately.

### Data Assembly
1. Current holdings + market prices → current weights via `get_derived_weights(conn, price_fn)`
2. Previous targets via `get_previous_computed_target(conn)`
3. Constraints via `get_constraints(conn)`
4. Recent snapshots (last 30) from `snapshots` table for drawdown
5. 30-day price histories from `MarketDataPoint.price_history_30d`

### Calculations

**Per-Ticker Drift:**
- `drift = current_weight - previous_target_weight`
- Health flags: `warning` if `|drift| > DRIFT_WARNING_THRESHOLD` (0.03), `breach` if `|drift| > DRIFT_BREACH_THRESHOLD` (0.05)

**Per-Ticker Volatility (30-day):**
- Compute daily log returns from `price_history_30d`
- Annualized volatility = `std(log_returns) * sqrt(252)`
- Health flags: `warning` if volatility > 1.5x of rolling mean, `breach` if > 2x

**Portfolio-Level Volatility:**
- Weighted sum using portfolio weights and covariance matrix of daily log returns

**Sector Concentration:**
- Sum current weights per sector
- Compare against `max_sector_concentration` constraint
- Status: `normal`, `warning` (within 5% of limit), or `breach` (exceeds limit)

**Max Drawdown (30-day):**
- From `snapshots.total_value` (last 30 entries)
- `max_drawdown = (trough - peak) / peak` where peak is running maximum

**Constraint Violations:**
- Check all 4 constraints: cash floor, single position cap, sector cap, dust floor
- Return list of violation descriptions

**Overall Status:**
- `breach` if any constraint violation or any per-ticker breach
- `warning` if any per-ticker warning or sector warning
- `normal` otherwise

### Output Schema
```json
{
  "portfolio_level": {
    "total_value": 100000.00,
    "cash_pct": 0.08,
    "portfolio_volatility_30d": 0.18,
    "max_drawdown_30d": -0.04,
    "overall_status": "normal | warning | breach"
  },
  "per_ticker": {
    "TICKER": {
      "current_weight": 0.12,
      "previous_target_weight": 0.10,
      "drift": 0.02,
      "volatility_30d": 0.35,
      "sector": "semiconductors",
      "health_score": "normal | warning | breach",
      "flags": ["drift_above_threshold"]
    }
  },
  "sector_concentrations": {
    "sector_name": { "weight": 0.28, "limit": 0.35, "status": "normal" }
  },
  "constraint_violations": []
}
```

### Correlation Edge Injection
- Compute pairwise Pearson correlation from 30-day price histories of all held tickers
- Call `inject_correlation_edges(rag, correlations, run_id)` from `graph_ops.py`
- Format: `[{"ticker_a": str, "ticker_b": str, "correlation": float}, ...]`
- Existing CORRELATED_WITH edges are removed first (handled by `inject_correlation_edges`)

### Storage
- Serialize output JSON to `agent_outputs` table with `agent='B'`

### Key Dependencies
- `pipeline/db/helpers.py` — `get_previous_computed_target()`, `get_derived_weights()`, `get_portfolio_value()`, `get_constraints()`, `is_first_run()`, `get_account()`, `get_watchlist()`
- `pipeline/knowledge/graph_ops.py` — `inject_correlation_edges()`
- `pipeline/knowledge/lightrag_config.py` — `rag_session()` for correlation injection
- `pipeline/ingestion/models.py` — `MarketDataPoint`
- `pipeline/config.py` — threshold constants
- `numpy` / `pandas` — for volatility, correlation, drawdown calculations

---

## Config Additions

Add to `pipeline/config.py`:
```python
# Agent B health score thresholds
DRIFT_WARNING_THRESHOLD: float = 0.03
DRIFT_BREACH_THRESHOLD: float = 0.05
VOLATILITY_WARNING_MULTIPLIER: float = 1.5
VOLATILITY_BREACH_MULTIPLIER: float = 2.0
SECTOR_WARNING_BUFFER: float = 0.05  # warn when within 5% of limit
```

---

## New Helper Needed

Add to `pipeline/db/helpers.py`:
```python
def get_active_standing_events(conn) -> list[dict]:
    """Return all standing events with status='active'."""
```

Add to `pipeline/db/helpers.py`:
```python
def store_agent_output(conn, run_id: int, agent: str, output: dict) -> int:
    """Insert agent output JSON into agent_outputs table. Returns output_id."""
```

---

## File Structure

```
pipeline/agents/
  __init__.py          — public exports
  agent_a.py           — retrieval strategy, Ollama calls, output parsing, re-query
  agent_b.py           — quant calculations, graph injection, output formatting
tests/
  test_agent_a.py      — mocked LightRAG + Ollama responses
  test_agent_b.py      — known portfolio states, price histories
```

---

## Test Criteria

### Agent A Tests
- Given pre-populated LightRAG graph, verify output JSON has correct structure
- Per-ticker sentiment is one of `{bullish, bearish, neutral}`
- Standing context assessment is included when standing events exist
- Re-query updates only the flagged ticker's entry
- Corrective prompt fires on malformed JSON; degraded output on double failure
- Output stored to `agent_outputs` with correct `agent='A'`

### Agent B Tests
- Returns `None` when `is_first_run()` is `True`
- Drift calculations match expected values for known holdings and targets
- 30-day volatility matches expected value for known price series
- Sector concentration flags breaches correctly
- Max drawdown calculation is correct for known snapshot series
- Constraint violations detected and reported
- `CORRELATED_WITH` edges injected with correct correlation values and `source: "agent_b"`
- Output stored to `agent_outputs` with correct `agent='B'`
