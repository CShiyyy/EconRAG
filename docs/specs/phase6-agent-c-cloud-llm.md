# Phase 6 Design Spec: Agent C — Cloud LLM Integration

**Date:** 2026-04-06
**Phase:** 6 of 9
**Goal:** Agent C calls the cloud API in both assessment and decision modes, produces validated structured JSON, and handles API errors. Starts with Gemini 1.5 Pro, abstracted for future Claude support.

---

## Cloud Client (`pipeline/agents/cloud_client.py`)

### Protocol

```python
class CloudLLMClient(Protocol):
    async def generate(self, messages: list[dict], json_mode: bool = True) -> str:
        """Send messages to cloud LLM, return raw response text."""
        ...
```

### GeminiClient Implementation

- Uses `google-generativeai` SDK (`google.generativeai`)
- Constructor: `model_name` (from config), `api_key` (from config), `temperature`, `max_tokens`
- Converts `messages` list (system/user role dicts) to Gemini's `GenerativeModel.generate_content_async()` format:
  - System message → `system_instruction` parameter
  - User messages → `contents` parameter
- When `json_mode=True`, sets `generation_config.response_mime_type = "application/json"`
- Returns `response.text`

### Factory

```python
def create_cloud_client() -> CloudLLMClient:
    """Read CLOUD_PROVIDER from config, return appropriate client."""
```

### Error Handling & Retry

Built into the client:
- **Rate limits (429 / `ResourceExhausted`):** Exponential backoff, up to `CLOUD_RETRY_MAX_ATTEMPTS` (default 3)
- **Timeouts:** Single retry, then raise
- **Auth errors (401/403 / `PermissionDenied`):** Raise immediately (halts the run)
- Backoff formula: `CLOUD_RETRY_BASE_DELAY * 2^attempt` seconds

---

## Agent C Module (`pipeline/agents/agent_c.py`)

### Entry Point

```python
async def run_agent_c(
    conn: sqlite3.Connection,
    agent_a_output: dict,
    agent_b_output: dict | None,
    source_health: list[dict],
    run_id: int,
    run_type: str,            # "pre_open" | "post_close"
    is_first_run: bool,
    client: CloudLLMClient | None = None,  # injectable for testing
) -> dict:
```

### Internal Flow

1. Gather portfolio state: `get_derived_weights()`, `get_constraints()`, `get_active_standing_events()`, `get_holdings()`, `get_account()`
2. If `run_type == "pre_open"` and not first run: `get_previous_assessment(conn)` for prior post-close context
3. Select prompt builder:
   - `is_first_run` → `_build_first_run_prompt`
   - `run_type == "post_close"` → `_build_assessment_prompt`
   - `run_type == "pre_open"` → `_build_decision_prompt`
4. Call `client.generate(messages, json_mode=True)`
5. Parse and validate JSON response
6. On failure: retry once with corrective prompt; on second failure: produce degraded output
7. Store raw output to `agent_outputs` (agent='C')
8. Store parsed per-ticker entries to `recommendations`
9. Return parsed output dict (including `standing_event_actions` for orchestrator)

### Prompt Builders

**`_build_assessment_prompt(agent_a_output, agent_b_output, source_health, holdings, constraints, standing_context)`**

System prompt instructs:
- Role: "You are the lead portfolio manager synthesizing end-of-day analysis"
- Mode: Assessment only — produce updated conviction scores, no trade actions
- Output must be JSON matching the assessment schema
- Evaluate each held ticker against the day's data

User message assembles:
- Agent A macro overview + per-ticker analysis
- Agent B quant metrics (portfolio health, per-ticker drift/vol)
- Source health status (which sources succeeded/failed)
- Current holdings with derived weights
- Active standing events with summaries
- Portfolio constraints

**`_build_decision_prompt(agent_a_output, agent_b_output, source_health, holdings, constraints, standing_context, previous_assessment)`**

Same as assessment but:
- Instructs trade actions (Buy/Hold/Trim/Exit) + conviction scores
- Includes previous post-close assessment as additional context
- May recommend standing event promotions or resolutions

**`_build_first_run_prompt(agent_a_output, source_health, constraints, watchlist)`**

- No Agent B output (null)
- No previous assessment
- No holdings (100% cash)
- Provides full watchlist for the model to recommend initial positions from
- Instructs `narrative_alignment`-only scoring: `quant_support: "n/a"`, `signal_agreement: "n/a"`

### Output Schemas

**Assessment Mode:**
```json
{
  "per_ticker": {
    "NVDA": {
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
```

**Decision Mode:**
```json
{
  "per_ticker": {
    "NVDA": {
      "action": "Buy | Hold | Trim | Exit",
      "conviction": {
        "narrative_alignment": "strong | moderate | weak",
        "quant_support": "strong | moderate | weak | n/a",
        "signal_agreement": "strong | moderate | weak | n/a"
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
        "affected_tickers": ["TICKER1", "TICKER2"]
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
```

### Validation

**Per-ticker validation:**
- `action` in `{"Hold", "Buy", "Trim", "Exit", "assessment"}`
- `conviction.narrative_alignment` in `{"strong", "moderate", "weak"}`
- `conviction.quant_support` in `{"strong", "moderate", "weak", "n/a"}`
- `conviction.signal_agreement` in `{"strong", "moderate", "weak", "n/a"}`
- `rationale` is non-empty string
- `key_risk_factors` is a list of strings

**Mode-specific validation:**
- Assessment mode: all tickers must have `action: "assessment"`
- Decision mode: no ticker should have `action: "assessment"`
- First run: `quant_support` and `signal_agreement` must be `"n/a"`

**Standing event action validation:**
- `promote_to_standing`: `category` in valid set, `summary` non-empty, `affected_tickers` is a list
- `recommend_resolution`: `standing_id` is an integer

**Retry & Degraded Output:**
- On parse/validation failure: one retry with corrective prompt (same pattern as Agent A)
- On second failure: degraded output — `action: "assessment"` for all tickers, `narrative_alignment: "weak"`, `quant_support: "weak"`, `signal_agreement: "weak"`, rationale: "Cloud LLM extraction failed"

---

## New DB Helpers (`pipeline/db/helpers.py`)

```python
def store_recommendations(
    conn: sqlite3.Connection,
    run_id: int,
    per_ticker: dict[str, dict],
    requery_triggered: bool = False,
    requery_reason: str | None = None,
) -> list[int]:
    """Insert per-ticker recommendations from Agent C output.

    Each ticker entry becomes a row in the recommendations table.
    conviction_scores stored as JSON string. conviction_weight is null
    (computed later by Position Sizing Engine for decision mode).
    Returns list of recommendation_ids.
    """

def get_previous_assessment(conn: sqlite3.Connection) -> dict | None:
    """Get the most recent post-close assessment from the recommendations table.

    Queries for the latest run_id where all entries have action='assessment'.
    Returns dict keyed by ticker with conviction scores and rationale,
    or None if no post-close run has occurred yet.
    """

def get_holdings(conn: sqlite3.Connection) -> list[dict]:
    """Return all current holdings rows as dicts."""
```

---

## Config Additions (`pipeline/config.py`)

```python
# --- Cloud LLM config ---
CLOUD_PROVIDER: str = os.getenv("CLOUD_PROVIDER", "gemini")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
CLOUD_TEMPERATURE: float = float(os.getenv("CLOUD_TEMPERATURE", "0.3"))
CLOUD_MAX_TOKENS: int = int(os.getenv("CLOUD_MAX_TOKENS", "8192"))
CLOUD_RETRY_MAX_ATTEMPTS: int = 3
CLOUD_RETRY_BASE_DELAY: float = 1.0
```

---

## Package Exports (`pipeline/agents/__init__.py`)

Add `run_agent_c` and `CloudLLMClient` to the public API:

```python
from pipeline.agents.agent_c import run_agent_c
from pipeline.agents.cloud_client import CloudLLMClient, create_cloud_client

__all__ = ["run_agent_a", "run_agent_b", "run_agent_c", "CloudLLMClient", "create_cloud_client"]
```

---

## File Structure

```
pipeline/agents/
  __init__.py          — updated exports
  cloud_client.py      — CloudLLMClient protocol + GeminiClient + factory
  agent_c.py           — prompt construction, API call, response parsing, both modes
tests/
  test_agent_c.py      — mocked API responses for all modes, validation, error handling
```

---

## Test Plan

### Assessment Mode
- Mock Gemini returning valid assessment JSON
- Verify output has `action: "assessment"` for all tickers
- Verify valid conviction scores and non-empty rationale
- Verify stored to `agent_outputs` with `agent='C'`
- Verify stored to `recommendations` with correct fields

### Decision Mode
- Mock Gemini returning valid decision JSON with Buy/Hold/Trim/Exit actions
- Verify conviction scores are valid categorical values
- Verify rationale and key_risk_factors present
- Verify storage to both tables

### First-Run Mode
- Mock first-run response recommending initial positions
- Verify `quant_support: "n/a"` and `signal_agreement: "n/a"` for all tickers
- Verify positions recommended from the watchlist universe

### Standing Event Actions
- Mock response containing `promote_to_standing` entries
- Verify correct extraction with required fields (canonical_id, category, summary, affected_tickers)
- Mock response containing `recommend_resolution` entries
- Verify standing_id references are extracted

### Validation Failures
- Mock malformed JSON response → verify retry fires with corrective prompt
- Mock second malformed response → verify degraded output produced
- Verify degraded output has correct structure (assessment action, weak conviction)

### API Errors
- Mock 429 rate limit → verify retry with exponential backoff
- Mock timeout → verify single retry then degraded output
- Mock 401 auth error → verify immediate exception raised (no retry)

### Storage
- Verify `agent_outputs` row created with `agent='C'` and full raw response
- Verify `recommendations` rows created with correct per-ticker data
- Verify `get_previous_assessment()` retrieves the latest assessment run
