# Agent Flow Redesign — Narrative Modulation Model

**Date:** 2026-04-26
**Branch:** Improve-initialization
**Status:** Approved, ready for implementation

---

## Problem Statement

The current pipeline has two compounding defects:

1. **Agent B is excluded on the first run.** `engine.py` has an explicit `if is_first_run` branch that drops Agent B's weights and falls back to `FIRST_RUN_TABLE` — a static lookup that assigns positive base scores to every ticker based purely on Agent C's `narrative_alignment` label. This produces a near-equal-weight spread across all tickers on first run regardless of quantitative signal.

2. **Agent C spreads conviction too broadly.** In first-run mode the prompt forces `quant_support='n/a'` and `signal_agreement='n/a'`, collapsing the 27-combination conviction table to a single dimension. Agent C then assigns "strong" narrative alignment to the majority of tickers, which — combined with `FIRST_RUN_TABLE` base scores — floods the normalizer with near-equal positive scores across 18+ names.

The root cause is architectural: Agent C is acting as both the quantitative synthesizer and the primary weight generator, rather than as a narrative moderator on top of Agent B's signal.

---

## New Architecture

### Pipeline Order (unchanged)

```
Agent A → Agent B → Agent C → Sizing Engine
```

### Role Redefinition

| Agent | Role | Output |
|-------|------|--------|
| Agent A | Qualitative intelligence | Per-ticker narrative, sentiment, confidence, catalysts; macro overview |
| Agent B | Quantitative backbone | Per-ticker target weights from Second Tower mean-variance optimizer; always runs |
| Agent C | Narrative moderator | Per-ticker `narrative_score` (0–10); action label (UX only); standing event actions |

### Core Sizing Formula

```
multiplier[t]  = 0.5 + narrative_score[t] / 10.0          # range [0.5, 1.5]
raw_score[t]   = Agent_B.target_weight[t] × multiplier[t]
```

**Agent B weight is the gate.** If `target_weight[t] == 0`, then `raw_score[t] == 0` regardless of `narrative_score`. The narrative can modulate existing quantitative signals but cannot manufacture new ones. If Agent B signals 8 tickers, the final portfolio has at most 8 positions.

---

## Agent C — New Output Schema

```json
{
  "per_ticker": {
    "TICKER": {
      "narrative_score": 7.5,
      "action": "Buy | Hold | Trim | Exit",
      "rationale": "7-10 sentences citing Agent A narrative evidence, placing the ticker in context relative to peers, referencing Agent B target weight, and assessing the primary risk.",
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
```

### Narrative Score Guidance (in prompt)

| Score | Multiplier | Meaning |
|-------|-----------|---------|
| 8–10  | 1.30–1.50× | Strong positive — clear bullish catalysts, high confidence |
| 6–7   | 1.10–1.20× | Mild positive — net positive, mixed or thin evidence |
| 5     | 1.00×      | Neutral — no directional signal, Agent B weight unchanged |
| 3–4   | 0.80–0.90× | Mild negative — headwinds or deteriorating narrative |
| 0–2   | 0.50–0.70× | Strong negative — clear bearish catalysts |

### Mode Unification

The `first_run` / `decision` mode distinction in Agent C is **eliminated**. There is a single unified prompt. Agent B's target weights are always available and always shown to Agent C. Agent C's job is always the same: produce a narrative score per ticker.

`Trim` and `Exit` action labels remain restricted to tickers in current holdings (validation unchanged).

---

## Sizing Engine Changes (`engine.py`)

- Remove the `is_first_run` guard at line 107 that excluded Agent B weights
- Replace the scoring block with:
  ```python
  b_weight = b_per_ticker.get(ticker, {}).get("target_weight", 0.0)
  narrative_score = data.get("narrative_score", 5.0)
  raw_scores[ticker] = narrative_multiplier(narrative_score) * b_weight
  ```
- Agent C `action` field is extracted for `actions[ticker]` (display/storage) but drives no branching
- `is_first_run` parameter retained on `run_sizing_engine()` for compatibility with orchestration callers but has no effect on scoring logic

---

## `conviction_map.py` Changes

**Remove:**
- `FIRST_RUN_TABLE`
- `tilt_factor()`
- `raw_allocation_score()`
- `apply_confidence_modifier()`
- `get_conviction_weight()`
- `CONVICTION_TABLE`
- `_compute_table_weight()`
- `_LEVEL_SCORE`, `_DIM_WEIGHTS`, `_SENTIMENT_TILT`

**Add:**
```python
def narrative_multiplier(score: float) -> float:
    """Map Agent C narrative_score (0–10) to a weight multiplier [0.5, 1.5]."""
    return 0.5 + max(0.0, min(10.0, score)) / 10.0
```

The file becomes the single canonical location for the multiplier formula.

---

## Agent C Prompt Changes (`agent_c.py`)

- **Remove** `_FIRST_RUN_SYSTEM` prompt and `_build_first_run_prompt()`
- **Remove** `_DECISION_SYSTEM` prompt and `_build_decision_prompt()`
- **Add** single unified `_SYSTEM` prompt and `_build_prompt()` function
- Prompt receives: Agent A per-ticker narratives, Agent B target weights (all tickers), portfolio state, constraints, standing events, watchlist
- Prompt instructs: produce `narrative_score` 0–10 per ticker; score 5 = neutral (no change to Agent B weight); score >5 amplifies; score <5 dampens; scores should use the full range to differentiate tickers
- **Remove** `_build_response_schema()` conviction sub-score fields; add `narrative_score` as required float [0, 10]
- **Remove** `_validate_per_ticker()` checks for `quant_support`/`signal_agreement`; add check that `narrative_score` is a float in [0, 10]
- Degraded fallback: `narrative_score=5.0` (neutral — no distortion of Agent B's signal)

---

## Database & API Changes

### `recommendations` table

`conviction_scores` JSON blob (no schema migration needed) changes from:
```json
{"narrative_alignment": "strong", "narrative_confidence": 8.0, "quant_support": "n/a", ...}
```
to:
```json
{"narrative_score": 8.5, "multiplier": 1.35}
```

`conviction_weight` column stores the computed `multiplier` (float 0.5–1.5).

### `db/helpers.py` — `store_recommendations`

- Extract `narrative_score` from Agent C per-ticker output
- Compute `multiplier = 0.5 + narrative_score / 10.0`
- Write `{"narrative_score": ..., "multiplier": ...}` to `conviction_scores`
- Write `multiplier` to `conviction_weight`
- Remove `quant_support`/`signal_agreement` extraction

### `schemas.py` and `backend/routers/recommendations.py`

No changes needed. There is no typed `Recommendation` Pydantic model — the router passes raw dicts. The `conviction_scores` column is already JSON-parsed by `_parse_recommendation()` and the new shape flows through automatically.

### Frontend (`RecommendationsPage.jsx`)

Replace three conviction badge columns with a single narrative score + multiplier display: `Score: 8.5 → 1.35×`.

---

## Files Changed

| File | Change |
|------|--------|
| `pipeline/agents/agent_c.py` | Unify prompts, new schema, new validation |
| `pipeline/sizing/engine.py` | New scoring formula, remove first_run branch |
| `pipeline/sizing/conviction_map.py` | Remove all legacy functions, add `narrative_multiplier()` |
| `pipeline/db/helpers.py` | Update `store_recommendations` |
| `frontend/src/pages/RecommendationsPage.jsx` | Update conviction display |

---

## What Does Not Change

- Agent A output format and prompt — unchanged
- Agent B output format and Second Tower strategy — unchanged
- `normalizer.py` — unchanged (receives `raw_scores`, enforces constraints identically)
- `trade_builder.py` — unchanged
- Standing event processing — unchanged
- Re-query logic (`conditions.py`) — unchanged
- All other orchestration, ingestion, and knowledge graph code — unchanged
