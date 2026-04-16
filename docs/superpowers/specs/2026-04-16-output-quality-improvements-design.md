# Output Quality Improvements Design

**Date**: 2026-04-16
**Branch**: Improve-initialization
**Status**: Design approved, pending implementation

## Context

After running the first pipeline execution (pre-open on Dow 30), four quality issues were observed:

1. Portfolio value + cash does not equal initial capital
2. Weight allocations are near-equally distributed across all 30 tickers
3. Seeded profiles are shallow (3-4 sentences per ticker)
4. Recommendation rationales are short, and the "quant metrics" section displays wrong data

The system currently runs on `gemma3:4b` via Ollama for all LLM tasks (seeding, extraction, Agent C). All prompt and output format changes are designed to optimize for this 4B parameter model using few-shot examples as the primary lever.

---

## Section 1: Price Model Fix

### Problem

Trades execute at `open_price` (yfinance) but snapshots value holdings at `current_price`. On a pre-open run these prices differ, so portfolio value + cash != initial capital immediately after first trades.

### Design

Introduce run-type-aware pricing:

| Run Type | Fill Price (trade execution) | Valuation Price (snapshot) |
|----------|------------------------------|----------------------------|
| `pre_open` | `previous_close` | `previous_close` |
| `post_close` | `open_price` | `current_price` |

**Rationale**: Pre-open, the most recent settled price is the previous close. Post-close, trades happened at the open and the portfolio should be marked to the closing price.

### Files to Change

- `pipeline/orchestration/execution_graph.py` (lines 44-48) — `position_sizing_node` selects fill price based on `run_type` from state
- `pipeline/orchestration/snapshots.py` (line 42) — `record_snapshot` selects valuation price based on `run_type`
- No changes to `pipeline/ingestion/market_data.py` — `previous_close` is already fetched (line 29)

---

## Section 2: Hybrid Conviction Scoring

### Problem

First-run conviction map has only 3 categorical levels (strong/moderate/weak). The LLM defaults to "moderate" for most tickers, producing near-equal weights (~3.17% each) after normalization across 30 tickers.

### Design

**Hybrid approach**: Each conviction dimension outputs both a categorical level (5-level) and a numeric confidence score (0.0-10.0).

#### New Agent C conviction output format

```json
"conviction": {
    "narrative_alignment": "strong",
    "narrative_confidence": 7.5,
    "quant_support": "moderate",
    "quant_confidence": 5.0,
    "signal_agreement": "n/a",
    "signal_confidence": 0.0
}
```

#### Expanded categorical levels

```python
FIRST_RUN_TABLE = {
    "very_strong": 0.90,
    "strong":      0.70,
    "moderate":    0.45,
    "weak":        0.25,
    "very_weak":   0.10,
}
```

Full `CONVICTION_TABLE` expands from 27 (3x3x3) to 125 (5x5x5) combinations. Generate programmatically using a weighted-sum formula rather than hand-writing all 125 entries: `weight = 0.4 * na_score + 0.35 * qs_score + 0.25 * sa_score` where each level maps to a base value (very_strong=1.0, strong=0.8, moderate=0.5, weak=0.3, very_weak=0.1).

#### Weight calculation

```
final_weight = base_weight * (0.7 + 0.06 * numeric_confidence)
```

- Confidence 0.0 → 70% of base weight
- Confidence 5.0 → 100% of base weight (neutral)
- Confidence 10.0 → 130% of base weight

Two "strong" tickers with confidence 3.0 vs 8.0 get weights 0.588 vs 0.826.

#### First run

Only `narrative_alignment` + `narrative_confidence` are used. `quant_support`/`signal_agreement` = "n/a", confidence = 0.0.

#### Validation

Confidence values are validated as floats in [0.0, 10.0]. Out-of-range values are clamped.

#### No distribution constraint

The numeric confidence provides natural differentiation without forcing a distribution.

### Files to Change

- `pipeline/sizing/conviction_map.py` — expand `VALID_SCORES`, expand `FIRST_RUN_TABLE` and `CONVICTION_TABLE`, add `apply_confidence_modifier()` function
- `pipeline/sizing/engine.py` — pass confidence scores through to weight calculation
- `pipeline/agents/agent_c.py` — update all 3 system prompts with 5-level scale + confidence fields, add few-shot examples, update validation

---

## Section 3: Deeper Seeded Profiles

### Problem

Ticker profiles are ~3-4 sentences. The prompt asks for only a "Current state" paragraph. Grounding data is thin (1500-char business summary cap, 5 headlines, sparse macro indicators).

### Design

#### Part A: Enrich grounding data

**`TickerFactPack` additions**:
- Remove 1500-char business summary cap (increase to 4000 chars)
- Increase headline cap from 5 to 10
- Add fields: `revenue`, `net_income`, `free_cash_flow`, `dividend_yield`, `beta`, `debt_to_equity` (all from yfinance `info`)

**`MacroFactPack` additions**:
- `gold_price` (GLD ticker)
- `oil_price` (USO or CL=F ticker)
- `sector_performance`: 1-month returns for top sector ETFs (XLF, XLK, XLE, XLV, XLI)

#### Part B: Expand profile prompt

Replace the "3-4 sentences Current state" with multi-section structure:

```
## REQUIRED SECTIONS (in order)
1. Opening sentences: use the REQUIRED PHRASES verbatim
2. Business model (2-3 sentences): revenue drivers, competitive moat
3. Financial snapshot (2-3 sentences): market cap, P/E, revenue, margins
4. Current catalysts and risks (3-4 sentences): ground in headlines and price action
5. Closing: "As of {date}."

Target length: ~400-500 words total.
```

#### Part C: Few-shot example

Include one complete ~400-word example profile in the system prompt. Use a generic but realistic company example to teach `gemma3:4b` the expected depth and style without leaking content into real profiles.

#### Part D: Macro profile expansion

- Include oil, gold, and sector performance data in the prompt
- Add a "Market breadth and risk appetite" section
- Target ~300-400 words

### Files to Change

- `pipeline/knowledge/profile_grounding.py` — add new fields to `TickerFactPack` and `MacroFactPack`, increase caps, fetch additional data
- `pipeline/knowledge/profile_seeder.py` — rewrite `_build_ticker_messages()` and `_build_macro_messages()` prompts, add few-shot examples
- `pipeline/ingestion/models.py` — add fields if needed for sector performance data

---

## Section 4: Rationale Depth & Quant Metrics Fix

### Problem

Three distinct issues:
1. **Data bug**: `helpers.py:162` stores `key_risk_factors` into the `key_quant_metrics` column
2. **Short rationales**: Agent C prompts have no length guidance for the rationale field
3. **Raw display**: Frontend uses `JSON.stringify()` on quant metrics, conviction abbreviations are unexplained

### Design

#### Part A: Fix `key_quant_metrics` data bug

Add a `key_risk_factors TEXT` column to the `recommendations` table. Update `store_recommendations()`:

- `key_quant_metrics` → stores actual Agent B metrics (drift, volatility_30d, health_score, current_weight, flags)
- `key_risk_factors` → stores Agent C's risk factors list

The function signature changes to accept `quant_assessment` (Agent B output dict) so per-ticker metrics can be looked up.

#### Part B: Agent C rationale length guidance + few-shot

All 3 Agent C system prompts get:
- Rationale field description updated to: `"3-5 sentence analysis covering: (1) narrative evidence from Agent A, (2) quant metrics from Agent B if available, (3) risk/reward assessment"`
- A few-shot example showing one complete ticker entry with a ~60-80 word rationale referencing both narrative and quant data

#### Part C: Frontend structured quant metrics card

Replace `JSON.stringify(r.key_quant_metrics)` in `RecommendationsPage.jsx` with a structured grid:

```
Drift: +0.5%        30d Vol: 22.3%
Health: [normal]     Weight: 3.2%
Flags: [drift_above_threshold]
```

Color-code health scores (green/yellow/red). Add tooltips to conviction abbreviations:
- NA → "Narrative Alignment"
- QS → "Quant Support"
- SA → "Signal Agreement"

Add a separate "Risk Factors" display for the new `key_risk_factors` column (bulleted list).

### Files to Change

- `pipeline/db/schema.py` — add `key_risk_factors TEXT` column to recommendations table
- `pipeline/db/helpers.py` — fix `store_recommendations()` to accept quant data and store correctly
- `pipeline/orchestration/reasoning_graph.py` — pass `quant_assessment` to `store_recommendations()`
- `pipeline/agents/agent_c.py` — update rationale descriptions and add few-shot examples
- `frontend/src/pages/RecommendationsPage.jsx` — structured metrics card + risk factors display
- `frontend/src/components/conviction/ConvictionScores.jsx` — full labels or tooltips
- `backend/schemas.py` — add `key_risk_factors` to response model
- `backend/routers/recommendations.py` — include new field in API response

---

## Verification Plan

### Unit Tests
1. **Price model**: Test `get_fill_price()` returns `previous_close` for pre_open, `open_price` for post_close
2. **Conviction scoring**: Test all 125 combinations produce valid weights; test confidence modifier at 0.0, 5.0, 10.0 boundaries
3. **Store recommendations**: Verify `key_quant_metrics` contains Agent B data and `key_risk_factors` contains Agent C data

### Integration Tests
1. Run the full pipeline in pre_open mode on a small watchlist (3-5 tickers). Verify:
   - Portfolio value + cash == initial capital (within float tolerance)
   - Weight allocations show meaningful differentiation (max/min weight ratio > 2x)
   - Seeded profiles are 300+ words
   - Rationales are 3+ sentences
   - Quant metrics contain drift/volatility/health data

### Frontend Verification
1. Start dev server and navigate to Recommendations page
2. Verify quant metrics show labeled rows, not raw JSON
3. Verify conviction scores show full labels or tooltips
4. Verify risk factors display as a bulleted list
