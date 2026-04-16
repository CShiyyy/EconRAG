# Architectural Blueprint: Autonomous Portfolio Monitoring Agent

**System Overview:** A multi-agent orchestration system for US equities sentiment and fundamental monitoring, utilizing local hardware for data processing and cloud intelligence for strategic synthesis. Operates in **simulation-only mode** on a virtual portfolio with no live broker integration — but designed as if it were real, to support a future transition to live trading.

**Scope:** US equities only. Scheduling, data sources, and market assumptions are tied to NYSE/NASDAQ hours. The scheduler and ingestion interfaces are designed per-asset-class to allow future extension to crypto, forex, or other markets without architectural changes.

**Tech Stack:**
- **Backend / Pipeline:** Python, FastAPI (serves both the REST API for the web app and orchestrates the pipeline).
- **Frontend:** React (SPA communicating with the FastAPI backend).
- **Database:** SQLite (WAL mode) via `sqlite3` or SQLAlchemy.
- **Local LLM:** Ollama (Gemma 4 E4B).
- **Agent Orchestration:** LangGraph.
- **Knowledge Graph:** LightRAG (NanoVectorDB + NetworkX).

**Implementation Status (as of 2026-04-16):**
Phases 1–9 are complete and committed. Phase 8 (FastAPI backend) and Phase 9 (React frontend) are implemented. The Improve-initialization branch adds profile seeding (§1.1), admin reset, and an OllamaClient for the seeder. Scheduler (`APScheduler`) and notification webhooks (`Discord/Telegram`) are not yet implemented. The LangGraph orchestration (§3, §6) is implemented as three subgraphs (`data_graph`, `reasoning_graph`, `execution_graph`) composed by a parent graph, rather than the single flat graph described in this document. A shared `pipeline/config.py` module centralizes paths, API keys, and model settings.

**Ticker Universe:** The user selects a pre-defined index universe at initialization (S&P 500, Nasdaq 100, or Dow Jones 30). The system pulls the constituent ticker list from a public source (e.g., Wikipedia tables) and uses this as the tracked universe for ingestion, extraction, and Agent A queries. The universe can be changed via the UI; changes take effect on the next run.

---

## 1. Data Ingestion Layer (The Senses)
*Scheduled twice daily: Pre-Open (~8:30 AM ET) and Post-Close (~4:30 PM ET).*

| Source | Tooling | Filtering Logic |
| :--- | :--- | :--- |
| **Market Data** | `yfinance` / Polygon.io | EOD and pre-market prices, fundamental ratios, key Greeks for options-exposed positions. |
| **News** | NewsData.io / GNews | 48-hour rolling window (`timeframe=48h`). Relevance-scored — see §2 on pruning. |
| **Social** | PRAW (Reddit) / X API | Top posts for cashtags in the active watchlist. Filtered to US equities only. |
| **Parsing** | **Crawl4AI** | Converts raw URLs from news/social hits into clean, LLM-ready Markdown. |

### Data Quality & Resilience
Each source has an independent health check. Per-run status (success, timeout, rate-limited, stale) is logged to the **Run Log** (§2). Failure of any single source does **not** block the pipeline — the run proceeds with degraded input, and Agent C is informed of which sources were unavailable via the state object.

**Entity Extraction & Resolution Pipeline:** Extraction uses **LightRAG's built-in LLM extraction**, constrained by a custom prompt template that enforces the project's closed ontology (see §2.2 for full schema). The pipeline is:

1. Crawl4AI outputs clean Markdown.
2. LightRAG's extraction prompt (customized with the closed ontology) runs on each chunk via the local Gemma 4 E4B, producing typed entities and typed relationships.
3. **Canonical Registry** resolves raw entity strings to canonical IDs before graph insertion (e.g., "NVIDIA," "Nvidia Corp," "Jensen's company" → `NVDA`). See §2.2.
4. **Post-Extraction Validator** discards or remaps any entity types or relationship types the LLM produced outside the closed ontology.
5. Validated, canonicalized entities and edges are inserted into LightRAG's graph.

### 1.1 Profile Seeding (Initialization Bootstrap)

Before the first ingest runs, Agent A would query an empty LightRAG graph and produce no useful output. The profile seeder solves this cold-start problem by generating structured baseline profiles at initialization time.

**Flow (per run, idempotent):**
1. `seed_profiles_node` runs as the first node of the data subgraph.
2. SQL check: which watchlist tickers are missing from `kg_seed_log`? Which profiles already exist are skipped.
3. For each pending ticker: `fetch_ticker_fact_pack()` calls yfinance concurrently (company info + news headlines). Produces a `TickerFactPack` with sector, industry, CEO, market cap, P/E, 52-week range, business summary, peer tickers, and recent headlines. Falls back to minimal structural pack if yfinance is unavailable.
4. Cloud LLM (Gemini or Ollama, per `SEEDING_LLM_PROVIDER`) generates a grounded markdown profile using the fact pack. Strict prompt instructions prevent hallucination of financials not in the fact pack.
5. The markdown is run through the standard extract→validate→insert pipeline (same as normal ingestion). Tier 2 edges from seeding receive an extended TTL (`PROFILE_SEED_TTL_HOURS`, default 720 hours / 30 days) to survive routine pruning cycles.
6. `kg_seed_log` row is written (only on full success), making the operation idempotent.
7. Macro profile: `fetch_macro_fact_pack()` fetches 10Y and 3M Treasury yields, VIX, DXY, SPY, QQQ levels and macro-relevant news headlines. Cloud LLM generates a macro environment snapshot using these facts.
8. Ticker profiles run with bounded concurrency (`PROFILE_SEED_CONCURRENCY`, default 3). Macro profile runs sequentially after tickers.

**Configuration:**
- `PROFILE_SEED_ENABLED` (default: true) — set to false to disable seeding entirely.
- `PROFILE_SEED_TTL_HOURS` (default: 720) — how long seeded Tier 2 edges persist before pruning.
- `PROFILE_SEED_CONCURRENCY` (default: 3) — max concurrent ticker profile generations.
- `SEEDING_LLM_PROVIDER` (default: "auto") — "auto" uses Gemini if `GEMINI_API_KEY` is set, else Ollama. "gemini" or "ollama" force the choice.

---

## 2. Memory & Knowledge Layer (The Context)

### 2.1 Quantitative Store — SQLite (Local, WAL Mode)

The database is the system's single source of truth. Use `PRAGMA journal_mode=WAL` from initialization to allow concurrent reads (web app) during writes (pipeline runs).

**Schema:**

**`account`** — Global account state. Single row.
- `account_id` (PK, always 1), `cash_balance`, `initial_cash` (immutable after initialization, for total return calculations), `universe` (text: "sp500" / "nasdaq100" / "djia30"), `initialized_at` (timestamp).

**`watchlist`** — The active ticker universe, derived from the selected index. Populated at initialization and refreshable.
- `ticker` (PK), `company_name`, `sector` (GICS sector), `added_at` (timestamp).
- Sourced from Wikipedia constituent tables (e.g., "List of S&P 500 companies") at initialization and on manual refresh.
- This table defines the universe for ingestion (which cashtags to query), Agent A (which tickers to run per-ticker queries for), and Agent C (what it can recommend).
- Holdings must always be a subset of the watchlist. If a ticker is removed from the index (and thus the watchlist) while the portfolio holds it, the system flags it for review rather than auto-selling.

**`constraints`** — User-configurable portfolio rules. Set at initialization, adjustable via UI.
- `constraint_name` (PK), `value` (REAL), `description` (text).
- **Default rows:**

| Constraint | Default | Description |
| :--- | :--- | :--- |
| `cash_floor` | `0.05` | Minimum portfolio fraction held in cash at all times. |
| `max_single_position` | `0.15` | Maximum weight for any single ticker. |
| `max_sector_concentration` | `0.35` | Maximum combined weight for all tickers in one sector. |
| `min_position_size` | `0.02` | Below this weight, a position is not worth opening (avoids dust trades). |

**`holdings`** — Current simulated portfolio state. Mutated after every executed trade.
- `ticker` (PK), `shares`, `cost_basis_per_share`, `sector` (cached GICS sector for constraint checks).
- **Weightings are never stored.** They are derived at query time as `(shares × current_price) / total_portfolio_value`.

**`computed_targets`** — The target allocation computed by each run's Position Sizing Engine (§4.1). Append-only.
- `target_id` (PK), `run_id` (FK), `timestamp`, `per_ticker_json` (JSON blob: ticker → target_weight, conviction_weight, action).
- Agent B measures drift against the **most recent row** in this table. On the very first run, no previous target exists; Agent B is skipped (see §6).

**`snapshots`** — Append-only time series. One full portfolio snapshot per run.
- `snapshot_id` (PK), `run_id` (FK), `timestamp`, `total_value`, `cash`, `per_ticker_json` (JSON blob: ticker → shares, price, value, weight at that moment).
- Never updated or deleted. This is the foundation for future calibration.

**`recommendations`** — Agent C's raw output, stored *before* any simulated execution.
- `recommendation_id` (PK), `run_id` (FK), `timestamp`, `ticker`, `action` (Hold/Buy/Trim/Exit/**assessment**), `conviction_scores` (JSON, see §4 Agent C output schema), `conviction_weight` (REAL, computed by Position Sizing Engine — `null` for assessment-mode entries), `rationale` (text), `key_quant_metrics` (JSON from Agent B, nullable on first run), `requery_triggered` (boolean), `requery_reason` (text, nullable).
- Post-close assessment entries have `action: "assessment"` and `conviction_weight: null`. They record the system's end-of-day conviction state and serve as input context for the next pre-open decision run.

**`trades`** — Simulated execution log, referencing the recommendation that triggered it. Only produced by pre-open runs.
- `trade_id` (PK), `recommendation_id` (FK), `ticker`, `action`, `shares`, `simulated_fill_price`, `fill_type` (open_price), `gap_pct` (REAL, nullable — percentage gap between previous close and fill price, for future analysis), `slippage_applied` (REAL, default 0.0 — configurable for future realism), `realized_pnl` (REAL, nullable — populated on Trim/Exit trades: `(fill_price - cost_basis) × shares`), `timestamp`.
- Separated from recommendations to support future features: vetoes, position limits, cooldown periods, manual overrides.

**`run_log`** — One row per system run. Diagnostic metadata.
- `run_id` (PK), `timestamp`, `run_type` (pre_open / post_close / non_trading_day), `is_first_run` (boolean), `source_status` (JSON: per-source success/failure), `requery_triggered` (boolean), `requery_reason` (text, nullable), `wall_clock_seconds`.

**`agent_outputs`** — Raw Agent A, Agent B, and Agent C outputs, keyed to run.
- `output_id` (PK), `run_id` (FK), `agent` (A/B/C), `output_blob` (text/JSON).
- Small per-run but irreplaceable for future calibration. Log generously, compute lazily.
- Agent C's raw LLM response is also stored here (in addition to the parsed `recommendations` rows) so the full reasoning chain is preserved.

**`canonical_entities`** — The Canonical Entity Registry. Maps raw extracted strings to canonical IDs.
- `canonical_id` (PK), `entity_type` (COMPANY/PERSON/SECTOR/INDEX/PRODUCT/EVENT/MACRO_THEME/INSTITUTION), `display_name`, `aliases` (JSON array of known raw strings that map to this ID).
- Append-only for new entities. Aliases can be updated. Pre-seeded with tracked tickers and their common variants.

**`standing_events`** — Persistent macro conditions that outlive the 48-hour ephemeral window. See §2.2.4 for the three-tier persistence model.
- `standing_id` (PK), `canonical_id` (FK to `canonical_entities`), `status` (active / resolved), `category` (geopolitical / monetary_policy / regulatory / trade_policy / sector_crisis / other), `summary` (text — compressed context, periodically refreshed), `affected_tickers` (JSON array of ticker symbols this standing event is relevant to), `promoted_at` (timestamp), `promotion_source` (auto / manual / agent_c), `last_reinforced` (timestamp — most recent run where ingested content referenced this event), `reinforcement_count` (INTEGER — total number of runs that have referenced this event), `stale_run_threshold` (INTEGER, default 28 — number of consecutive runs with zero references before flagging for review), `resolved_at` (timestamp, nullable), `created_from_run_id` (FK to `run_log`).
- Active standing events are **never pruned** from the LightRAG graph. Their associated nodes and edges persist regardless of the ephemeral TTL logic.
- When resolved, the graph nodes are not deleted but are marked `status: resolved` and excluded from Agent A's active query set.

**`kg_seed_log`** — Idempotency log for LightRAG profile seeding. Tracks which ticker and macro profiles have been seeded into LightRAG at initialization.
- `seed_type` + `seed_key` (PK composite), `seeded_at` (timestamp), `run_id` (FK to `run_log`, nullable), `source_id` (text, e.g. `"profile:NVDA"`), `seed_text` (text — the full markdown profile generated by the LLM, for audit/display).
- Written by the profile seeder after successful extract→validate→insert. Never overwritten except via `INSERT OR REPLACE` (reseed).
- Exposed by `GET /api/profiles` and `GET /api/profiles/{key}`.

**Total: 13 tables** (`account`, `watchlist`, `constraints`, `holdings`, `computed_targets`, `snapshots`, `recommendations`, `trades`, `run_log`, `agent_outputs`, `canonical_entities`, `standing_events`, `kg_seed_log`).

### 2.2 Qualitative Store — LightRAG (Local)

- **Vector DB:** NanoVectorDB.
- **Graph Engine:** NetworkX.
- **Extraction Engine:** LightRAG's built-in LLM extraction with a **custom prompt template** that constrains output to the closed ontology defined below. The default extraction prompt is overridden to enumerate all valid entity types, relationship types, and provide 1-2 examples of each. This keeps extraction within LightRAG's native pipeline while giving the project control over graph structure.

#### 2.2.1 Canonical Entity Registry

A lookup table (SQLite table or in-memory dictionary) that maps raw extracted entity strings to canonical identifiers. Resolves synonyms, abbreviations, and informal references before graph insertion.

**Entity Types (Closed Set):**

| Entity Type | Canonical ID Format | Examples |
| :--- | :--- | :--- |
| `COMPANY` | Ticker symbol | `NVDA`, `AAPL`, `TSMC` |
| `PERSON` | `person:<normalized_name>` | `person:jensen_huang`, `person:jerome_powell` |
| `SECTOR` | `sector:<gics_name>` | `sector:semiconductors`, `sector:software` |
| `INDEX` | `index:<symbol>` | `index:SPX`, `index:NDX` |
| `PRODUCT` | `product:<company>:<n>` | `product:NVDA:H100`, `product:AAPL:iPhone` |
| `EVENT` | `event:<yyyymmdd>:<slug>` | `event:20260401:nvda_q1_earnings` |
| `MACRO_THEME` | `theme:<slug>` | `theme:ai_capex_cycle`, `theme:china_export_restrictions` |
| `INSTITUTION` | `inst:<n>` | `inst:federal_reserve`, `inst:sec` |

**Registry Behavior:**
- On first encounter of an unknown string, the extraction LLM proposes a canonical ID and entity type. A validator checks format compliance before insertion.
- Known aliases are pre-seeded for tracked tickers (e.g., "Nvidia" / "NVIDIA" / "nVidia" → `NVDA`).
- The registry is persisted in SQLite alongside the main schema. New entries are append-only; manual review can flag mismatches.

#### 2.2.2 Closed Relationship Ontology

Every edge in the graph must be one of the following types. LightRAG's custom extraction prompt enumerates these types with definitions and examples. The post-extraction validator discards or remaps any edge whose type is not in this set.

**Tier 1 — Structural Relationships** *(Relatively stable. Updated weekly or via registry seeding. Form the graph skeleton.)*

| Relationship | Direction | Definition | Example |
| :--- | :--- | :--- | :--- |
| `BELONGS_TO_SECTOR` | Company → Sector | GICS sector classification. | `NVDA → sector:semiconductors` |
| `CONSTITUENT_OF` | Company → Index | Index membership. Relevant for rebalancing flows. | `AAPL → index:SPX` |
| `LED_BY` | Company → Person | Executive leadership link. | `NVDA → person:jensen_huang` |
| `COMPETES_WITH` | Company ↔ Company | Direct competitive relationship. Symmetrical. A catalyst for one is relevant to the other. | `AMD ↔ NVDA` |
| `SUPPLIES_TO` | Company → Company | Supply chain dependency. Directional. Disruption propagation path. | `TSMC → NVDA` |
| `PRODUCES` | Company → Product | Product ownership. Links product-level news to ticker. | `NVDA → product:NVDA:H100` |
| `SUBSIDIARY_OF` | Company → Company | Corporate hierarchy. Routes subsidiary news to parent ticker. | `Instagram → META` |

**Tier 2 — Dynamic Relationships** *(Core system value. Updated every run from fresh ingestion. Carry temporal metadata: timestamp, source, significance score, TTL.)*

| Relationship | Direction | Definition | Example |
| :--- | :--- | :--- | :--- |
| `AFFECTED_BY_EVENT` | Company/Sector → Event | Causal link between an occurrence and a holding. Events are nodes with timestamp and type (earnings, regulatory, product_launch, macro_announcement). | `NVDA → event:20260401:nvda_q1_earnings` |
| `DRIVEN_BY_THEME` | Company → Macro Theme | The persistent narrative driving a stock. Distinct from events: themes span multiple runs, events are point-in-time. | `NVDA → theme:ai_capex_cycle` |
| `SENTIMENT_TOWARD` | Source → Company | Measured sentiment from a data source. Carries attributes: `polarity` (bullish/bearish/neutral), `intensity` (high/low), `source_type` (news/reddit/x). Not a judgment — a measurement. | `reddit:wsb → NVDA (bullish, high)` |
| `ANNOUNCED_BY` | Event → Person/Institution | Attribution of a catalyst to its origin. Enables tracing who/what is behind a market-moving event. | `event:20260401:rate_hold → person:jerome_powell` |
| `POLICY_AFFECTS` | Institution/Event → Sector/Company | Regulatory or policy risk link. Captures risk that isn't ticker-specific but hits a class of holdings. | `inst:sec → sector:software` (re: AI disclosure rule) |

**Tier 3 — Cross-Portfolio Relationships** *(Connect holdings to each other. Enable portfolio-level reasoning beyond ticker-by-ticker analysis. Start with `EXPOSED_TO` and `CORRELATED_WITH`; defer `THEMATICALLY_LINKED` to a later iteration.)*

| Relationship | Direction | Definition | Example |
| :--- | :--- | :--- | :--- |
| `EXPOSED_TO` | Company → Macro Theme | Risk exposure, distinct from `DRIVEN_BY_THEME`. One is why you own it (driver); the other is what could hurt it (exposure). | `NVDA → theme:china_export_restrictions` |
| `CORRELATED_WITH` | Company ↔ Company | Quantitative co-movement or shared exposure. **Injected by Agent B** (computed, not LLM-extracted). Symmetrical. Enables concentration risk detection. | `NVDA ↔ AMD (ρ=0.82)` |
| `THEMATICALLY_LINKED` | Theme ↔ Theme | Second-order interaction between macro themes. *Deferred to future iteration* — high noise risk. | `theme:ai_capex_cycle ↔ theme:semiconductor_supply_constraints` |

**Implementation total: 14 relationship types (12 active at launch, 2 deferred).**

#### 2.2.3 LightRAG Integration Details

**Custom Extraction Prompt:** Override LightRAG's default `entity_extract_prompt` with a template that:
1. Lists all valid entity types with their canonical ID format.
2. Lists all valid Tier 1 and Tier 2 relationship types with one-line definitions and one example each.
3. Instructs the LLM to output entities and relationships strictly within these types.
4. Includes a fallback instruction: "If a relationship does not clearly fit any defined type, label it `UNTYPED` for manual review."

**Post-Extraction Validator:** A lightweight Python function that runs between LightRAG's extraction output and graph insertion:
1. Checks each entity type against the closed set. Rejects unknown types.
2. Resolves each entity string through the Canonical Registry. Creates a new registry entry if unknown (with format validation).
3. Checks each relationship type against the closed set. `UNTYPED` edges are logged but not inserted into the active graph. All other non-matching types are discarded.
4. Attaches temporal metadata to Tier 2 edges: `extracted_at`, `source_run_id`, `significance_score`, `ttl`.

**Agent B Graph Injection:** After Agent B computes quantitative metrics, it writes `CORRELATED_WITH` edges directly into the NetworkX graph via LightRAG's graph interface. These edges carry a `source: "agent_b"` attribute to distinguish them from LLM-extracted edges. Updated every run; stale correlation edges from previous runs are replaced, not accumulated.

#### 2.2.4 Three-Tier Persistence Model

Graph nodes and edges are managed according to three persistence tiers, replacing the previous flat TTL approach.

**Tier: Structural (Never Pruned)**
- Tier 1 relationship edges (sector membership, leadership, supply chain, etc.) and their associated entity nodes.
- Updated manually or via registry refresh. These form the permanent graph skeleton.

**Tier: Ephemeral (TTL + Decay)**
- Tier 2 dynamic relationship edges and their associated event/theme nodes from routine ingestion.
- **Default TTL:** 48 hours.
- **Relevance Decay:** Nodes and edges carry a `significance_score` (assigned during extraction, 0.0–1.0). The pruner applies a decay function: effective TTL = `base_ttl × (1 + significance_score)`. A routine social media post (score 0.1) expires in ~53 hours. An earnings surprise (score 0.9) persists for ~91 hours.
- **Agent B edges (`CORRELATED_WITH`) are overwritten each run**, not pruned by TTL.

**Tier: Standing (Persistent Until Resolved)**
- Ongoing macro conditions that are not single events but persisting forces: geopolitical conflicts, monetary policy cycles, prolonged regulatory actions, trade wars, sector-wide crises. These shape every decision for weeks or months and must remain in context beyond the ephemeral window.
- Standing event nodes and their connected edges are **exempt from TTL pruning**. They persist in the graph as long as their `standing_events` row has `status: active`.
- Each standing event carries a **compressed summary** rather than accumulating raw article extracts. The summary is periodically refreshed (see below).

**Standing Event Lifecycle:**

*Creation — Three Mechanisms:*

1. **Auto-Promotion from Ephemeral:** If the same `MACRO_THEME` or `EVENT` entity is re-extracted across ≥6 consecutive runs spanning ≥3 calendar days, the system automatically promotes it to standing. The logic: if it keeps appearing, it's not ephemeral. A new row is inserted into `standing_events` with `promotion_source: auto`.

2. **Manual Pinning via UI:** The user can browse active macro themes in the Standing Events view (§7) and pin any of them as standing. This handles situations where a condition is known to be persistent even during quiet news periods. Sets `promotion_source: manual`.

3. **Agent C Recommendation:** During synthesis, if Agent C's rationale references a macro condition that isn't currently standing, it can include a `promote_to_standing` entry in its output (see §4 Agent C schema). The system then creates the standing event with `promotion_source: agent_c`. This leverages the cloud model's reasoning capacity to recognize regime changes that frequency counting alone would miss.

*Maintenance:*

- **Reinforcement Tracking:** Every run, the post-extraction validator checks whether any ingested content references an active standing event's canonical ID. If so, `last_reinforced` and `reinforcement_count` are updated. This costs nothing — it's a simple ID lookup during existing extraction.
- **Summary Compression:** Every 10 runs (configurable), the system uses Gemma 4 E4B to compress the accumulated recent context about each active standing event into an updated summary (~150-200 words). Format: current status, key developments since last compression, primary market impact channels, affected sectors/tickers. The previous summary plus new ephemeral context referencing this event are the input; the compressed output replaces the old summary. This prevents unbounded context growth.
- **Affected Tickers Update:** The `affected_tickers` array on each standing event is updated whenever extraction produces a new edge linking the standing event's entity to a company node. This ensures Agent A includes the standing event in per-ticker queries for all relevant holdings.

*Resolution:*

- **Staleness Detection:** If a standing event's `last_reinforced` timestamp falls behind by more than `stale_run_threshold` consecutive runs (default 28 runs = ~2 weeks), the system flags it for review. It is **not** auto-resolved — instead, it appears in the UI with a "stale" badge, and Agent C is asked during the next synthesis pass: "The following standing event has not been referenced in recent data. Is it still relevant to current portfolio decisions?" If Agent C says no, the system sets `status: resolved`. If yes, the `last_reinforced` is bumped and the threshold resets.
- **Manual Resolution:** The user can resolve any standing event through the UI at any time.
- **Resolved events** are not deleted from the graph. Their nodes and edges are marked `status: resolved` and excluded from Agent A's active query set, but they remain queryable for historical analysis and future calibration.

---

## 3. Agent Orchestration (The Nervous System)
*Framework: **LangGraph** (Stateful Multi-Agent Workflow with conditional edges).*

**Implementation note:** The orchestration is split into three LangGraph subgraphs composed by a parent graph (`pipeline/orchestration/graph.py`):
- **Data subgraph** (`data_graph.py`): seed_profiles → snapshot → ingest → extract & resolve → embed & prune → standing maintenance (seed_profiles runs first so LightRAG has baseline context before ingestion; it is idempotent — near-no-op after init)
- **Reasoning subgraph** (`reasoning_graph.py`): agent A → agent B → re-query check → agent C (assessment or decision)
- **Execution subgraph** (`execution_graph.py`): standing actions → position sizing → trade execution → logging → notification

The state object (`pipeline/orchestration/state.py`) is a `PipelineState` TypedDict shared across all subgraphs.

The system maintains a **State Object** containing:

| Field | Description |
| :--- | :--- |
| `portfolio_data` | Current snapshot from SQLite `holdings` + `account` + derived weightings. |
| `market_narrative` | Agent A's structured JSON output from LightRAG (see §4 for schema). |
| `quant_assessment` | Agent B's structured JSON (drift, volatility, exposure). `null` on first run. |
| `risk_assessment` | Agent C's final output (action + conviction scores per ticker). |
| `previous_assessment` | Most recent post-close assessment from `recommendations` table (where `action == "assessment"`). Provided to Agent C during pre-open decision runs as additional context. `null` if no post-close run has occurred yet. |
| `source_health` | Per-source ingestion status from the current run. |
| `standing_context` | List of active standing events with summaries and affected tickers, from `standing_events` table. |
| `constraints` | Current constraint values from `constraints` table. |
| `requery_count` | Tracks re-queries to enforce the single-requery limit. |
| `requery_reason` | Structured reason code if a re-query was triggered. |
| `is_first_run` | Boolean. `true` when `computed_targets` table is empty. Skips Agent B and disables re-query triggers. |

### Re-Query Logic (Conditional Edges in LangGraph)

**Disabled on first run** (`is_first_run == true`). On all subsequent runs, **at most one re-query** may fire per run if any of the following conditions are met:

1. **Conflicting Signals:** Agent A's per-ticker sentiment polarity (bullish/bearish) contradicts Agent B's per-ticker health score direction. Measured by deterministic comparison in LangGraph *before* Agent C is called — this is a Python function, not an LLM judgment. If conflicts exist, Agent A re-runs a targeted query for conflicted tickers, then Agent C receives the augmented context.

2. **Missing Data with Expected Significance:** A tracked ticker has zero news/social hits in the current window, AND Agent B flags it for material drift (weight drift > threshold from previous computed target) or volatility change. The absence of information is treated as a signal worth investigating, not ignored.

3. **Magnitude Threshold Breach:** Agent B flags a position that has drifted beyond a configurable threshold (e.g., >5% weight drift from previous run's computed target, or volatility exceeds 2σ of recent history), but Agent A's per-ticker sentiment shows no significant catalyst or explanation.

On re-query, Agent A re-runs with a **targeted query** (specific ticker + expanded search terms), and Agent C receives the augmented context for a second synthesis pass. If the re-query still yields insufficient information, Agent C must decide with what it has — no further loops.

**Implementation Note:** Re-query conditions are evaluated by **deterministic Python logic** in a LangGraph conditional edge node. This node receives Agent A and Agent B outputs, compares structured fields, and routes to either "re-query Agent A" or "proceed to Agent C." Agent C does not decide whether to re-query — it only receives the final assembled context.

---

## 4. The Agent Fleet (The Workers)

### Agent A: The Context Retriever (Local)
- **Model:** `Gemma 4 E4B` (via Ollama). 4.5B effective parameters, Apache 2.0 license. Strong structured output and JSON consistency for ontology-constrained extraction. Runs in ~6GB VRAM at Q4_K_M quantization.
- **Role:** Queries LightRAG with a structured retrieval strategy:
    - **One macro query:** "What are the dominant market narratives in the last 48 hours?" (Global).
    - **One standing context query:** "How do the following ongoing conditions affect the current picture?" with the list of all active standing event summaries injected into the prompt. This ensures persistent macro forces are always represented even when no fresh content references them.
    - **One query per tracked ticker:** "What events, catalysts, or sentiment shifts affect [TICKER]?" (Local). For tickers listed in any active standing event's `affected_tickers`, the standing event summary is appended to the query context.
- **Output Schema (Structured JSON):**
```json
{
  "macro_overview": "200-word narrative summary of dominant themes",
  "standing_context_assessment": {
    "standing:iran_conflict": {
      "still_relevant": true,
      "current_impact": "Elevated oil prices, defense sector tailwind",
      "affected_tickers_update": ["LMT", "XOM", "CVX"]
    }
  },
  "per_ticker": {
    "NVDA": {
      "sentiment": "bullish | bearish | neutral",
      "confidence": "strong | moderate | weak",
      "key_catalysts": ["Q1 earnings beat expectations", "New H200 orders"],
      "narrative": "100-150 word summary of ticker-specific context"
    }
  }
}
```
- Output length scales with portfolio size — one entry per tracked ticker plus macro overview.
- **Re-query mode:** When triggered, runs a single targeted query for the flagged ticker(s) with broadened search terms. Updates only the affected ticker entries in the output.

### Agent B: The Quant Analyst (Local Script)
- **Model:** Deterministic Python code (no LLM).
- **Skipped on first run** (`is_first_run == true`). On the first run, the portfolio is pure cash and there are no previous computed targets to measure drift against. Agent B's output in the state object is set to `null`, and Agent C operates on Agent A's context alone.
- **Role (all subsequent runs):** Calculates portfolio drift from the **most recent `computed_targets`** row, per-ticker and portfolio-level volatility, sector/factor exposure concentrations (checked against `constraints` table), and drawdown metrics.
- **Graph Injection:** After computing metrics, writes `CORRELATED_WITH` edges into LightRAG's NetworkX graph (tagged `source: "agent_b"`). Overwrites previous run's correlation edges. This gives Agent A access to quantitative relationships alongside narrative ones.
- **Output Schema (Structured JSON):**
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
    "NVDA": {
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
    "semiconductors": { "weight": 0.28, "limit": 0.35, "status": "normal" }
  },
  "constraint_violations": []
}
```

### Agent C: The Portfolio Synthesizer (Cloud)
- **Model:** `Gemini 1.5 Pro` or `Claude 3.5 Sonnet`.
- **Role:** The "Lead PM." Operates in two modes depending on run type.
- **Receives:** Agent A's structured output (including standing context assessment), Agent B's quant JSON (or `null` on first run), source health status, current holdings, portfolio constraints. In pre-open decision mode, also receives the **previous post-close assessment** as additional context.

**Assessment Mode (Post-Close Runs):**
- Evaluates end-of-day portfolio state against the full day's data.
- Produces **updated conviction scores only** — no trade actions.
- Output reflects: "Given what happened today, here's how confident I am in each position."
- Stored to `recommendations` with `action: "assessment"`.

**Decision Mode (Pre-Open Runs):**
- Produces concrete trade actions (Buy/Hold/Trim/Exit) with conviction sub-scores.
- On first run: Evaluates Agent A's narratives across the watchlist universe. Recommends initial positions from cash.
- On subsequent runs: Cross-references Agent A's per-ticker sentiment against Agent B's per-ticker health, informed by the overnight post-close assessment.

**Decision Logic:**
1. On first run: Recommend initial positions from the watchlist universe. Uses `narrative_alignment`-only conviction scoring.
2. On subsequent pre-open runs: Cross-reference Agent A + Agent B + post-close assessment. Produce per-ticker recommendation.
3. For each ticker, produce an action and **three categorical conviction sub-scores**.
4. Evaluate standing context for risk weighting — standing events inform `key_risk_factors` and may temper conviction scores, but fresh signals are weighted more heavily for action decisions.
5. Optionally recommend promotion of ephemeral themes to standing status, or resolution of stale standing events.

**Output Schema — Assessment Mode (Post-Close):**
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
      "key_risk_factors": ["China export restrictions", "Valuation stretch"],
      "notable_change": "Upgraded from moderate to strong narrative_alignment after earnings beat"
    }
  },
  "standing_event_actions": { ... }
}
```

**Output Schema — Decision Mode (Pre-Open):**
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
      "key_risk_factors": ["China export restrictions", "Valuation stretch"]
    }
  },
  "standing_event_actions": { ... }
}
```
- `standing_event_actions` is optional — omitted when no promotions or resolutions are warranted.
- `quant_support` and `signal_agreement` are `"n/a"` on the first run (no Agent B data).
- Output is stored to `recommendations` table *before* the Position Sizing Engine runs (pre-open) or as the final step (post-close).

**Cloud LLM Client Implementations:**
- `GeminiClient` — Gemini 1.5 Pro via google.genai SDK. Used for Agent C in production mode.
- `OllamaClient` — Local Ollama endpoint. Used as Agent C fallback (`CLOUD_PROVIDER=ollama`) and as the default seeding LLM when `SEEDING_LLM_PROVIDER=ollama` or when no Gemini key is present.
- `create_cloud_client()` — factory for Agent C client (respects `CLOUD_PROVIDER` env).
- `create_seeding_client()` — factory for profile seeder client (respects `SEEDING_LLM_PROVIDER`: auto/gemini/ollama).

### 4.1 Position Sizing Engine (Deterministic)

A pure Python module that sits between Agent C's output and trade execution. **No LLM involvement.** Takes Agent C's per-ticker conviction sub-scores and translates them into concrete share counts.

**Step 1 — Conviction Weight Mapping:**

Each combination of sub-scores maps to a numeric conviction weight via a lookup table:

| narrative_alignment | quant_support | signal_agreement | conviction_weight |
| :--- | :--- | :--- | :--- |
| strong | strong | strong | 1.0 |
| strong | moderate | strong | 0.85 |
| strong | strong | moderate | 0.80 |
| moderate | strong | strong | 0.75 |
| strong | weak | strong | 0.65 |
| strong | moderate | moderate | 0.60 |
| moderate | moderate | moderate | 0.50 |
| moderate | weak | moderate | 0.40 |
| weak | moderate | weak | 0.30 |
| weak | weak | weak | 0.15 |

*This table is not exhaustive — all 27 combinations are defined in code. Values are initial estimates and are a primary target for the future calibration framework.*

**First-run mapping:** When `quant_support` and `signal_agreement` are `"n/a"`, conviction weight is determined by `narrative_alignment` alone: strong → 0.70, moderate → 0.45, weak → 0.20.

**Step 2 — Normalization to Target Weights:**

1. Filter to actionable tickers: only those with action = Buy or Hold. (Trim/Exit positions are being reduced, not allocated to.)
2. Compute raw target weight per ticker: `conviction_weight_i / sum(all conviction_weights)`.
3. Scale by `(1.0 - cash_floor)` to reserve the cash floor.

**Step 3 — Constraint Enforcement (in order):**

1. **Cap single positions:** If any ticker's target weight exceeds `max_single_position`, cap it and redistribute the excess proportionally across remaining tickers.
2. **Cap sector concentrations:** Sum target weights per sector. If any sector exceeds `max_sector_concentration`, proportionally scale down all tickers in that sector and redistribute.
3. **Floor dust positions:** If any ticker's target weight falls below `min_position_size`, set it to zero (not worth trading). Redistribute its weight proportionally.
4. **Re-normalize** after all caps/floors to ensure weights sum to `(1.0 - cash_floor)`.

**Step 4 — Compute Trade List:**

1. Diff target weights against current holdings to produce per-ticker delta (in dollar terms).
2. Convert dollar deltas to share counts at the expected fill price (see §6 for pricing rules).
3. For Trim/Exit actions: compute shares to sell to reach target weight (or zero for Exit).
4. Verify total cash after all trades remains ≥ `cash_floor × total_portfolio_value`.

**Step 5 — Store & Execute:**

1. Write target allocation to `computed_targets` table (append-only, keyed to `run_id`).
2. Pass trade list to the simulated execution step.

---

## 5. Deployment & Hardware Mapping
*Optimized for NVIDIA 5060 Laptop GPU (8GB VRAM).*

- **GPU (VRAM) Allocation:**
    - `nomic-embed-text` (Permanent): ~0.5GB.
    - `Gemma 4 E4B Q4_K_M` (Active during ingestion + Agent A runs): ~6GB.
    - Remaining ~1.5GB reserved for system overhead.
    - **Upgrade Path:** A 16GB GPU (e.g., RTX 5060 Ti or used RTX 3090) would unlock Gemma 4 26B-A4B MoE (~15GB at Q4), providing substantially better extraction and narrative synthesis quality.
- **System RAM Allocation:**
    - LightRAG Graph (NetworkX): ~2-4GB.
    - Python Orchestration (LangGraph): ~1GB.
    - SQLite + Web App: negligible.
- **Alerting:** Webhook to private Discord/Telegram. Sent after each run with a Markdown summary of recommendations and any triggered re-queries.

---

## 6. Execution Flow

### Run Type Distinction

The two daily runs serve fundamentally different roles:

| Run Type | Purpose | Agent C Mode | Trades Produced? | Portfolio Revalued? |
| :--- | :--- | :--- | :--- | :--- |
| **Post-Close** (~4:30 PM ET) | Evaluate end-of-day state. Refresh graph with full-day data. Produce updated conviction assessments. | **Assessment mode** — outputs updated conviction scores and rationale per ticker. No trade actions. | **No.** | **Yes** — all positions revalued at closing price. |
| **Pre-Open** (~8:30 AM ET) | Decide on trades for the day. Incorporates overnight news and the previous post-close assessment. | **Decision mode** — outputs trade actions (Buy/Hold/Trim/Exit) with conviction scores. | **Yes** — trades execute at official opening auction price (`Open` from yfinance). | **No.** |

- Post-close is the system's reflection step. It ingests the full day's data, revalues the portfolio, and produces a confidence snapshot — "here's how I feel about every position after today's action." No trades are queued or executed.
- Pre-open is the system's action step. It incorporates overnight developments, references the previous post-close assessment as additional context, and produces concrete trade decisions that execute at that day's open.
- **Gap tracking:** Each trade logs `gap_pct` — the percentage difference between the previous close and the fill price. Stored for future analysis of whether large gaps correlate with poor recommendation quality.
- **Slippage:** A configurable `slippage_applied` field (default 0.0) is stored on every trade. Placeholder for future realism.
- **Market Calendar:** Before executing, the scheduler checks whether today is a trading day (via `exchange_calendars` package). On weekends and holidays, the run is skipped entirely (or runs ingestion-only to keep the graph fresh, logging `run_type: non_trading_day`).

### Standard Run (Both Run Types)
1. **Trigger:** APScheduler fires at configured time. Market calendar check — skip if not a trading day. *(Scheduler and market calendar not yet implemented — runs are triggered manually or via API.)*
2. **Snapshot:** Record current portfolio state to `snapshots` table. On post-close runs, revalue all positions at closing price.
3. **Ingest:** Scrapers gather data from all sources for the watchlist universe; per-source health is logged. Crawl4AI cleans raw content.
4. **Extract & Resolve:** LightRAG's custom extraction prompt (constrained to closed ontology) runs on cleaned chunks via Gemma 4 E4B. Raw entities are resolved through the Canonical Registry. Post-extraction validator enforces type compliance and attaches temporal metadata.
5. **Embed:** LightRAG updates the local graph with new entities/edges. Pruner runs decay logic on ephemeral nodes (standing event nodes are exempt).
6. **Standing Event Maintenance:** Auto-promotion check (has any ephemeral theme appeared in ≥6 runs over ≥3 days?). Reinforcement tracking (update `last_reinforced` for any active standing event referenced in this run's extraction). Summary compression (if run count since last compression ≥ 10 for any standing event, re-compress its summary via Gemma 4 E4B). Staleness check (flag any standing event exceeding `stale_run_threshold` consecutive unreferenced runs).
7. **Research (Agent A):** Gemma 4 E4B queries LightRAG with structured retrieval strategy, including standing context injection. Produces structured JSON output. Stored to `agent_outputs`.
8. **First-Run Check:** If `is_first_run == true`, skip to step 11 (Agent B is skipped, re-query is disabled).
9. **Quantify (Agent B):** Deterministic script computes portfolio health metrics against most recent `computed_targets`. Writes `CORRELATED_WITH` edges to graph. Output stored to `agent_outputs`.
10. **Re-Query Check:** Deterministic Python node compares Agent A and Agent B structured outputs. If re-query conditions are met and `requery_count == 0`: Agent A re-runs targeted query → updates affected ticker entries. `requery_count` increments.

### Post-Close Run (Assessment Only — continues from step 10)
11. **Assess (Agent C — Assessment Mode):** Cloud model receives Agent A's output + Agent B's output + source health + current holdings + constraints. Produces updated **conviction scores only** — no trade actions. Output schema uses `assessment` action type (see §4 Agent C schema).
12. **Standing Event Actions:** Process any `standing_event_actions` from Agent C's output.
13. **Log:** Agent C's assessment stored to `agent_outputs` and `recommendations` table (with `action: "assessment"` for each ticker). No Position Sizing Engine run. No trades.
14. **Notify:** Discord/Telegram alert with conviction snapshot, any standing event changes, and notable shifts from previous assessment. *(Notification webhooks not yet implemented.)*

### Pre-Open Run (Trade Producing — continues from step 10)
11. **Synthesize (Agent C — Decision Mode):** Cloud model receives Agent A's output + Agent B's output (or `null` on first run) + **previous post-close assessment** as additional context + source health + current holdings + constraints. Produces per-ticker trade actions (Buy/Hold/Trim/Exit) with conviction sub-scores. May recommend standing event promotions or resolutions.
12. **Standing Event Actions:** Process any `standing_event_actions` from Agent C's output.
13. **Log:** Agent C's raw output stored to `agent_outputs`. Parsed recommendations stored to `recommendations` table.
14. **Size (Position Sizing Engine):** Conviction sub-scores → conviction weights → normalized target weights → constraint enforcement → trade list. Target allocation written to `computed_targets`.
15. **Execute (Simulated):** Trades execute at official opening auction price. Logged to `trades` table with `gap_pct` and `slippage_applied`. `holdings` and `account.cash_balance` updated. Cost basis updated via weighted average method (see below).
16. **Notify:** Discord/Telegram alert with recommendations, conviction levels, trades executed, any re-query details, and any standing event changes. *(Notification webhooks not yet implemented.)*

### Cost Basis Accounting (Weighted Average Method)
- **On Buy (new position):** `cost_basis_per_share = fill_price`.
- **On Buy (adding to existing):** `cost_basis_per_share = (old_shares × old_basis + new_shares × fill_price) / total_shares`.
- **On Trim (partial sell):** Cost basis per share is unchanged. Only `shares` decreases.
- **On Exit (full sell):** Row removed from `holdings`. Realized P&L = `(fill_price - cost_basis) × shares`, logged on the `trades` row.

### First Run Flow
1. User initializes portfolio via UI: selects ticker universe (S&P 500 / Nasdaq 100 / Dow 30), enters starting cash amount, optionally adjusts default constraints.
2. System pulls the constituent ticker list from Wikipedia, populates `watchlist` table with tickers + sectors. Creates `account` row with cash balance and universe selection, empty `holdings`, default `constraints` rows.
3. First scheduled pre-open run fires (or user triggers manually).
4. Steps 1-7 execute normally (ingestion for full watchlist, extraction, Agent A).
5. Agent B is **skipped entirely** — no previous targets, no holdings to measure drift on, no correlations to compute.
6. Re-query logic is **disabled** — no Agent B output to conflict with Agent A.
7. Agent C (decision mode) receives Agent A's output + `null` quant assessment. Recommends initial positions from the watchlist universe based on tickers it finds compelling. Uses `narrative_alignment`-only conviction scoring.
8. Position Sizing Engine runs with first-run conviction mapping. Produces initial target allocation.
9. Trades execute at that day's opening price to move from 100% cash toward the target allocation.

---

## 7. Web Application Layer

**Mode:** Simulation only. No live broker integration. UI is designed as if real to support future transition.

**Implementation status:** FastAPI backend (`backend/`) and React frontend (`frontend/`) are committed (phases 8 and 9 complete). Backend routers cover init, portfolio, recommendations, runs, standing events, constraints, watchlist, profiles (`profiles.py` — GET /api/profiles, GET /api/profiles/{key}), and admin (`admin.py` — POST /api/admin/reset, requires `{"confirm": true}`). Frontend has pages for all core views (Dashboard, Recommendations, Runs, Run Detail, P&L, Standing Events, Constraints, Init) with shared components and API client modules.

### Initialization Flow
- **First-time setup screen:** User selects a ticker universe (S&P 500 / Nasdaq 100 / Dow Jones 30), enters starting cash amount (required), and optionally adjusts constraint defaults (`cash_floor`, `max_single_position`, `max_sector_concentration`, `min_position_size`).
- On submit: System scrapes the selected index's Wikipedia constituent table to populate the `watchlist`. `account` and `constraints` tables are populated. User is redirected to the main dashboard (which shows 100% cash, the selected universe, no holdings, awaiting first run).
- User can trigger the first pre-open run manually or wait for the next scheduled time.

### Core Views
- **Dashboard:** Current portfolio value, allocation breakdown (derived weights, not stored), daily P&L, cash balance, constraint status (any current violations highlighted), and the latest post-close conviction snapshot (per-ticker conviction scores with notable changes highlighted).
- **Recommendation History:** Chronological log of all Agent C decisions with conviction sub-scores, the conviction weights computed by the Position Sizing Engine, linked to the trades they triggered and subsequent actual returns.
- **Run Inspector:** Per-run detail view showing Agent A structured output (including standing context assessment), Agent B structured output (or "skipped" on first run), source health, re-query details, standing event changes, and the full trade list produced by the Position Sizing Engine.
- **Simulated P&L:** Portfolio performance over time, charted from `snapshots` table. Includes benchmark comparison (e.g., SPY buy-and-hold from same starting cash).
- **Standing Events:** View and manage persistent macro conditions. Shows all active standing events with summary, category, affected tickers, promotion source, last reinforced date, and reinforcement count. User can: **pin** any ephemeral macro theme as standing (manual promotion), **resolve** any active standing event, **edit** affected tickers or summary text, and **review** stale-flagged events. Resolved events are visible in a collapsed archive section for historical reference.
- **Constraints Editor:** View and modify portfolio constraints. Changes take effect on the next run.

### Future Additions (Designed For, Not Implemented)
- **Manual Override / Veto:** Ability to reject a recommendation before simulated execution. Supported by the `recommendations` → `trades` separation.
- **Calibration Dashboard:** Compare recommendation accuracy against actual outcomes. Analyze conviction weight mapping effectiveness. Supported by joining `recommendations` against future `snapshots` and the `computed_targets` history.
- **Per-asset-class scheduling:** Extend scheduler to support crypto/forex with independent cadences. Supported by per-asset-class scheduler interface.
- **Live Broker Integration:** Replace simulated execution with real order routing. Supported by the `trades` table abstraction and `slippage_applied` field.
