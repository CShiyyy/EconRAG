# Development Plan: Autonomous Portfolio Monitoring Agent

**Reference:** `architecture.md` (547 lines, finalized)
**Approach:** Bottom-up. Each phase produces testable, working code before the next begins. No phase depends on anything not yet built.

---

## Phase 1 — SQLite Schema & Initialization Flow

**Goal:** The database exists, migrations run, and the initialization flow creates a ready-to-run system state from user inputs.

**Tasks:**
1. Define the project structure: `backend/`, `frontend/`, `pipeline/`, `tests/`. Shared config module for paths, constants, constraint defaults.
2. Implement schema creation script (all 12 tables): `account`, `watchlist`, `constraints`, `holdings`, `computed_targets`, `snapshots`, `recommendations`, `trades`, `run_log`, `agent_outputs`, `canonical_entities`, `standing_events`.
3. Set `PRAGMA journal_mode=WAL` on database initialization.
4. Build the watchlist scraper: given a universe key (`sp500`, `nasdaq100`, `djia30`), scrape the Wikipedia constituent table and return a list of `(ticker, company_name, sector)` tuples. Handle table format variations across the three pages.
5. Build the initialization function: accepts `universe`, `starting_cash`, and optional constraint overrides → creates `account` row, populates `watchlist`, inserts default `constraints` rows (with any overrides applied).
6. Pre-seed `canonical_entities` from the watchlist: for each ticker, create a `COMPANY` entity with the ticker as `canonical_id` and common aliases (company name, common abbreviations) in the `aliases` JSON array. Pre-seed sector entities from GICS sectors found in the watchlist.
7. Build helper functions: `get_portfolio_value()` (cash + sum of holdings × current price), `get_derived_weights()` (per-ticker weight calculation), `get_previous_computed_target()` (most recent `computed_targets` row), `is_first_run()` (check if `computed_targets` is empty).

**Deliverables:**
- `pipeline/db/schema.py` — table definitions and creation.
- `pipeline/db/init.py` — initialization flow.
- `pipeline/db/helpers.py` — query helpers.
- `pipeline/scrapers/watchlist.py` — Wikipedia scraper for index constituents.
- `tests/test_schema.py` — schema creation, FK integrity, WAL mode verification.
- `tests/test_init.py` — initialization with each universe, constraint overrides, watchlist population, canonical entity seeding.

**Test Criteria:**
- Initialize with each of the three universes. Verify watchlist row counts match expected (roughly 500, 100, 30).
- Verify constraints table has 4 rows with correct defaults.
- Verify canonical_entities pre-seeded with tickers and sectors.
- Verify `is_first_run()` returns `True` on fresh database.
- Verify WAL mode is active.

---

## Phase 2 — Position Sizing Engine

**Goal:** A pure Python module that takes mock Agent C output and produces a concrete trade list, fully respecting constraints.

**Tasks:**
1. Implement the conviction weight mapping lookup table (all 27 combinations of strong/moderate/weak × 3 sub-scores). Implement the first-run mapping (narrative_alignment only → weight).
2. Implement normalization: filter to Buy/Hold tickers → compute raw weights → scale by `(1 - cash_floor)`.
3. Implement constraint enforcement in order: single position cap → sector concentration cap → dust position floor → re-normalization. Each step redistributes excess proportionally.
4. Implement trade list computation: diff target weights against current holdings → convert dollar deltas to share counts at a given fill price → verify cash floor is maintained after all trades.
5. Implement cost basis accounting: weighted average on buy, unchanged on trim, realized P&L on exit.
6. Implement `computed_targets` storage: write the target allocation to the database keyed to a `run_id`.

**Deliverables:**
- `pipeline/sizing/conviction_map.py` — lookup table and mapping logic.
- `pipeline/sizing/normalizer.py` — normalization and constraint enforcement.
- `pipeline/sizing/trade_builder.py` — trade list computation and cost basis.
- `pipeline/sizing/engine.py` — orchestrates the full 5-step pipeline.
- `tests/test_sizing.py` — comprehensive test suite (see below).

**Test Criteria:**
- **Conviction mapping:** Verify all 27 combinations produce expected weights. Verify first-run mapping with `n/a` sub-scores.
- **Normalization:** Given 5 tickers with known conviction weights, verify output weights sum to `(1 - cash_floor)`.
- **Single position cap:** Input a ticker with conviction weight that would exceed `max_single_position`. Verify it's capped and excess redistributed proportionally.
- **Sector cap:** Input 4 tickers in the same sector whose combined weight exceeds `max_sector_concentration`. Verify sector is capped and excess redistributed.
- **Dust floor:** Input a ticker whose weight falls below `min_position_size`. Verify it's zeroed out and weight redistributed.
- **Trade list:** Given current holdings and target weights, verify correct share deltas, correct dollar amounts, correct fill price application.
- **Cost basis:** Verify weighted average on additional buy. Verify unchanged on trim. Verify realized P&L calculation on exit.
- **Cash floor:** Verify the engine refuses to produce a trade list that would violate the cash floor.
- **Edge case — first run from 100% cash:** Verify the engine produces buy trades for all recommended tickers from a zero-holdings starting state.
- **Edge case — all Exit:** Verify the engine correctly liquidates all positions and returns to cash.

---

## Phase 3 — Data Ingestion (yfinance + Crawl4AI)

**Goal:** The system can fetch market data for the watchlist and parse news URLs into clean Markdown. Source health is tracked.

**Tasks:**
1. Implement yfinance scraper: given a list of tickers, fetch current price, open price, previous close, fundamental ratios (P/E, market cap), and 30-day price history (for Agent B volatility calculations later). Return structured data keyed by ticker.
2. Implement news scraper (NewsData.io or GNews): given a list of tickers, fetch headlines and URLs within a 48-hour window. Return structured list of `(ticker, headline, url, source, published_at)`.
3. Implement social scraper (PRAW for Reddit): given a list of cashtags, fetch top posts from relevant subreddits (r/wallstreetbets, r/stocks, r/investing). Return structured list of `(ticker, title, body, url, score, subreddit, posted_at)`.
4. Implement Crawl4AI integration: given a list of URLs from news/social hits, convert each to clean Markdown. Handle timeouts, failed fetches, and rate limits gracefully.
5. Implement source health logging: each scraper returns a status object `{source: str, status: "success" | "timeout" | "rate_limited" | "error", items_fetched: int, duration_ms: int}`. These are aggregated into the `source_status` JSON for the run log.
6. Implement content volume cap: max 20 articles and 30 social posts per ticker per run. Prioritize by source quality (major outlets first) and recency.
7. Implement URL-level deduplication: if the same URL appears across multiple tickers or sources, only process it once through Crawl4AI.

**Deliverables:**
- `pipeline/ingestion/market_data.py` — yfinance scraper.
- `pipeline/ingestion/news.py` — news API scraper.
- `pipeline/ingestion/social.py` — Reddit scraper.
- `pipeline/ingestion/parser.py` — Crawl4AI integration.
- `pipeline/ingestion/health.py` — source health tracking.
- `pipeline/ingestion/orchestrator.py` — runs all scrapers, applies volume caps and dedup, returns unified ingestion result.
- `tests/test_ingestion.py` — per-source tests with mocked API responses, volume cap verification, dedup verification, health status generation.

**Test Criteria:**
- yfinance returns structured data for a known ticker. Handles invalid tickers gracefully.
- News scraper returns results within the 48-hour window. Handles API key errors and rate limits.
- Social scraper returns Reddit posts. Handles subreddit unavailability.
- Crawl4AI converts a known URL to Markdown. Handles timeouts and 404s without crashing.
- Volume cap: inject 50 articles for one ticker, verify only 20 are passed to Crawl4AI.
- Dedup: inject the same URL under two tickers, verify Crawl4AI processes it once.
- Health status: verify each scraper reports correct status on success and failure.

---

## Phase 4 — LightRAG, Extraction Prompt & Canonical Registry

**Goal:** LightRAG is running locally with the custom ontology-constrained extraction prompt. Entities are resolved through the canonical registry. The graph is queryable.

**Tasks:**
1. Install and configure LightRAG with NanoVectorDB and NetworkX as backends. Verify persistence (graph survives process restart).
2. Write the custom extraction prompt template: enumerate all 8 entity types with canonical ID formats, all 12 active relationship types with definitions and examples. Include the `UNTYPED` fallback instruction.
3. Override LightRAG's default `entity_extract_prompt` with the custom template. Verify extraction runs through Gemma 4 E4B via Ollama.
4. Implement the canonical registry resolution layer: given raw extracted entity strings, look up in `canonical_entities` table. If found, return canonical ID. If not found, validate the LLM-proposed ID format and insert a new entry.
5. Implement the post-extraction validator: check entity types against closed set, check relationship types against closed set, log `UNTYPED` edges, attach temporal metadata (`extracted_at`, `source_run_id`, `significance_score`, `ttl`) to Tier 2 edges.
6. Implement the graph insertion pipeline: validated, canonicalized entities and edges → LightRAG graph.
7. Implement the ephemeral pruner: scan graph for Tier 2 edges past their effective TTL (`base_ttl × (1 + significance_score)`). Remove expired nodes/edges. Leave Tier 1 and standing event nodes untouched.
8. Test the full pipeline end-to-end: Crawl4AI Markdown → extraction → validation → canonicalization → graph insertion → query.

**Deliverables:**
- `pipeline/knowledge/lightrag_config.py` — LightRAG setup with custom prompt.
- `pipeline/knowledge/extraction_prompt.py` — the custom prompt template.
- `pipeline/knowledge/canonical.py` — canonical registry resolution.
- `pipeline/knowledge/validator.py` — post-extraction validation.
- `pipeline/knowledge/pruner.py` — ephemeral TTL pruner.
- `pipeline/knowledge/graph_ops.py` — graph insertion and query helpers.
- `tests/test_extraction.py` — extraction with sample Markdown, validation, canonicalization.
- `tests/test_pruner.py` — TTL expiry, significance-based decay, structural edge immunity.

**Test Criteria:**
- Feed a sample news article about NVDA earnings through the pipeline. Verify correct entity types and relationship types are extracted.
- Verify canonical resolution: "Nvidia Corp" and "NVIDIA" both resolve to `NVDA`.
- Verify invalid entity types are rejected. Verify `UNTYPED` relationships are logged but not inserted.
- Verify temporal metadata is attached to Tier 2 edges.
- Verify pruner removes expired ephemeral nodes but leaves structural nodes intact.
- Verify graph survives process restart (LightRAG persistence).

---

## Phase 5 — Agent A & Agent B

**Goal:** Both agents produce their specified structured JSON outputs. Agent A queries LightRAG and returns per-ticker sentiment. Agent B computes quant metrics against computed targets.

**Tasks:**

### Agent A (Local LLM)
1. Implement the structured retrieval strategy: one macro query, one standing context query (injecting active standing event summaries), one query per tracked ticker (with standing event context appended for affected tickers).
2. Implement the output parser: Gemma 4 E4B's raw response → validated JSON matching Agent A's output schema (`macro_overview`, `standing_context_assessment`, `per_ticker` with sentiment/confidence/catalysts/narrative).
3. Implement re-query mode: given a list of flagged tickers, re-run targeted queries with broadened search terms, update only affected entries.
4. Handle output validation: if Gemma 4 E4B produces malformed JSON or missing fields, retry once with a corrective prompt. If still malformed, produce a degraded output (neutral sentiment, weak confidence) and log the failure.
5. Store raw output to `agent_outputs` table.

### Agent B (Deterministic Script)
1. Implement drift calculation: compare current holdings weights against most recent `computed_targets` row.
2. Implement volatility calculation: 30-day rolling volatility per ticker and portfolio-level, using price history from yfinance.
3. Implement sector concentration check: sum weights per sector, compare against `max_sector_concentration` constraint.
4. Implement drawdown calculation: max drawdown over 30-day window from `snapshots` table.
5. Implement health score flags: per-ticker status (normal/warning/breach) based on configurable thresholds for drift, volatility.
6. Implement constraint violation detection: check all constraints, return list of any current violations.
7. Implement `CORRELATED_WITH` graph injection: compute pairwise correlation between held tickers using 30-day price data, write edges to LightRAG graph with `source: "agent_b"` attribute. Overwrite previous run's edges.
8. Store structured JSON output to `agent_outputs` table.

**Deliverables:**
- `pipeline/agents/agent_a.py` — retrieval strategy, LLM query, output parsing, re-query mode.
- `pipeline/agents/agent_b.py` — all quant calculations, graph injection, output formatting.
- `tests/test_agent_a.py` — with mocked LightRAG responses and Ollama responses.
- `tests/test_agent_b.py` — with known portfolio states and price histories, verify drift/volatility/concentration calculations.

**Test Criteria:**
- **Agent A:** Given a pre-populated LightRAG graph, verify output JSON has correct structure. Verify per-ticker sentiment is one of bullish/bearish/neutral. Verify standing context assessment is included when standing events exist.
- **Agent A re-query:** Trigger re-query for a specific ticker, verify only that ticker's entry is updated.
- **Agent B:** Given holdings at known weights and a computed target, verify drift calculations are correct. Given 30 days of price data, verify volatility matches expected value. Given sector weights, verify concentration check flags breaches correctly.
- **Agent B skip:** Verify Agent B returns `null` when `is_first_run()` is true.
- **Agent B graph injection:** Verify `CORRELATED_WITH` edges appear in the graph with correct correlation values and `source: "agent_b"` attribute.

---

## Phase 6 — Agent C (Cloud LLM Integration)

**Goal:** Agent C calls the cloud API in both assessment and decision modes, produces validated structured JSON, and handles API errors.

**Tasks:**
1. Implement the cloud API client: support both Gemini 1.5 Pro and Claude 3.5 Sonnet. Abstract behind a common interface so the model is swappable via config.
2. Build the prompt constructor for assessment mode (post-close): assemble Agent A output + Agent B output + source health + current holdings + constraints + standing context → system prompt + user message. Instruct the model to produce assessment-mode JSON (conviction scores, no trade actions, notable changes).
3. Build the prompt constructor for decision mode (pre-open): same inputs plus the previous post-close assessment → instruct the model to produce decision-mode JSON (trade actions + conviction scores + standing event actions).
4. Build the prompt constructor for first-run mode: Agent A output + `null` Agent B + source health + cash-only portfolio + constraints + watchlist → instruct the model to recommend initial positions with narrative_alignment-only scoring.
5. Implement output parser and validator: parse JSON response, verify all required fields, verify action values are valid, verify conviction sub-scores are valid categorical values. Handle malformed responses with one retry.
6. Implement `standing_event_actions` extraction: parse any `promote_to_standing` or `recommend_resolution` entries.
7. Store raw LLM response to `agent_outputs` (agent = "C"). Parse and store per-ticker entries to `recommendations` table.
8. Implement API error handling: rate limits (retry with backoff), timeouts (retry once, then degrade), authentication errors (log and halt run).

**Deliverables:**
- `pipeline/agents/agent_c.py` — prompt construction, API call, response parsing, both modes.
- `pipeline/agents/cloud_client.py` — abstracted API client for Gemini/Claude.
- `tests/test_agent_c.py` — with mocked API responses for both modes, validation, error handling.

**Test Criteria:**
- Assessment mode: given mocked inputs, verify output JSON has `action: "assessment"` for all tickers, valid conviction scores, rationale text.
- Decision mode: verify output JSON has valid trade actions, valid conviction scores, rationale and risk factors.
- First-run mode: verify output recommends positions from the watchlist universe with `quant_support: "n/a"` and `signal_agreement: "n/a"`.
- Standing event actions: verify `promote_to_standing` entries are correctly parsed. Verify `recommend_resolution` entries reference valid standing IDs.
- API error: verify retry on rate limit. Verify graceful degradation on timeout.

---

## Phase 7 — LangGraph Orchestration

**Goal:** The full pipeline runs end-to-end as a LangGraph state machine with conditional edges for re-query logic and run-type branching.

**Tasks:**
1. Define the LangGraph state schema matching the architecture's State Object (all 11 fields).
2. Implement the pipeline as a LangGraph graph with nodes: `snapshot`, `ingest`, `extract_and_resolve`, `embed_and_prune`, `standing_maintenance`, `agent_a`, `agent_b`, `requery_check`, `agent_c_assess`, `agent_c_decide`, `standing_actions`, `log`, `position_sizing`, `execute_trades`, `notify`.
3. Implement conditional edges:
   - After `snapshot`: branch on run type → if post-close, set revalue flag.
   - After `agent_a`: branch on `is_first_run` → if true, skip to `agent_c_decide`.
   - After `requery_check`: branch on conflict detection → if re-query needed, loop back to `agent_a` (re-query mode).
   - After `requery_check` / `agent_a` (if no re-query): branch on run type → `agent_c_assess` (post-close) or `agent_c_decide` (pre-open).
   - After `agent_c_decide`: proceed to `position_sizing` → `execute_trades`.
   - After `agent_c_assess`: proceed to `log` → `notify` (no sizing or execution).
4. Implement the re-query condition evaluator: deterministic Python comparing Agent A's per-ticker sentiment against Agent B's per-ticker health scores. Check the three trigger conditions (conflicting signals, missing data with significance, magnitude threshold breach).
5. Implement the standing event action processor: create new standing events from promotions, mark resolutions.
6. Implement the simulated trade executor: apply trades at the correct fill price (open price), update `holdings`, update `account.cash_balance`, apply cost basis accounting, log to `trades` table.
7. Implement the snapshot recorder: capture full portfolio state to `snapshots` table with per-ticker JSON.
8. Implement the APScheduler integration: two daily triggers (8:30 AM ET, 4:30 PM ET), market calendar check via `exchange_calendars`, `max_instances=1`, `coalesce=True`.
9. Implement the notification sender: webhook to Discord/Telegram with a Markdown-formatted run report.

**Deliverables:**
- `pipeline/orchestration/graph.py` — LangGraph state machine definition.
- `pipeline/orchestration/conditions.py` — re-query evaluator, run-type branching.
- `pipeline/orchestration/executor.py` — simulated trade execution and cost basis.
- `pipeline/orchestration/snapshots.py` — portfolio snapshot recording.
- `pipeline/orchestration/standing.py` — standing event action processing.
- `pipeline/orchestration/scheduler.py` — APScheduler setup with market calendar.
- `pipeline/orchestration/notify.py` — Discord/Telegram webhook.
- `tests/test_orchestration.py` — full pipeline with mocked agents, verify state transitions.
- `tests/test_requery.py` — verify re-query triggers fire correctly, verify re-query limit of 1.
- `tests/test_executor.py` — trade execution, cost basis, cash floor enforcement.

**Test Criteria:**
- Run a full post-close pipeline with mocked agents. Verify: snapshot recorded, ingestion ran, agents A and B ran, Agent C ran in assessment mode, no trades produced, recommendations logged with `action: "assessment"`.
- Run a full pre-open pipeline with mocked agents. Verify: agents ran, Agent C ran in decision mode, Position Sizing Engine ran, trades executed, holdings updated, cost basis correct.
- Run a first-run pipeline. Verify: Agent B skipped, re-query disabled, Agent C in decision mode with first-run mapping, trades move from cash to initial allocation.
- Trigger a re-query: mock conflicting signals between Agent A and B. Verify Agent A re-runs, re-query count increments, Agent C receives augmented context.
- Verify re-query limit: after one re-query, mock another conflict. Verify no second re-query fires.
- Market calendar: mock a Saturday trigger. Verify the run is skipped or logged as `non_trading_day`.
- Scheduler: verify `max_instances=1` prevents overlapping runs.

---

## Phase 8 — FastAPI Backend

**Goal:** The web app has a REST API that exposes all data and operations needed by the frontend.

**Tasks:**
1. Set up FastAPI application structure with routers for each domain.
2. **Initialization endpoints:**
   - `POST /api/init` — accepts `{universe, starting_cash, constraints?}`, runs initialization flow, returns account state.
   - `GET /api/init/status` — returns whether the system is initialized.
3. **Portfolio endpoints:**
   - `GET /api/portfolio` — current holdings with derived weights, total value, cash, P&L.
   - `GET /api/portfolio/snapshots` — time series for the P&L chart.
   - `GET /api/portfolio/benchmark` — SPY buy-and-hold comparison data.
4. **Recommendations endpoints:**
   - `GET /api/recommendations` — paginated chronological log with conviction scores and linked trade outcomes.
   - `GET /api/recommendations/{run_id}` — single run's recommendations.
5. **Run endpoints:**
   - `GET /api/runs` — paginated run log.
   - `GET /api/runs/{run_id}` — full run detail (agent outputs, source health, re-query info, trade list).
   - `POST /api/runs/trigger` — manually trigger a run (accepts `run_type: pre_open | post_close`).
6. **Standing events endpoints:**
   - `GET /api/standing-events` — all active + resolved events.
   - `POST /api/standing-events` — manually pin a new standing event.
   - `PATCH /api/standing-events/{id}` — edit summary, affected tickers, or resolve.
7. **Constraints endpoints:**
   - `GET /api/constraints` — current constraint values.
   - `PATCH /api/constraints` — update constraint values.
8. **Watchlist endpoints:**
   - `GET /api/watchlist` — current watchlist with sector info.
   - `POST /api/watchlist/refresh` — re-scrape Wikipedia and update.
9. Implement CORS middleware for React frontend communication.

**Deliverables:**
- `backend/main.py` — FastAPI app setup.
- `backend/routers/` — one router per domain (portfolio, recommendations, runs, standing_events, constraints, watchlist, init).
- `backend/deps.py` — database session dependency.
- `tests/test_api.py` — endpoint tests with test database.

**Test Criteria:**
- Every endpoint returns correct status codes and response shapes.
- `POST /api/init` creates all expected database state.
- `POST /api/runs/trigger` starts a pipeline run and returns the run ID.
- `PATCH /api/constraints` updates values and they take effect on subsequent queries.
- Standing event CRUD works correctly (create, read, update/resolve).

---

## Phase 9 — React Frontend

**Goal:** A functional dashboard that visualizes portfolio state, recommendations, run history, and standing events.

**Tasks:**
1. Set up React project with routing, Tailwind CSS (or preferred styling), and an HTTP client (axios or fetch wrapper) pointed at the FastAPI backend.
2. **Initialization page:** universe selector (dropdown: S&P 500 / Nasdaq 100 / Dow 30), starting cash input, constraint editor with defaults pre-filled. Submit calls `POST /api/init`.
3. **Dashboard page:** portfolio value card, allocation pie/bar chart (derived weights), daily P&L, cash balance, constraint status badges, latest post-close conviction snapshot table (per-ticker conviction scores with notable changes highlighted).
4. **Recommendation History page:** sortable/filterable table of all recommendations. Each row: timestamp, ticker, action, conviction sub-scores, conviction weight, rationale (expandable), linked trade outcome. Filter by ticker, action, date range.
5. **Run Inspector page:** list of all runs. Click into a run to see: Agent A output (collapsible JSON), Agent B output (collapsible JSON), source health cards, re-query details, standing event changes, trade list. Post-close runs show assessment, pre-open runs show decisions + trades.
6. **Simulated P&L page:** line chart of portfolio value over time from snapshots. Overlay SPY benchmark. Selectable date range.
7. **Standing Events page:** table of active standing events with summary, category, affected tickers, promotion source, last reinforced, reinforcement count, stale badge. Buttons: pin new event, edit, resolve. Collapsed section for resolved events.
8. **Constraints Editor page:** form with current constraint values. Edit and save calls `PATCH /api/constraints`. Show effective-next-run notice.
9. **Manual run trigger:** button in the dashboard header to trigger a run (with run type selector).

**Deliverables:**
- `frontend/src/pages/` — one page component per view.
- `frontend/src/components/` — reusable components (charts, tables, cards, conviction score badges).
- `frontend/src/api/` — API client module.
- `frontend/src/App.jsx` — routing setup.

**Test Criteria:**
- Initialization page creates a working system and redirects to dashboard.
- Dashboard displays correct values matching API responses.
- Recommendation history displays all entries with correct data.
- Run inspector displays correct agent outputs and trade lists.
- P&L chart renders from snapshot data.
- Standing events CRUD works through the UI.
- Manual run trigger fires and the dashboard updates after completion.

---

## Development Notes

**Start with Dow 30 for development and testing.** 30 tickers keeps ingestion fast, LLM inference quick, and iteration cycles short. Switch to Nasdaq 100 or S&P 500 only after the full pipeline is stable.

**Run overlap prevention:** Configure APScheduler with `max_instances=1` and `coalesce=True` from the start. Log skipped runs.

**LightRAG persistence:** Verify early in Phase 4 that the graph survives process restart. If LightRAG doesn't persist by default, add an explicit save step at the end of each run and a load step at startup.

**Environment variables:** API keys (NewsData.io, Reddit, Gemini/Claude), Ollama endpoint URL, Discord/Telegram webhook URL, database path. Use a `.env` file with `python-dotenv`.

**Git strategy:** One feature branch per phase. Merge to main only when tests pass. Tag each phase completion.
