# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous Portfolio Monitoring Agent — a multi-agent system for US equities sentiment and fundamental monitoring operating in simulation-only mode on a virtual portfolio. Twice-daily pipeline (pre-open 8:30 AM ET, post-close 4:30 PM ET) ingests market data, news, and social content, builds a knowledge graph, runs three agents (local LLM, deterministic quant, cloud LLM), and produces trade recommendations executed against a simulated portfolio.

## Tech Stack

- **Backend/Pipeline:** Python, FastAPI
- **Frontend:** React SPA
- **Database:** SQLite (WAL mode)
- **Local LLM:** Gemma 4 E4B via Ollama
- **Cloud LLM:** Gemini 1.5 Pro or Claude 3.5 Sonnet (swappable)
- **Orchestration:** LangGraph (stateful multi-agent workflow with conditional edges)
- **Knowledge Graph:** LightRAG (NanoVectorDB + NetworkX) with custom ontology-constrained extraction prompt
- **Web Scraping:** Crawl4AI, yfinance, PRAW, NewsData.io/GNews
- **Scheduling:** APScheduler with `exchange_calendars` for market day checks (not yet implemented)
- **Environment:** `python-dotenv` for API keys and config

## Project Structure

```
backend/                    # FastAPI app (Phase 8)
  main.py                   # FastAPI app setup with CORS
  deps.py                   # Database session dependency
  schemas.py                # Pydantic response models
  routers/
    init.py                 # POST /api/init, GET /api/init/status
    portfolio.py            # GET /api/portfolio, snapshots, benchmark
    recommendations.py      # GET /api/recommendations
    runs.py                 # GET /api/runs, POST /api/runs/trigger
    standing_events.py      # CRUD for standing events
    constraints.py          # GET/PATCH /api/constraints
    watchlist.py            # GET /api/watchlist, POST /api/watchlist/refresh
    profiles.py             # GET /api/profiles, GET /api/profiles/{key}
    admin.py                # POST /api/admin/reset
pipeline/
  config.py                 # Shared config (paths, API keys, model settings)
  db/
    connection.py           # SQLite connection with WAL mode
    schema.py               # All 13 table definitions (includes kg_seed_log)
    init.py                 # Initialization flow (account, watchlist, constraints, canonical entities)
    reset.py                # hard_reset() — deletes DB + WAL sidecars + LightRAG storage
    helpers.py              # Query helpers (portfolio value, derived weights, first-run check)
  scrapers/
    watchlist.py            # Wikipedia scraper for index constituents (SP500/Nasdaq100/Dow30)
  ingestion/
    models.py               # Data models for ingestion layer
    health.py               # Source health tracking
    market_data.py          # yfinance scraper
    news.py                 # NewsData.io / GNews scraper
    social.py               # PRAW Reddit scraper
    parser.py               # Crawl4AI URL-to-Markdown parser
    orchestrator.py         # Runs all scrapers, volume caps, URL dedup
  knowledge/
    lightrag_config.py      # LightRAG setup with Ollama config
    extraction_prompt.py    # Custom ontology-constrained extraction prompt
    canonical.py            # Canonical entity registry resolution
    validator.py            # Post-extraction validator with tier classification
    graph_ops.py            # Graph insertion, query, correlation edge injection
    extraction.py           # Extraction orchestrator with Ollama LLM calls
    pruner.py               # Ephemeral TTL pruner with tier immunity
    profile_grounding.py    # yfinance fact packs (TickerFactPack, MacroFactPack) for seeder
    profile_seeder.py       # Cloud LLM profile generation → extract → validate → insert into LightRAG
  agents/
    agent_a.py              # Local LLM context retriever (Gemma 4 E4B via Ollama)
    agent_b.py              # Deterministic quant script (drift, volatility, health scores)
    agent_c.py              # Cloud LLM synthesis (Gemini/Claude, assessment + decision modes)
    cloud_client.py         # Abstracted API client for Gemini/Claude
  sizing/
    conviction_map.py       # 27-combination conviction weight lookup table
    normalizer.py           # Normalization and constraint enforcement
    trade_builder.py        # Trade list computation and cost basis
    engine.py               # Orchestrates the full sizing pipeline
  orchestration/
    state.py                # PipelineState TypedDict
    graph.py                # Parent LangGraph graph and pipeline entry point
    data_graph.py           # Data pipeline LangGraph subgraph (seed_profiles → ingest → extract → embed → standing)
    reasoning_graph.py      # Agent reasoning LangGraph subgraph
    execution_graph.py      # Execution LangGraph subgraph
    conditions.py           # Re-query evaluator, run-type branching
    executor.py             # Simulated trade execution and cost basis
    snapshots.py            # Portfolio snapshot recording
    standing.py             # Standing event action processing
frontend/                   # React SPA (Phase 9)
  src/
    App.jsx                 # Routing setup
    main.jsx                # Entry point
    pages/                  # InitPage, DashboardPage, RecommendationsPage, RunsPage,
                            # RunDetailPage, PnlPage, StandingEventsPage, ConstraintsPage
    components/             # charts/, conviction/, layout/, shared/
    hooks/                  # Custom React hooks
    api/                    # API client modules (one per domain)
tests/                      # One test file per module
docs/
  specs/                    # Design specs used for each build phase
```

## Build Phase Status

| Phase | Name | Status | Branch |
| :--- | :--- | :--- | :--- |
| 1 | SQLite Schema & Initialization | Complete | phase3-data-ingestion |
| 2 | Position Sizing Engine | Complete | phase3-data-ingestion |
| 3 | Data Ingestion (yfinance + Crawl4AI) | Complete | phase3-data-ingestion |
| 4 | LightRAG, Extraction & Canonical Registry | Complete | phase3-data-ingestion |
| 5 | Agent A & Agent B | Complete | phase3-data-ingestion |
| 6 | Agent C (Cloud LLM) | Complete | phase3-data-ingestion |
| 7 | LangGraph Orchestration | Complete | phase3-data-ingestion |
| 8 | FastAPI Backend | Complete | phase3-data-ingestion |
| 9 | React Frontend | Complete | phase3-data-ingestion |
| 10 | Profile Seeding & Admin Reset | In progress | Improve-initialization |

Design specs for each phase are in `docs/specs/`.

## Architecture Reference

Read `ARCHITECTURE.md` for the full system design (ontology, schemas, agent output formats, persistence tiers). Read `PLAN.md` for the phased build order — phases are bottom-up, each producing testable code before the next begins.

### Key Architectural Concepts

- **Three agents:** Agent A (local Gemma 4 E4B via LightRAG queries), Agent B (deterministic Python quant metrics), Agent C (cloud LLM synthesis producing trade actions or assessments)
- **Two run types:** Post-close (assessment only, no trades) and pre-open (decision mode, produces trades)
- **First-run special case:** Agent B is skipped, re-query disabled, Agent C uses narrative_alignment-only conviction scoring
- **Position Sizing Engine:** Pure Python, no LLM — maps 27 conviction score combinations to weights, enforces constraints (single position cap, sector cap, dust floor, cash floor) in strict order
- **Re-query logic:** Deterministic Python (not LLM) comparing Agent A sentiment vs Agent B health scores — max 1 re-query per run
- **Three-tier graph persistence:** Structural (never pruned), Ephemeral (TTL + significance decay), Standing (persistent until resolved)
- **Canonical Entity Registry:** Maps raw extracted strings to canonical IDs (e.g., "Nvidia Corp" -> `NVDA`). 8 entity types, 12 active relationship types in a closed ontology
- **Standing events:** Persistent macro conditions with auto-promotion (6+ runs over 3+ days), manual pinning, or Agent C recommendation. Staleness detection at 28 consecutive unreferenced runs
- **Profile Seeding:** At initialization and on subsequent runs, a cloud LLM (or Ollama fallback) generates per-ticker and macro markdown profiles grounded in live yfinance fact packs. These are extracted into LightRAG so Agent A has non-empty context on the first pipeline run. Tracked in `kg_seed_log`; idempotent.

## Development Guidelines

- **Start with Dow 30** for development and testing (30 tickers keeps iteration fast). Switch to larger universes after pipeline is stable.
- **Git strategy:** One feature branch per phase. Merge to main when tests pass. Tag each phase completion.
- **Environment variables needed:** API keys (NewsData.io, Reddit, Gemini/Claude), Ollama endpoint URL, Discord/Telegram webhook URL, database path. Use `.env` file.
- **Weights are never stored** in the holdings table — always derived at query time as `(shares * current_price) / total_portfolio_value`.
- **Cost basis:** Weighted average method. Unchanged on trim, realized P&L on exit.
- **Constraint enforcement order matters:** single position cap -> sector cap -> dust floor -> re-normalize. Excess is always redistributed proportionally.
