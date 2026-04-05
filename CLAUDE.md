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
- **Scheduling:** APScheduler with `exchange_calendars` for market day checks
- **Environment:** `python-dotenv` for API keys and config

## Project Structure (Planned)

```
backend/          # FastAPI app, routers, dependencies
pipeline/
  db/             # SQLite schema, init, query helpers
  scrapers/       # Watchlist scraper (Wikipedia)
  ingestion/      # Market data, news, social, Crawl4AI parser, health tracking
  knowledge/      # LightRAG config, extraction prompt, canonical registry, validator, pruner
  agents/         # Agent A (local LLM), Agent B (quant script), Agent C (cloud API)
  sizing/         # Conviction mapping, normalization, constraint enforcement, trade builder
  orchestration/  # LangGraph graph, conditions, executor, scheduler, notifications
frontend/
  src/pages/      # Dashboard, recommendations, run inspector, P&L, standing events, constraints
  src/components/ # Charts, tables, cards, conviction badges
  src/api/        # API client
tests/            # One test file per module
```

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

## Development Guidelines

- **Start with Dow 30** for development and testing (30 tickers keeps iteration fast). Switch to larger universes after pipeline is stable.
- **Git strategy:** One feature branch per phase. Merge to main when tests pass. Tag each phase completion.
- **Environment variables needed:** API keys (NewsData.io, Reddit, Gemini/Claude), Ollama endpoint URL, Discord/Telegram webhook URL, database path. Use `.env` file.
- **Weights are never stored** in the holdings table — always derived at query time as `(shares * current_price) / total_portfolio_value`.
- **Cost basis:** Weighted average method. Unchanged on trim, realized P&L on exit.
- **Constraint enforcement order matters:** single position cap -> sector cap -> dust floor -> re-normalize. Excess is always redistributed proportionally.
