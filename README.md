# EconRAG

Autonomous Portfolio Monitoring Agent for US equities. A multi-agent system that ingests market data, news, and social content, builds a knowledge graph, runs three reasoning agents, and produces trade recommendations executed against a simulated portfolio.

The pipeline runs twice daily (pre-open 8:30 AM ET, post-close 4:30 PM ET) with two run types: **post-close** (assessment only) and **pre-open** (decision mode, produces trades).

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com/) running locally with **Gemma 4 E4B** pulled
- API keys: NewsData.io, Reddit (PRAW), Gemini or Claude

## Setup

```bash
# Clone and install Python dependencies
git clone https://github.com/CShiyyy/EconRAG.git
cd EconRAG
pip install -e ".[dev]"

# Create .env with your API keys
cp .env.example .env  # then edit with your keys

# Install frontend dependencies
cd frontend
npm install
cd ..
```

## Running

```bash
# Start the backend (port 8000)
python -m backend.main

# Start the frontend dev server (port 5173)
cd frontend
npm run dev
```

The frontend is available at `http://localhost:5173` and proxies API requests to the backend.

## Tests

```bash
pytest                    # run all tests
pytest -m "not network"   # skip tests requiring internet
```

## Directory Structure

```
pipeline/                   # Core pipeline logic
  config.py                 # Paths, API keys, model settings
  db/                       # SQLite schema, init, connection, helpers
  scrapers/                 # Index constituent scraper (Wikipedia)
  ingestion/                # Market data (yfinance), news, social, URL parser
  knowledge/                # LightRAG setup, extraction, canonical registry, pruning
  agents/                   # Agent A (local LLM), Agent B (quant), Agent C (cloud LLM)
  sizing/                   # Position sizing engine (conviction map, normalizer, trade builder)
  orchestration/            # LangGraph workflow (state, subgraphs, conditions, executor)

backend/                    # FastAPI REST API
  main.py                   # App setup with CORS
  deps.py                   # Database session dependency
  schemas.py                # Pydantic response models
  routers/                  # Endpoint modules (portfolio, runs, recommendations, etc.)

frontend/                   # React SPA (Vite + Tailwind)
  src/
    pages/                  # Dashboard, Recommendations, Runs, P&L, Standing Events, etc.
    components/             # Charts, conviction badges, layout shell, shared UI
    hooks/                  # useInitCheck, usePolling
    api/                    # API client modules (one per domain)

tests/                      # Pytest test suite
docs/specs/                 # Design specs for each build phase
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design including ontology, schemas, agent output formats, and persistence tiers.
