# Phase 7 — LangGraph Orchestration Design

**Date:** 2026-04-06
**Scope:** Core state machine only. APScheduler and Discord/Telegram notifications deferred.

---

## Overview

Wire all existing pipeline components (ingestion, knowledge graph, agents A/B/C, sizing engine) into an end-to-end LangGraph state machine with conditional edges for re-query logic, first-run handling, and run-type branching. Checkpointing via `langgraph-checkpoint-sqlite` for run durability.

## Architecture: Three Subgraphs + Parent Composer

### Subgraph 1 — Data Pipeline (`data_graph`)

Nodes: `snapshot` → `ingest` → `extract_and_resolve` → `embed_and_prune` → `standing_maintenance`

Responsibility: Get data in, update knowledge graph, maintain standing events. No agent reasoning.

| Node | Calls | Updates State |
|------|-------|---------------|
| `snapshot` | `snapshots.record_snapshot()` | (side effect: writes to `snapshots` table) |
| `ingest` | `run_ingestion(tickers)` | `ingestion_result`, `source_health` |
| `extract_and_resolve` | `extraction.process_content_batch()` → `canonical.resolve_entities()` → `validator.validate_extraction()` → `graph_ops.insert_validated_data()` | (side effect: updates graph) |
| `embed_and_prune` | `pruner.prune_expired_edges()` | (side effect: removes expired graph edges) |
| `standing_maintenance` | `standing.run_maintenance()` — reinforcement tracking + staleness detection (auto-promotion and summary compression deferred) | `standing_context` |

### Subgraph 2 — Agent Reasoning (`reasoning_graph`)

Nodes: `agent_a` → `agent_b` → `requery_check` → `agent_c`

Conditional edges:
- After `agent_a`: if `is_first_run` → skip `agent_b` and `requery_check`, go to `agent_c` (decision mode)
- After `requery_check`: if conflict detected AND `requery_count == 0` → loop back to `agent_a` (re-query mode), increment `requery_count`
- After `requery_check` (no re-query): route to `agent_c` based on `run_type`
- `agent_c` node internally selects assessment vs decision mode based on `run_type`

| Node | Calls | Updates State |
|------|-------|---------------|
| `agent_a` | `run_agent_a(conn, rag, tickers, run_id, flagged_tickers?)` | `market_narrative` |
| `agent_b` | `run_agent_b(conn, rag, market_data, run_id)` | `quant_assessment` (None on first run) |
| `requery_check` | `conditions.evaluate_requery()` | `requery_count`, `requery_reason` |
| `agent_c` | `run_agent_c(conn, agent_a_out, agent_b_out, source_health, run_id, run_type, is_first_run)` | `risk_assessment` |

### Subgraph 3 — Execution (`execution_graph`)

Nodes: `standing_actions` → `position_sizing` → `execute_trades` → `log`

- On post-close runs: only `standing_actions` → `log` (no sizing or trades)
- On pre-open runs: full chain

| Node | Calls | Updates State |
|------|-------|---------------|
| `standing_actions` | `standing.process_actions(agent_c_output)` | (side effect: creates/resolves standing events) |
| `position_sizing` | `run_sizing_engine(conn, run_id, agent_c_output, fill_prices, is_first_run)` | `sizing_result` |
| `execute_trades` | `executor.execute_trades(conn, sizing_result, market_data, run_id)` | `trade_list` |
| `log` | Update `run_log` row with `wall_clock_seconds`, `source_status`, `requery_triggered` | (side effect: updates run_log) |

### Parent Graph (`graph.py`)

Composes: `data_graph` → `reasoning_graph` → conditional → `execution_graph` or `log_only`

The parent graph creates the `run_log` entry at the start (to get `run_id`), initializes `PipelineState`, and runs the three subgraphs in sequence.

## State Schema

```python
class PipelineState(TypedDict):
    # Run metadata
    run_id: int
    run_type: str                        # "pre_open" | "post_close"
    is_first_run: bool
    start_time: float                    # time.time() for wall_clock_seconds

    # Data pipeline outputs
    ingestion_result: dict | None        # IngestionResult serialized to dict for checkpointing
    source_health: list[dict]

    # Agent outputs
    market_narrative: dict | None        # Agent A output
    quant_assessment: dict | None        # Agent B output (None on first run)
    risk_assessment: dict | None         # Agent C output
    previous_assessment: dict | None     # From DB: latest post-close assessment
    standing_context: list[dict]         # Active standing events

    # Re-query tracking
    requery_count: int                   # 0 or 1
    requery_reason: str | None
    flagged_tickers: list[str]           # Tickers flagged for re-query

    # Execution outputs
    sizing_result: dict | None
    trade_list: list[dict]

    # Infrastructure (not checkpointed)
    db_path: str                         # Path to SQLite DB
    rag_storage_dir: str                 # Path to LightRAG storage
```

Note: `conn` and `rag` are not stored in state (not serializable). Each node opens its own connection/LightRAG instance using `db_path` and `rag_storage_dir`. This is required for checkpoint serialization.

## Re-Query Condition Evaluator (`conditions.py`)

Deterministic Python, no LLM. Compares Agent A per-ticker sentiment against Agent B per-ticker health scores.

```python
def evaluate_requery(
    agent_a_output: dict,
    agent_b_output: dict | None,
    source_health: list[dict],
) -> dict:
    """Returns {"should_requery": bool, "flagged_tickers": [...], "reason": str}"""
```

Three trigger conditions (any one fires a re-query):

1. **Conflicting Signals:** Agent A sentiment polarity contradicts Agent B health direction for a ticker. E.g., Agent A says `bullish` but Agent B flags `warning`/`breach`, or Agent A says `bearish` but Agent B shows `normal` with positive drift.

2. **Missing Data + Significance:** A ticker has zero news/social hits in source_health AND Agent B flags it for material drift (> `DRIFT_BREACH_THRESHOLD`) or volatility spike.

3. **Magnitude Threshold Breach:** Agent B drift > 0.05 or volatility > 2x historical, but Agent A shows `neutral` sentiment with `weak` confidence (no catalyst explanation).

Returns `should_requery: False` if `agent_b_output is None` (first run).

## Simulated Trade Executor (`executor.py`)

```python
def execute_trades(
    conn: sqlite3.Connection,
    sizing_result: dict,
    market_data: dict,
    run_id: int,
) -> list[dict]:
    """Execute trades, update holdings/cash/cost-basis, return trade records."""
```

For each trade in `sizing_result["trade_list"]`:
1. Compute `gap_pct`: `(open_price - previous_close) / previous_close` from market data
2. Call `update_cost_basis()` from `pipeline.sizing.trade_builder`
3. Update `holdings` table: insert (Buy new), update shares (Buy add / Trim), delete (Exit)
4. Update `account.cash_balance`
5. Insert into `trades` table with `gap_pct`, `slippage_applied=0.0`, `realized_pnl`
6. Final verification: assert cash >= `cash_floor * total_portfolio_value`

## Snapshot Recorder (`snapshots.py`)

```python
def record_snapshot(
    conn: sqlite3.Connection,
    run_id: int,
    market_data: dict | None,
    run_type: str,
) -> None:
```

- Post-close: revalue all positions at `current_price` from market data
- Pre-open: snapshot current state before trades
- Writes to `snapshots` table: `run_id`, `timestamp`, `total_value`, `cash`, `per_ticker_json`

## Standing Event Processor (`standing.py`)

Two functions:

```python
def run_maintenance(conn: sqlite3.Connection, rag: Any) -> list[dict]:
    """Auto-promotion check, reinforcement tracking, staleness detection.
    Returns list of active standing events for state."""

def process_actions(
    conn: sqlite3.Connection,
    rag: Any,
    agent_c_output: dict,
) -> None:
    """Process promote_to_standing and recommend_resolution from Agent C."""
```

- `promote_to_standing`: Insert `standing_events` row with `promotion_source: "agent_c"`, create/update `canonical_entities`, mark graph nodes as standing-tier
- `recommend_resolution`: Set `status: "resolved"`, `resolved_at: now`
- Auto-promotion: Check if any ephemeral `MACRO_THEME`/`EVENT` entity appeared in >= 6 runs over >= 3 calendar days
- Staleness: Flag standing events with > 28 consecutive unreferenced runs

## Checkpointing

- Use `langgraph-checkpoint-sqlite` backed by the same SQLite database
- Checkpoint table is managed by langgraph (separate from our schema)
- Enables interrupted runs to resume from the last completed node
- Non-serializable objects (`sqlite3.Connection`, LightRAG instance) are reconstructed per-node from `db_path` and `rag_storage_dir` in state

## Entry Point

```python
async def run_pipeline(
    db_path: str,
    run_type: str,  # "pre_open" | "post_close"
    rag_storage_dir: str | None = None,
) -> dict:
    """Main entry point. Creates run_log, builds state, runs the graph."""
```

## File Structure

```
pipeline/orchestration/
    __init__.py           # exports run_pipeline
    state.py              # PipelineState TypedDict
    data_graph.py         # Subgraph 1: snapshot → ingest → extract → prune → standing
    reasoning_graph.py    # Subgraph 2: agent_a → agent_b → requery → agent_c
    execution_graph.py    # Subgraph 3: standing_actions → sizing → trades → log
    graph.py              # Parent graph composing subgraphs
    conditions.py         # Re-query evaluator
    executor.py           # Simulated trade execution
    snapshots.py          # Portfolio snapshot recording
    standing.py           # Standing event maintenance + action processing
```

## Dependencies to Add

```
langgraph>=0.2
langgraph-checkpoint-sqlite>=2.0
exchange-calendars>=4.3      # for future scheduler phase
```

## Existing Modules Referenced

| Module | Key Functions |
|--------|--------------|
| `pipeline.db.connection` | `get_connection(db_path)` |
| `pipeline.db.helpers` | `is_first_run()`, `get_account()`, `get_constraints()`, `get_watchlist()`, `get_holdings()`, `get_active_standing_events()`, `get_previous_assessment()`, `store_agent_output()`, `store_recommendations()`, `get_portfolio_value()`, `get_derived_weights()` |
| `pipeline.ingestion.orchestrator` | `run_ingestion(tickers)` |
| `pipeline.knowledge.extraction` | `process_content_batch()`, `extract_from_markdown()` |
| `pipeline.knowledge.canonical` | `resolve_entities()`, `resolve_entity()` |
| `pipeline.knowledge.validator` | `validate_extraction()` |
| `pipeline.knowledge.graph_ops` | `insert_validated_data()`, `query_graph()`, `inject_correlation_edges()` |
| `pipeline.knowledge.pruner` | `prune_expired_edges()` |
| `pipeline.knowledge.lightrag_config` | LightRAG initialization |
| `pipeline.agents.agent_a` | `run_agent_a()` |
| `pipeline.agents.agent_b` | `run_agent_b()` |
| `pipeline.agents.agent_c` | `run_agent_c()` |
| `pipeline.sizing.engine` | `run_sizing_engine()` |
| `pipeline.sizing.trade_builder` | `update_cost_basis()` |

## Test Plan

### `tests/test_orchestration.py` — End-to-end pipeline tests

1. **Post-close full run:** Mock all agents and ingestion. Verify: snapshot recorded, ingestion ran, agents A+B ran, Agent C ran in assessment mode, no trades, recommendations logged with `action: "assessment"`.
2. **Pre-open full run:** Mock agents. Verify: Agent C in decision mode, sizing ran, trades executed, holdings updated, cost basis correct.
3. **First-run pipeline:** Verify: Agent B skipped (`quant_assessment` is None), re-query disabled, Agent C in decision mode with first-run mapping, trades from cash to initial allocation.
4. **State flow:** Verify each subgraph receives and passes state correctly across boundaries.

### `tests/test_requery.py` — Re-query trigger conditions

1. **Conflicting signals:** Mock Agent A bullish + Agent B warning for same ticker. Verify re-query fires.
2. **Missing data + significance:** Mock zero news hits + Agent B drift breach. Verify re-query fires.
3. **Magnitude breach:** Mock Agent B volatility spike + Agent A neutral/weak. Verify re-query fires.
4. **Re-query limit:** After one re-query, mock another conflict. Verify no second re-query.
5. **First run:** Verify re-query never fires when `agent_b_output is None`.

### `tests/test_executor.py` — Trade execution

1. **Buy new position:** Verify holdings inserted, cash reduced, cost basis = fill price.
2. **Buy add to existing:** Verify weighted average cost basis.
3. **Trim:** Verify shares reduced, cost basis unchanged, realized P&L computed.
4. **Exit:** Verify holdings row removed, full realized P&L logged.
5. **Cash floor enforcement:** Verify executor refuses trades that would violate cash floor.
6. **Gap percentage:** Verify `gap_pct` computed correctly from previous_close vs open_price.

## Deferred to Follow-Up

- APScheduler integration with `exchange_calendars` market day checks
- Discord/Telegram webhook notifications
- Standing event summary compression (requires Gemma 4 E4B call)
- Auto-promotion frequency counting (requires tracking extraction history)
