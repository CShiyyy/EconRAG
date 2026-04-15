"""Portfolio endpoints."""

import json
import sqlite3
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_db
from backend.schemas import BenchmarkResponse, PortfolioResponse
from pipeline.db.helpers import (
    get_account,
    get_derived_weights,
    get_holdings,
    get_portfolio_value,
)

router = APIRouter(tags=["portfolio"])

# In-memory benchmark cache: {data, fetched_at}
_benchmark_cache: dict[str, Any] = {}
_BENCHMARK_TTL = 900  # 15 minutes


def _snapshot_price_fn(conn: sqlite3.Connection):
    """Build a price lookup from the latest snapshot's per_ticker_json."""
    row = conn.execute(
        "SELECT per_ticker_json FROM snapshots ORDER BY snapshot_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    per_ticker = json.loads(row["per_ticker_json"])
    prices: dict[str, float] = {}
    for ticker, data in per_ticker.items():
        if isinstance(data, dict) and "price" in data:
            prices[ticker] = data["price"]
    if not prices:
        return None
    return lambda t: prices.get(t, 0.0)


@router.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio(conn: sqlite3.Connection = Depends(get_db)):
    try:
        account = get_account(conn)
    except (RuntimeError, sqlite3.OperationalError):
        raise HTTPException(status_code=404, detail="System not initialized")

    holdings = get_holdings(conn)
    price_fn = _snapshot_price_fn(conn) if holdings else None

    total_value = get_portfolio_value(conn, price_fn)
    weights = get_derived_weights(conn, price_fn)

    return PortfolioResponse(
        total_value=total_value,
        cash=account["cash_balance"],
        holdings=holdings,
        weights=weights,
    )


@router.get("/portfolio/snapshots")
def get_snapshots(conn: sqlite3.Connection = Depends(get_db)):
    rows = conn.execute(
        "SELECT * FROM snapshots ORDER BY snapshot_id ASC"
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["per_ticker_json"] = json.loads(d["per_ticker_json"])
        result.append(d)
    return {"snapshots": result}


@router.get("/portfolio/benchmark", response_model=BenchmarkResponse)
def get_benchmark(conn: sqlite3.Connection = Depends(get_db)):
    try:
        account = get_account(conn)
    except (RuntimeError, sqlite3.OperationalError):
        raise HTTPException(status_code=404, detail="System not initialized")

    initial_cash = account["initial_cash"]
    initialized_at = account["initialized_at"]

    # Portfolio series from snapshots
    rows = conn.execute(
        "SELECT timestamp, total_value FROM snapshots ORDER BY snapshot_id ASC"
    ).fetchall()
    portfolio_series = [
        {"timestamp": r["timestamp"], "value": r["total_value"]} for r in rows
    ]

    # SPY benchmark — cached
    now = time.time()
    if (
        _benchmark_cache.get("data") is not None
        and now - _benchmark_cache.get("fetched_at", 0) < _BENCHMARK_TTL
    ):
        benchmark_series = _benchmark_cache["data"]
    else:
        benchmark_series = _fetch_spy_benchmark(initial_cash, initialized_at)
        _benchmark_cache["data"] = benchmark_series
        _benchmark_cache["fetched_at"] = now

    return BenchmarkResponse(
        portfolio_series=portfolio_series,
        benchmark_series=benchmark_series,
        initial_cash=initial_cash,
    )


def _fetch_spy_benchmark(initial_cash: float, initialized_at: str) -> list[dict]:
    """Fetch SPY history and scale to initial_cash."""
    try:
        import yfinance as yf

        spy = yf.Ticker("SPY")
        hist = spy.history(start=initialized_at[:10])
        if hist.empty:
            return []
        start_price = hist["Close"].iloc[0]
        return [
            {
                "timestamp": idx.isoformat(),
                "value": round(initial_cash * (row["Close"] / start_price), 2),
            }
            for idx, row in hist.iterrows()
        ]
    except Exception:
        return []
