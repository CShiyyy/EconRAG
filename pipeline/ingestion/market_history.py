"""OHLCV history cache — fetches multi-year price data via yfinance and stores in SQLite."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10


def refresh_ohlcv_cache(
    conn: sqlite3.Connection,
    tickers: list[str],
    lookback_days: int,
) -> None:
    """Fetch missing OHLCV tail from yfinance and upsert into market_history_cache.

    For each ticker, finds the latest cached date and only fetches from that
    point forward, minimizing API calls on subsequent runs.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    start_str = cutoff.strftime("%Y-%m-%d")

    for i in range(0, len(tickers), _BATCH_SIZE):
        batch = tickers[i : i + _BATCH_SIZE]
        latest_dates: dict[str, str | None] = {}
        for t in batch:
            row = conn.execute(
                "SELECT MAX(date) FROM market_history_cache WHERE ticker = ?", (t,)
            ).fetchone()
            latest_dates[t] = row[0] if row and row[0] else None

        # Determine per-ticker fetch start
        for ticker in batch:
            ld = latest_dates[ticker]
            fetch_start = ld if ld and ld > start_str else start_str
            try:
                df = yf.download(
                    ticker, start=fetch_start, progress=False, auto_adjust=True
                )
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.get_level_values(1) else df
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                _upsert_ohlcv(conn, ticker, df)
            except Exception as e:
                logger.warning(f"OHLCV fetch failed for {ticker}: {e}")


def _upsert_ohlcv(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    rows = [
        (ticker, str(idx.date()), float(row["Open"]), float(row["High"]),
         float(row["Low"]), float(row["Close"]), float(row["Volume"]))
        for idx, row in df.iterrows()
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO market_history_cache
           (ticker, date, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
