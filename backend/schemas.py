"""Pydantic request/response models for the EconRAG API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Init ---

class InitRequest(BaseModel):
    universe: Literal["sp500", "nasdaq100", "djia30"]
    starting_cash: float = Field(gt=0)
    constraints: dict[str, float] | None = None


class InitStatusResponse(BaseModel):
    initialized: bool
    account: dict[str, Any] | None = None


# --- Portfolio ---

class PortfolioResponse(BaseModel):
    total_value: float
    cash: float
    holdings: list[dict[str, Any]]
    weights: dict[str, float]


class BenchmarkResponse(BaseModel):
    portfolio_series: list[dict[str, Any]]
    benchmark_series: list[dict[str, Any]]
    initial_cash: float


# --- Pagination ---

class PaginatedResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    skip: int
    limit: int


# --- Runs ---

class TriggerRunRequest(BaseModel):
    run_type: Literal["pre_open", "post_close"]


class TriggerRunResponse(BaseModel):
    trigger_id: int
    status: str


class TriggerStatusResponse(BaseModel):
    trigger_id: int
    status: str
    run_id: int | None = None
    error: str | None = None


# --- Standing Events ---

STANDING_CATEGORIES = Literal[
    "geopolitical",
    "monetary_policy",
    "regulatory",
    "trade_policy",
    "sector_crisis",
    "other",
]


class StandingEventCreate(BaseModel):
    canonical_id: str
    category: STANDING_CATEGORIES
    summary: str
    affected_tickers: list[str]
    stale_run_threshold: int = 28


class StandingEventPatch(BaseModel):
    summary: str | None = None
    affected_tickers: list[str] | None = None
    status: Literal["active", "resolved"] | None = None


# --- Constraints ---

class ConstraintPatch(BaseModel):
    constraints: dict[str, float]
