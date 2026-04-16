"""EconRAG FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import (
    admin,
    constraints,
    init,
    portfolio,
    profiles,
    recommendations,
    runs,
    standing_events,
    watchlist,
)

app = FastAPI(title="EconRAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(init.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(runs.router, prefix="/api")
app.include_router(standing_events.router, prefix="/api")
app.include_router(constraints.router, prefix="/api")
app.include_router(watchlist.router, prefix="/api")
app.include_router(profiles.router, prefix="/api")
app.include_router(admin.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=5000, reload=True)
