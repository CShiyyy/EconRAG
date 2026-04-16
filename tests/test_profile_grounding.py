"""Tests for pipeline/knowledge/profile_grounding.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yf_ticker(info: dict | None = None, news: list | None = None):
    """Return a mock yfinance Ticker with configurable .info and .news."""
    t = MagicMock()
    t.info = info if info is not None else {}
    t.news = news if news is not None else []
    return t


def _make_info(
    *,
    ceo: str | None = "Tim Cook",
    industry: str = "Consumer Electronics",
    summary: str = "Apple designs smartphones.",
    employees: int = 164_000,
    market_cap: float = 3_190_000_000_000,
    pe: float = 31.2,
    high_52: float = 237.49,
    low_52: float = 164.08,
    price: float = 197.30,
) -> dict:
    officers = []
    if ceo:
        officers = [{"name": ceo, "title": "Chief Executive Officer"}]
    return {
        "industry": industry,
        "longBusinessSummary": summary,
        "fullTimeEmployees": employees,
        "marketCap": market_cap,
        "trailingPE": pe,
        "fiftyTwoWeekHigh": high_52,
        "fiftyTwoWeekLow": low_52,
        "currentPrice": price,
        "companyOfficers": officers,
    }


# ---------------------------------------------------------------------------
# _extract_ceo tests
# ---------------------------------------------------------------------------

class TestExtractCeo:
    def test_extracts_from_chief_executive_title(self):
        from pipeline.knowledge.profile_grounding import _extract_ceo
        officers = [{"name": "Tim Cook", "title": "Chief Executive Officer"}]
        assert _extract_ceo(officers) == "Tim Cook"

    def test_extracts_from_ceo_title(self):
        from pipeline.knowledge.profile_grounding import _extract_ceo
        officers = [{"name": "Jensen Huang", "title": "CEO"}]
        assert _extract_ceo(officers) == "Jensen Huang"

    def test_returns_none_for_empty_list(self):
        from pipeline.knowledge.profile_grounding import _extract_ceo
        assert _extract_ceo([]) is None

    def test_skips_non_ceo_officers(self):
        from pipeline.knowledge.profile_grounding import _extract_ceo
        officers = [
            {"name": "Luca Maestri", "title": "Chief Financial Officer"},
            {"name": "Tim Cook", "title": "Chief Executive Officer"},
        ]
        assert _extract_ceo(officers) == "Tim Cook"

    def test_returns_none_when_no_ceo_present(self):
        from pipeline.knowledge.profile_grounding import _extract_ceo
        officers = [{"name": "Luca Maestri", "title": "Chief Financial Officer"}]
        assert _extract_ceo(officers) is None


# ---------------------------------------------------------------------------
# _get_sector_peers tests
# ---------------------------------------------------------------------------

class TestGetSectorPeers:
    def test_returns_same_sector_tickers(self, db_conn):
        from pipeline.knowledge.profile_grounding import _get_sector_peers
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        db_conn.executemany(
            "INSERT INTO watchlist (ticker, company_name, sector, added_at) VALUES (?,?,?,?)",
            [
                ("AAPL", "Apple Inc.", "Information Technology", now),
                ("MSFT", "Microsoft Corp.", "Information Technology", now),
                ("GOOGL", "Alphabet Inc.", "Information Technology", now),
                ("JPM", "JPMorgan Chase", "Financials", now),
            ],
        )
        db_conn.commit()

        peers = _get_sector_peers(db_conn, "AAPL", "Information Technology")
        assert "AAPL" not in peers
        assert set(peers) == {"MSFT", "GOOGL"}

    def test_excludes_self(self, db_conn):
        from pipeline.knowledge.profile_grounding import _get_sector_peers
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        db_conn.execute(
            "INSERT INTO watchlist (ticker, company_name, sector, added_at) VALUES (?,?,?,?)",
            ("AAPL", "Apple Inc.", "Information Technology", now),
        )
        db_conn.commit()

        peers = _get_sector_peers(db_conn, "AAPL", "Information Technology")
        assert "AAPL" not in peers

    def test_caps_at_five(self, db_conn):
        from pipeline.knowledge.profile_grounding import _get_sector_peers, _MAX_PEER_TICKERS
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        rows = [(f"T{i}", f"Company {i}", "Technology", now) for i in range(10)]
        db_conn.executemany(
            "INSERT INTO watchlist (ticker, company_name, sector, added_at) VALUES (?,?,?,?)",
            rows,
        )
        db_conn.commit()

        peers = _get_sector_peers(db_conn, "T0", "Technology")
        assert len(peers) <= _MAX_PEER_TICKERS


# ---------------------------------------------------------------------------
# fetch_ticker_fact_pack tests
# ---------------------------------------------------------------------------

class TestFetchTickerFactPack:
    @pytest.mark.asyncio
    async def test_grounded_pack_on_success(self, db_conn):
        from pipeline.knowledge.profile_grounding import fetch_ticker_fact_pack

        mock_ticker = _make_yf_ticker(
            info=_make_info(),
            news=[{"title": "Apple Q2 Results"}, {"title": "iPhone demand rises"}],
        )
        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", return_value=mock_ticker):
            pack = await fetch_ticker_fact_pack(db_conn, "AAPL", "Apple Inc.", "Information Technology")

        assert pack.grounding_available is True
        assert pack.ticker == "AAPL"
        assert pack.ceo_name == "Tim Cook"
        assert pack.industry == "Consumer Electronics"
        assert pack.market_cap == 3_190_000_000_000
        assert pack.trailing_pe == 31.2
        assert pack.current_price == 197.30
        assert "Apple Q2 Results" in pack.recent_headlines

    @pytest.mark.asyncio
    async def test_minimal_pack_when_info_fails(self, db_conn):
        from pipeline.knowledge.profile_grounding import fetch_ticker_fact_pack

        def raising_ticker(sym):
            t = MagicMock()
            t.info = None  # empty info
            t.news = []
            return t

        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", side_effect=raising_ticker):
            pack = await fetch_ticker_fact_pack(db_conn, "AAPL", "Apple Inc.", "Information Technology")

        assert pack.grounding_available is False
        assert pack.ticker == "AAPL"
        assert pack.company_name == "Apple Inc."
        assert pack.sector == "Information Technology"
        assert pack.ceo_name is None
        assert pack.market_cap is None

    @pytest.mark.asyncio
    async def test_grounding_false_when_info_raises(self, db_conn):
        from pipeline.knowledge.profile_grounding import fetch_ticker_fact_pack

        def exploding_ticker(sym):
            t = MagicMock()
            type(t).info = property(lambda self: (_ for _ in ()).throw(ConnectionError("timeout")))
            t.news = []
            return t

        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", side_effect=exploding_ticker):
            pack = await fetch_ticker_fact_pack(db_conn, "AAPL", "Apple Inc.", "Information Technology")

        assert pack.grounding_available is False

    @pytest.mark.asyncio
    async def test_business_summary_truncated(self, db_conn):
        from pipeline.knowledge.profile_grounding import fetch_ticker_fact_pack, _BUSINESS_SUMMARY_MAX_CHARS

        long_summary = "A" * (_BUSINESS_SUMMARY_MAX_CHARS + 500)
        mock_ticker = _make_yf_ticker(info=_make_info(summary=long_summary))
        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", return_value=mock_ticker):
            pack = await fetch_ticker_fact_pack(db_conn, "AAPL", "Apple Inc.", "Information Technology")

        assert pack.business_summary is not None
        assert len(pack.business_summary) <= _BUSINESS_SUMMARY_MAX_CHARS + 10  # +10 for ellipsis

    @pytest.mark.asyncio
    async def test_peer_tickers_from_watchlist(self, db_conn):
        from pipeline.knowledge.profile_grounding import fetch_ticker_fact_pack
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        db_conn.executemany(
            "INSERT INTO watchlist (ticker, company_name, sector, added_at) VALUES (?,?,?,?)",
            [
                ("AAPL", "Apple Inc.", "Information Technology", now),
                ("MSFT", "Microsoft Corp.", "Information Technology", now),
                ("GOOGL", "Alphabet Inc.", "Information Technology", now),
            ],
        )
        db_conn.commit()

        mock_ticker = _make_yf_ticker(info=_make_info())
        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", return_value=mock_ticker):
            pack = await fetch_ticker_fact_pack(db_conn, "AAPL", "Apple Inc.", "Information Technology")

        assert "MSFT" in pack.peer_tickers
        assert "GOOGL" in pack.peer_tickers
        assert "AAPL" not in pack.peer_tickers

    @pytest.mark.asyncio
    async def test_headlines_capped_at_max(self, db_conn):
        from pipeline.knowledge.profile_grounding import fetch_ticker_fact_pack, _MAX_HEADLINES

        many_news = [{"title": f"Headline {i}"} for i in range(20)]
        mock_ticker = _make_yf_ticker(info=_make_info(), news=many_news)
        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", return_value=mock_ticker):
            pack = await fetch_ticker_fact_pack(db_conn, "AAPL", "Apple Inc.", "Information Technology")

        assert len(pack.recent_headlines) <= _MAX_HEADLINES


# ---------------------------------------------------------------------------
# fetch_macro_fact_pack tests
# ---------------------------------------------------------------------------

class TestFetchMacroFactPack:
    @pytest.mark.asyncio
    async def test_grounded_pack_on_success(self):
        from pipeline.knowledge.profile_grounding import fetch_macro_fact_pack

        def make_ticker(sym):
            t = MagicMock()
            prices = {
                "^TNX": 4.32, "^IRX": 5.20, "^VIX": 18.5,
                "DX-Y.NYB": 103.4, "SPY": 520.1, "QQQ": 442.3,
            }
            t.info = {"regularMarketPrice": prices.get(sym, 100.0)}
            t.news = [{"title": "Fed holds rates steady at May meeting"}]
            return t

        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", side_effect=make_ticker):
            pack = await fetch_macro_fact_pack()

        assert pack.grounding_available is True
        assert pack.treasury_10y == pytest.approx(4.32)
        assert pack.vix == pytest.approx(18.5)
        assert pack.spy_level == pytest.approx(520.1)

    @pytest.mark.asyncio
    async def test_grounding_false_when_all_prices_fail(self):
        from pipeline.knowledge.profile_grounding import fetch_macro_fact_pack

        def broken_ticker(sym):
            t = MagicMock()
            t.info = {}
            t.news = []
            return t

        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", side_effect=broken_ticker):
            pack = await fetch_macro_fact_pack()

        assert pack.grounding_available is False

    @pytest.mark.asyncio
    async def test_macro_headlines_filtered_by_keyword(self):
        from pipeline.knowledge.profile_grounding import fetch_macro_fact_pack

        def ticker_with_mixed_news(sym):
            t = MagicMock()
            t.info = {"regularMarketPrice": 500.0}
            t.news = [
                {"title": "Fed raises rates by 25bps"},
                {"title": "Tech stocks rally on earnings"},  # not macro
                {"title": "CPI report shows inflation steady"},
            ]
            return t

        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", side_effect=ticker_with_mixed_news):
            pack = await fetch_macro_fact_pack()

        macro_titles = " ".join(pack.macro_headlines).lower()
        assert "fed raises rates" in macro_titles or "cpi report" in macro_titles
        # Non-macro headline should not appear unless keywords match
        for h in pack.macro_headlines:
            lower = h.lower()
            from pipeline.knowledge.profile_grounding import _MACRO_KEYWORDS
            assert any(kw in lower for kw in _MACRO_KEYWORDS), (
                f"Non-macro headline leaked into macro_headlines: {h!r}"
            )

    @pytest.mark.asyncio
    async def test_fetched_at_is_set(self):
        from pipeline.knowledge.profile_grounding import fetch_macro_fact_pack

        def empty_ticker(sym):
            t = MagicMock()
            t.info = {}
            t.news = []
            return t

        with patch("pipeline.knowledge.profile_grounding.yf.Ticker", side_effect=empty_ticker):
            pack = await fetch_macro_fact_pack()

        assert pack.fetched_at != ""
