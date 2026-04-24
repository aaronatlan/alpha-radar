"""Tests unitaires du collecteur yfinance — sans I/O réseau."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from collectors.yfinance_collector import YFinanceCollector


def _fake_history_df() -> pd.DataFrame:
    """Deux séances tz-aware typiques d'un retour `Ticker.history()`."""
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"]).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 101.0],
            "Close": [104.0, 105.5],
            "Adj Close": [104.0, 105.5],
            "Volume": [1_000_000, 1_200_000],
        },
        index=idx,
    )


# --------------------------------------------------------------- normalize


def test_normalize_produces_canonical_ohlcv():
    c = YFinanceCollector(tickers=["NVDA"])
    ts = datetime(2026, 1, 6, 21, 0, tzinfo=timezone.utc)
    raw = {
        "ticker": "NVDA", "timestamp": ts,
        "open": 100.0, "high": 110.0, "low": 95.0,
        "close": 108.0, "adj_close": 108.0, "volume": 1_000_000,
    }
    item = c.normalize(raw)

    assert item is not None
    assert item["entity_type"] == "ohlcv_daily"
    assert item["entity_id"] == "NVDA:2026-01-06"
    assert item["payload"]["close"] == 108.0
    assert item["payload"]["session_date"] == "2026-01-06"
    # Convention : content_at stocké naïf en UTC.
    assert item["content_at"].tzinfo is None


def test_normalize_rejects_missing_ticker_or_timestamp():
    c = YFinanceCollector(tickers=["NVDA"])
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert c.normalize({"ticker": None, "timestamp": now}) is None
    assert c.normalize({"ticker": "NVDA", "timestamp": None}) is None


# ------------------------------------------------------------------ collect


def test_collect_produces_one_item_per_session():
    c = YFinanceCollector(tickers=["NVDA"])
    c.request_delay = 0
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _fake_history_df()

    with patch(
        "collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker
    ):
        items = c.collect(datetime(2026, 1, 1), datetime(2026, 1, 10))

    assert len(items) == 2
    assert all(i["ticker"] == "NVDA" for i in items)
    assert items[0]["close"] == 104.0
    assert items[1]["volume"] == 1_200_000


def test_collect_degraded_mode_on_ticker_exception():
    """Un ticker qui plante ne doit pas casser la collecte des autres."""
    c = YFinanceCollector(tickers=["GOOD", "BAD"])
    c.request_delay = 0

    def ticker_factory(ticker):
        m = MagicMock()
        if ticker == "BAD":
            m.history.side_effect = RuntimeError("yahoo rate limit")
        else:
            m.history.return_value = _fake_history_df()
        return m

    with patch(
        "collectors.yfinance_collector.yf.Ticker", side_effect=ticker_factory
    ):
        items = c.collect(datetime(2026, 1, 1), datetime(2026, 1, 10))

    assert len(items) == 2
    assert all(i["ticker"] == "GOOD" for i in items)


def test_collect_empty_history_is_handled():
    """DataFrame vide : pas d'exception, pas d'item produit."""
    c = YFinanceCollector(tickers=["NVDA"])
    c.request_delay = 0
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch(
        "collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker
    ):
        items = c.collect(datetime(2026, 1, 1), datetime(2026, 1, 10))

    assert items == []


# ----------------------------------------------------------------- run / store


def test_run_stores_and_deduplicates(tmp_db):
    """Re-collecter la même période ne crée aucun nouveau doublon."""
    c = YFinanceCollector(tickers=["NVDA"])
    c.request_delay = 0
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _fake_history_df()

    with patch(
        "collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker
    ):
        inserted_1 = c.run(datetime(2026, 1, 1), datetime(2026, 1, 10))
        inserted_2 = c.run(datetime(2026, 1, 1), datetime(2026, 1, 10))

    assert inserted_1 == 2
    assert inserted_2 == 0
