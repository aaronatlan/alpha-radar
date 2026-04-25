"""Tests du `CoinGeckoCollector` — appels HTTP mockés."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from collectors.coingecko_collector import CoinGeckoCollector


def _fake_response(status: int = 200, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data if json_data is not None else []
    return r


def test_collect_empty_if_no_ids():
    c = CoinGeckoCollector(coin_ids=[])
    c.request_delay = 0
    assert c.collect(since=None, until=None) == []


def test_collect_parses_list_response():
    c = CoinGeckoCollector(coin_ids=["bitcoin", "ethereum"])
    c.request_delay = 0
    data = [
        {
            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
            "current_price": 120_000, "market_cap": 2_400_000_000_000,
            "total_volume": 30_000_000_000,
            "price_change_percentage_24h": 1.2,
            "high_24h": 121_000, "low_24h": 119_000,
            "circulating_supply": 19_700_000,
        },
        {
            "id": "ethereum", "symbol": "eth", "name": "Ethereum",
            "current_price": 4200, "market_cap": 500_000_000_000,
            "total_volume": 20_000_000_000,
            "price_change_percentage_24h": -0.8,
            "high_24h": 4250, "low_24h": 4150,
            "circulating_supply": 120_000_000,
        },
    ]
    with patch("collectors.coingecko_collector.requests.get",
               return_value=_fake_response(status=200, json_data=data)):
        out = c.collect(since=None, until=None)
    assert len(out) == 2
    assert out[0]["id"] == "bitcoin"


def test_collect_handles_http_error():
    c = CoinGeckoCollector(coin_ids=["bitcoin"])
    c.request_delay = 0
    resp = MagicMock(status_code=503, text="down")
    with patch("collectors.coingecko_collector.requests.get", return_value=resp):
        out = c.collect(since=None, until=None)
    assert out == []


def test_normalize_maps_fields():
    c = CoinGeckoCollector(coin_ids=["bitcoin"])
    raw = {
        "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
        "current_price": 120_000, "market_cap": 2e12,
        "total_volume": 3e10,
        "price_change_percentage_24h": 1.5,
        "high_24h": 121_000, "low_24h": 119_000,
        "circulating_supply": 19_700_000,
    }
    item = c.normalize(raw)
    assert item is not None
    assert item["entity_type"] == "crypto_daily"
    assert item["entity_id"].startswith("bitcoin:")
    assert item["payload"]["price_usd"] == 120_000
    assert item["payload"]["market_cap_usd"] == 2e12


def test_normalize_none_without_id():
    c = CoinGeckoCollector(coin_ids=["bitcoin"])
    assert c.normalize({"id": None}) is None
