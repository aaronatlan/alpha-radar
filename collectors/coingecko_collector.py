"""Collecteur CoinGecko — snapshot marché quotidien pour une watchlist crypto.

API publique gratuite, pas de clé obligatoire. Endpoint utilisé :
`GET /coins/markets?vs_currency=usd&ids=...`.

Un seul appel HTTP couvre tous les coin_ids — rate limit large (~10-30
req/min sur le tier public, bien au-dessus de notre besoin quotidien).

Un snapshot par coin et par jour (`YYYY-MM-DD` en UTC), `content_at` =
minuit UTC du jour de collecte. La volatilité intraday est donc perdue
— suffisant pour des features quotidiennes (momentum, volume ratio).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.watchlists import CRYPTO_COIN_IDS, CRYPTO_WATCHLIST


class CoinGeckoCollector(BaseCollector):
    """Snapshot marché quotidien des cryptos de la watchlist."""

    source_name = "coingecko"
    request_delay = 1.0
    api_root = "https://api.coingecko.com/api/v3"
    timeout = 20.0
    vs_currency = "usd"

    def __init__(self, coin_ids: list[str] | None = None) -> None:
        super().__init__()
        self._coin_ids: list[str] = (
            list(coin_ids) if coin_ids is not None else list(CRYPTO_COIN_IDS)
        )
        self._sectors_by_id = {c["coin_id"]: c["sectors"] for c in CRYPTO_WATCHLIST}

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
        """Récupère le snapshot marché pour tous les coin_ids d'un coup.

        `since` / `until` ignorés : l'endpoint `/coins/markets` ne renvoie
        que le spot. Pour de l'historique granulaire, il faudrait
        `/coins/{id}/market_chart` (non utilisé ici : suffit pour Phase 2).
        """
        if not self._coin_ids:
            return []

        url = f"{self.api_root}/coins/markets"
        params = {
            "vs_currency": self.vs_currency,
            "ids": ",".join(self._coin_ids),
            "price_change_percentage": "24h",
        }
        try:
            r = requests.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("[coingecko] GET /coins/markets a échoué : {}", exc)
            return []

        if r.status_code != 200:
            logger.warning("[coingecko] HTTP {} : {}", r.status_code, r.text[:200])
            return []

        data = r.json()
        if not isinstance(data, list):
            logger.warning("[coingecko] format inattendu : {}", type(data).__name__)
            return []

        self._throttle()
        return data

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: dict[str, Any]) -> NormalizedItem | None:
        coin_id = raw.get("id")
        if not coin_id:
            return None

        today_utc = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )
        entity_id = f"{coin_id}:{today_utc.strftime('%Y-%m-%d')}"

        payload = {
            "coin_id": coin_id,
            "symbol": raw.get("symbol"),
            "name": raw.get("name"),
            "sectors": list(self._sectors_by_id.get(coin_id, [])),
            "price_usd": raw.get("current_price"),
            "market_cap_usd": raw.get("market_cap"),
            "volume_24h_usd": raw.get("total_volume"),
            "price_change_24h_pct": raw.get("price_change_percentage_24h"),
            "high_24h_usd": raw.get("high_24h"),
            "low_24h_usd": raw.get("low_24h"),
            "circulating_supply": raw.get("circulating_supply"),
        }
        return NormalizedItem(
            entity_type="crypto_daily",
            entity_id=entity_id,
            content_at=today_utc,
            payload=payload,
        )
