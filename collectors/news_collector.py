"""Collecteur NewsAPI — articles récents par mot-clé sectoriel.

Tier gratuit : 100 req/jour, historique limité à 30 jours. Une clé est
obligatoire (`ALPHA_NEWSAPI_KEY`). **Si la clé est absente, le
collecteur devient un no-op silencieux** — conforme au principe SPEC
§12 "mode dégradé sans casser le système".

Une requête = un mot-clé / secteur. On utilise les `keywords` de
`config/sectors.py` en joignant par OR, ce qui reste dans les quotas
free tier pour une poignée de secteurs et un run quotidien.

`content_at` = `publishedAt` de l'article (UTC naïf). `entity_id` = URL
canonique. La contrainte UNIQUE `(source, entity_id, hash)` dédoublonne
les ré-indexations de NewsAPI.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.sectors import SECTORS, SectorDefinition
from config.settings import settings


class NewsAPICollector(BaseCollector):
    """Articles d'actualité par secteur, via newsapi.org."""

    source_name = "newsapi"
    request_delay = 1.0
    endpoint = "https://newsapi.org/v2/everything"
    timeout = 20.0
    page_size = 50

    def __init__(self, sectors: list[SectorDefinition] | None = None) -> None:
        super().__init__()
        self._sectors: list[SectorDefinition] = list(sectors) if sectors else list(SECTORS)
        self._api_key = settings.newsapi_key

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
        if not self._api_key:
            logger.info("[newsapi] ALPHA_NEWSAPI_KEY absent — collecte skippée.")
            return []

        since_iso = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") \
            if since.tzinfo else since.strftime("%Y-%m-%dT%H:%M:%S")
        until_iso = until.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") \
            if until.tzinfo else until.strftime("%Y-%m-%dT%H:%M:%S")

        items: list[dict[str, Any]] = []
        for sector in self._sectors:
            query = _build_query(sector["keywords"])
            if not query:
                continue

            params = {
                "q": query,
                "from": since_iso,
                "to": until_iso,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": self.page_size,
                "apiKey": self._api_key,
            }
            try:
                r = requests.get(self.endpoint, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                logger.warning(
                    "[newsapi] GET secteur {} : {}", sector["id"], exc
                )
                self._throttle()
                continue

            if r.status_code == 429:
                logger.warning("[newsapi] rate limit atteint — arrêt anticipé")
                break
            if r.status_code != 200:
                logger.warning(
                    "[newsapi] secteur {} : HTTP {}",
                    sector["id"], r.status_code,
                )
                self._throttle()
                continue

            data = r.json()
            if data.get("status") != "ok":
                logger.warning(
                    "[newsapi] secteur {} : status={}",
                    sector["id"], data.get("status"),
                )
                self._throttle()
                continue

            for article in data.get("articles", []):
                items.append({"sector_id": sector["id"], "article": article})

            self._throttle()

        return items

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: dict[str, Any]) -> NormalizedItem | None:
        article = raw.get("article") or {}
        url = article.get("url")
        if not url:
            return None

        published = article.get("publishedAt")
        content_at: datetime | None = None
        if published:
            # NewsAPI renvoie ISO8601 avec 'Z'
            try:
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                content_at = dt.astimezone(timezone.utc).replace(tzinfo=None)
            except ValueError:
                content_at = None

        payload = {
            "sector_id": raw.get("sector_id"),
            "title": article.get("title"),
            "description": article.get("description"),
            "content": article.get("content"),
            "url": url,
            "source_name": (article.get("source") or {}).get("name"),
            "published_at": published,
            "author": article.get("author"),
        }
        return NormalizedItem(
            entity_type="news_article",
            entity_id=url,
            content_at=content_at,
            payload=payload,
        )


def _build_query(keywords: list[str]) -> str:
    """OR logique des mots-clés, guillemets autour des expressions à espaces.

    Exemple : `["gene therapy", "crispr"]` → `"gene therapy" OR crispr`.
    """
    parts: list[str] = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        parts.append(f'"{kw}"' if " " in kw else kw)
    return " OR ".join(parts)
