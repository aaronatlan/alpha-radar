"""Collecteur USASpending.gov — contrats gouvernementaux US.

API publique : `POST https://api.usaspending.gov/api/v2/search/spending_by_award/`.
Pas de clé requise. Limite implicite : ~10 req/s, on reste largement
en-deçà. La requête est un POST JSON (pas un GET), particularité du
data.gov moderne.

On cible les sponsors de la watchlist au secteur `space` (Phase 4 v1 :
LMT, RKLB) — l'extension à `defense` / `cybersecurity` viendra avec
l'élargissement de watchlist en Étapes 5/6.

Chaque contrat est une entité distincte, clé = `Award ID` (string
unique chez USASpending). `content_at` = `Action Date` (date de
l'attribution / dernière modification du contrat).

Pagination
----------
On collecte la page 1 (taille 50) par sponsor et par run. Les contrats
plus anciens « tombent » naturellement hors fenêtre `[since, until]`
au prochain cycle. Si volume devient un sujet, paginer via
`page_metadata.hasNext` et incrementer `page`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.watchlists import STOCK_WATCHLIST


#: Ticker → nom de récipiendaire à requêter sur USASpending.
SPONSOR_NAME_OVERRIDES: dict[str, str] = {
    # Le nom officiel chez USASpending peut différer.
    "LMT": "Lockheed Martin",
    "RKLB": "Rocket Lab",
    "RTX": "RTX Corporation",
    "NOC": "Northrop Grumman",
    "PLTR": "Palantir Technologies",
}

#: Secteurs de la watchlist éligibles à la collecte de contrats fédéraux.
#: `defense` couvre LMT, RTX, NOC, PLTR ; `space` ajoute RKLB.
DEFENSE_SECTORS: tuple[str, ...] = ("space", "defense")


def _defense_sponsors() -> dict[str, str]:
    """Mapping ticker → nom de récipiendaire dérivé de la watchlist."""
    return {
        item["ticker"]: SPONSOR_NAME_OVERRIDES.get(
            item["ticker"], item["name"]
        )
        for item in STOCK_WATCHLIST
        if any(s in DEFENSE_SECTORS for s in item["sectors"])
    }


# Codes "Award Type" USASpending pour les contrats stricts (hors grants,
# loans, IDVs). A = BPA Call, B = Purchase Order, C = Delivery Order,
# D = Definitive Contract.
CONTRACT_TYPE_CODES: list[str] = ["A", "B", "C", "D"]


class USASpendingCollector(BaseCollector):
    """Collecte les contrats US des sponsors défense/spatial suivis."""

    source_name = "usaspending"
    request_delay = 0.5
    timeout = 30.0
    base_url = (
        "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    )

    page_size = 50

    #: Champs minimaux à demander à l'API (réduit la latence et le poids).
    fields: list[str] = [
        "Award ID",
        "Recipient Name",
        "Award Amount",
        "Description",
        "Action Date",
        "Awarding Agency",
        "Awarding Sub Agency",
        "Award Type",
        "Period of Performance Start Date",
        "Period of Performance Current End Date",
    ]

    def __init__(self, sponsors: dict[str, str] | None = None) -> None:
        super().__init__()
        self._sponsors = (
            dict(sponsors) if sponsors is not None else _defense_sponsors()
        )

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for ticker, sponsor in self._sponsors.items():
            results = self._fetch_sponsor(sponsor, since, until)
            for award in results:
                award["_ticker"] = ticker
                award["_sponsor_query"] = sponsor
                items.append(award)
            self._throttle()
        return items

    def _fetch_sponsor(
        self, sponsor: str, since: datetime, until: datetime,
    ) -> list[dict[str, Any]]:
        body = {
            "filters": {
                "recipient_search_text": [sponsor],
                "time_period": [{
                    "start_date": since.date().isoformat(),
                    "end_date": until.date().isoformat(),
                }],
                "award_type_codes": CONTRACT_TYPE_CODES,
            },
            "fields": self.fields,
            "page": 1,
            "limit": self.page_size,
            "sort": "Action Date",
            "order": "desc",
        }
        try:
            r = requests.post(self.base_url, json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("[usaspending] POST {} : {}", sponsor, exc)
            return []

        if r.status_code != 200:
            logger.warning(
                "[usaspending] {} : HTTP {}", sponsor, r.status_code,
            )
            return []
        try:
            data = r.json()
        except ValueError:
            logger.warning("[usaspending] {} : JSON invalide", sponsor)
            return []
        return list(data.get("results") or [])

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: dict[str, Any]) -> NormalizedItem | None:
        award_id = raw.get("Award ID") or raw.get("generated_internal_id")
        if not award_id:
            return None

        action_date_str = raw.get("Action Date")
        content_at = _parse_date(action_date_str)
        if content_at is None:
            return None

        amount = _to_float(raw.get("Award Amount"))

        payload = {
            "ticker": raw.get("_ticker"),
            "sponsor_query": raw.get("_sponsor_query"),
            "award_id": award_id,
            "recipient_name": raw.get("Recipient Name"),
            "award_amount": amount,
            "description": raw.get("Description"),
            "action_date": action_date_str,
            "awarding_agency": raw.get("Awarding Agency"),
            "awarding_sub_agency": raw.get("Awarding Sub Agency"),
            "award_type": raw.get("Award Type"),
            "period_start": raw.get("Period of Performance Start Date"),
            "period_end": raw.get("Period of Performance Current End Date"),
        }
        return NormalizedItem(
            entity_type="gov_contract",
            entity_id=str(award_id),
            content_at=content_at,
            payload=payload,
        )


# ------------------------------------------------------------------ helpers


def _parse_date(value: str | None) -> datetime | None:
    """Parse une date ISO YYYY-MM-DD, None si invalide."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    """Convertit en float, tolère str/None/int."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
