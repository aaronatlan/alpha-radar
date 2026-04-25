"""Collecteur FDA — approvals médicaments via openFDA.

API publique : `https://api.fda.gov/drug/drugsfda.json`. Pas de clé
requise, mais rate-limit non authentifié à ~240 req/min — largement
suffisant pour notre watchlist biotech.

On collecte uniquement les **submissions approuvées** (`AP`) pour les
sponsors de la watchlist. Chaque approval est une entité distincte,
clé = `application_number + submission_number` (concat unique chez FDA).
`content_at` = `submission_status_date` (date d'approbation).

Pourquoi pas les PDUFA dates ?
------------------------------
openFDA n'expose pas les calendriers PDUFA *à venir* — seulement les
décisions historiques. Le tracking PDUFA en amont est géré par
`config/pdufa_calendar.py` (manuel) + règle d'alerte dédiée.

Champs collectés :
    - application_number, submission_number, sponsor_name
    - submission_type (ORIG, SUPPL, …)
    - submission_status (AP), submission_status_date
    - brand_name, active_ingredients, dosage_form
    - ticker (mappé via watchlist biotech)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.watchlists import STOCK_WATCHLIST


# Surcharges si le nom légal sur openFDA diffère du nom watchlist.
SPONSOR_NAME_OVERRIDES: dict[str, str] = {
    # MRNA est référencé sous "MODERNATX INC" sur openFDA.
    "MRNA": "MODERNATX",
}


def _biotech_sponsors() -> dict[str, str]:
    """Mapping ticker → nom de sponsor à requêter (UPPERCASE comme openFDA)."""
    return {
        item["ticker"]: SPONSOR_NAME_OVERRIDES.get(
            item["ticker"], item["name"].upper()
        )
        for item in STOCK_WATCHLIST
        if "biotech" in item["sectors"]
    }


class FDACollector(BaseCollector):
    """Collecte les approvals FDA pour les sponsors biotech suivis."""

    source_name = "fda"
    request_delay = 0.3
    timeout = 30.0
    base_url = "https://api.fda.gov/drug/drugsfda.json"

    #: Limite par sponsor (par run). openFDA accepte jusqu'à 1000.
    page_size = 100

    def __init__(self, sponsors: dict[str, str] | None = None) -> None:
        super().__init__()
        self._sponsors = dict(sponsors) if sponsors is not None else _biotech_sponsors()

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for ticker, sponsor in self._sponsors.items():
            results = self._fetch_sponsor(sponsor, since, until)
            for app in results:
                # On expand chaque submission approuvée en item séparé.
                for sub in (app.get("submissions") or []):
                    if sub.get("submission_status") != "AP":
                        continue
                    items.append({
                        "ticker": ticker,
                        "sponsor_query": sponsor,
                        "application": app,
                        "submission": sub,
                    })
            self._throttle()
        return items

    def _fetch_sponsor(
        self, sponsor: str, since: datetime, until: datetime,
    ) -> list[dict[str, Any]]:
        """Appelle openFDA pour un sponsor, filtré sur la fenêtre."""
        # openFDA attend YYYYMMDD ; espaces ' ' compris dans la valeur de
        # `search` sont URL-encodés correctement par requests.
        date_lo = since.strftime("%Y%m%d")
        date_hi = until.strftime("%Y%m%d")
        search = (
            f'sponsor_name:"{sponsor}"'
            f' AND submissions.submission_status:"AP"'
            f' AND submissions.submission_status_date:[{date_lo} TO {date_hi}]'
        )
        params = {"search": search, "limit": self.page_size}
        try:
            r = requests.get(self.base_url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("[fda] GET {} : {}", sponsor, exc)
            return []

        # openFDA renvoie 404 quand 0 résultat — c'est normal, pas un échec.
        if r.status_code == 404:
            return []
        if r.status_code != 200:
            logger.warning("[fda] {} : HTTP {}", sponsor, r.status_code)
            return []
        try:
            data = r.json()
        except ValueError:
            logger.warning("[fda] {} : JSON invalide", sponsor)
            return []
        return list(data.get("results") or [])

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: dict[str, Any]) -> NormalizedItem | None:
        app = raw.get("application") or {}
        sub = raw.get("submission") or {}

        app_number = app.get("application_number")
        sub_number = sub.get("submission_number")
        if not app_number or not sub_number:
            return None

        date_str = sub.get("submission_status_date")
        content_at = _parse_yyyymmdd(date_str)
        if content_at is None:
            return None

        products = app.get("products") or []
        # Premier produit suffit pour le scoring ; on conserve la liste
        # complète dans le payload pour traçabilité.
        first_product = products[0] if products else {}
        brand = first_product.get("brand_name")
        active = [
            ai.get("name") for ai in (first_product.get("active_ingredients") or [])
            if ai.get("name")
        ]

        payload = {
            "ticker": raw.get("ticker"),
            "sponsor_query": raw.get("sponsor_query"),
            "sponsor_name": app.get("sponsor_name"),
            "application_number": app_number,
            "submission_number": sub_number,
            "submission_type": sub.get("submission_type"),
            "submission_status": sub.get("submission_status"),
            "submission_status_date": date_str,
            "submission_class_code": sub.get("submission_class_code"),
            "brand_name": brand,
            "active_ingredients": active,
            "dosage_form": first_product.get("dosage_form"),
            "n_products": len(products),
        }
        # Clé unique = application + submission. Stable chez la FDA.
        entity_id = f"{app_number}-{sub_number}"
        return NormalizedItem(
            entity_type="fda_approval",
            entity_id=entity_id,
            content_at=content_at,
            payload=payload,
        )


# ------------------------------------------------------------------ helpers


def _parse_yyyymmdd(value: str | None) -> datetime | None:
    """Parse YYYYMMDD (format openFDA). None si invalide."""
    if not value or len(value) != 8:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return None
