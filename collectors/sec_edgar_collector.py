"""Collecteur SEC EDGAR — filings récents des émetteurs surveillés.

API publique : `GET https://data.sec.gov/submissions/CIK{10digits}.json`.
Pas de clé, mais **User-Agent identifiant obligatoire** (nom + email —
règle SEC). Configuré via `ALPHA_SEC_USER_AGENT`.

Chaque filing est une entité distincte, clé = accession_number (unique
globalement chez SEC). `content_at` = `filing_date` (date légale du
dépôt). La fenêtre `[since, until]` filtre sur `filing_date`.

Formes suivies en Phase 2 (SPEC §6.3) : 10-K, 10-Q, 8-K, 13D, 13G,
Form D, S-1, 4. Les autres formes (424B*, CORRESP…) sont ignorées
pour limiter le volume.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.settings import settings
from config.watchlists import TICKER_TO_CIK, cik_padded


FORMS_OF_INTEREST = {
    "10-K", "10-Q", "8-K", "SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A",
    "D", "S-1", "S-1/A", "4", "6-K",
}


class SECEdgarCollector(BaseCollector):
    """Collecte les filings récents par CIK."""

    source_name = "sec_edgar"
    request_delay = 0.2  # SEC demande ≤ 10 req/s — on reste bien en-dessous
    timeout = 20.0
    submissions_url = "https://data.sec.gov/submissions/CIK{cik10}.json"

    def __init__(self, ticker_to_cik: dict[str, str] | None = None) -> None:
        super().__init__()
        self._map: dict[str, str] = dict(ticker_to_cik or TICKER_TO_CIK)

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
        since_d = since.date()
        until_d = until.date()

        headers = {
            "User-Agent": settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

        items: list[dict[str, Any]] = []
        for ticker, cik in self._map.items():
            url = self.submissions_url.format(cik10=cik_padded(cik))
            try:
                r = requests.get(url, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                logger.warning("[sec_edgar] GET {} : {}", ticker, exc)
                self._throttle()
                continue

            if r.status_code != 200:
                logger.warning(
                    "[sec_edgar] {} : HTTP {}", ticker, r.status_code
                )
                self._throttle()
                continue

            try:
                data = r.json()
            except ValueError:
                logger.warning("[sec_edgar] {} : JSON invalide", ticker)
                self._throttle()
                continue

            recent = (data.get("filings") or {}).get("recent") or {}
            # Les champs de `recent` sont des listes alignées par index.
            accessions = recent.get("accessionNumber") or []
            forms = recent.get("form") or []
            dates = recent.get("filingDate") or []
            primary_docs = recent.get("primaryDocument") or []

            for i, accession in enumerate(accessions):
                form = forms[i] if i < len(forms) else None
                date_str = dates[i] if i < len(dates) else None
                if form not in FORMS_OF_INTEREST or not date_str:
                    continue
                try:
                    filing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if filing_date < since_d or filing_date > until_d:
                    continue

                items.append(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "accession": accession,
                        "form": form,
                        "filing_date": date_str,
                        "primary_doc": (
                            primary_docs[i] if i < len(primary_docs) else None
                        ),
                        "company_name": data.get("name"),
                    }
                )

            self._throttle()

        return items

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: dict[str, Any]) -> NormalizedItem | None:
        accession = raw.get("accession")
        filing_date_str = raw.get("filing_date")
        if not accession or not filing_date_str:
            return None
        try:
            filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
        except ValueError:
            return None
        # content_at = filing_date à minuit UTC naïf
        content_at = filing_date.replace(tzinfo=timezone.utc).replace(tzinfo=None)

        payload = {
            "ticker": raw.get("ticker"),
            "cik": raw.get("cik"),
            "accession": accession,
            "form": raw.get("form"),
            "filing_date": filing_date_str,
            "primary_document": raw.get("primary_doc"),
            "company_name": raw.get("company_name"),
        }
        return NormalizedItem(
            entity_type="sec_filing",
            entity_id=accession,
            content_at=content_at,
            payload=payload,
        )
