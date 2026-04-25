"""Collecteur ClinicalTrials.gov — essais cliniques des biotech suivies.

API v2 publique (`https://clinicaltrials.gov/api/v2/studies`). Pas de
clé requise. On filtre par sponsor (nom de société dans la watchlist
biotech) et par date de dernière mise à jour pour récupérer en
priorité les études actives.

Chaque étude est une entité distincte, clé = `NCTId` (identifiant
global unique). `content_at` = `lastUpdatePostDate` — c'est cette
date qui fait foi pour la reconstruction PIT (la date d'inscription
de l'essai est souvent ancienne et peu informative).

Champs collectés (subset minimal mais utile pour le scoring pharma) :
    - nct_id, ticker, sponsor
    - brief_title, conditions, interventions
    - overall_status (RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED…)
    - phase (PHASE1 → PHASE4)
    - start_date, completion_date
    - last_update_post_date

Mode dégradé : un sponsor sans match côté API renvoie 0 résultat —
on log un warning et on continue sur les autres. Une erreur réseau
ne tue pas la collecte globale.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.watchlists import STOCK_WATCHLIST


# Sponsors connus avec un nom légal différent du nom usuel watchlist.
# Étendre au fur et à mesure des observations terrain.
SPONSOR_NAME_OVERRIDES: dict[str, str] = {
    # MRNA est référencé sous "ModernaTX, Inc." sur certains essais
    # mais "Moderna" matche aussi ; on garde le nom watchlist par défaut.
}


def _biotech_sponsors() -> dict[str, str]:
    """Mapping ticker → nom de sponsor à requêter depuis la watchlist."""
    return {
        item["ticker"]: SPONSOR_NAME_OVERRIDES.get(item["ticker"], item["name"])
        for item in STOCK_WATCHLIST
        if "biotech" in item["sectors"]
    }


class ClinicalTrialsCollector(BaseCollector):
    """Collecte les essais cliniques liés aux sponsors biotech suivis."""

    source_name = "clinicaltrials"
    request_delay = 0.5
    timeout = 30.0
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    #: Limite par sponsor (par run). Au-delà on tronque — Phase 4 reste
    #: en mode dégradé sur la pagination, à étendre si volumes le justifient.
    page_size = 100

    def __init__(self, sponsors: dict[str, str] | None = None) -> None:
        super().__init__()
        self._sponsors = dict(sponsors) if sponsors is not None else _biotech_sponsors()

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for ticker, sponsor_name in self._sponsors.items():
            studies = self._fetch_sponsor(sponsor_name, since, until)
            for study in studies:
                study["_ticker"] = ticker
                study["_sponsor_query"] = sponsor_name
            items.extend(studies)
            self._throttle()
        return items

    def _fetch_sponsor(
        self, sponsor: str, since: datetime, until: datetime,
    ) -> list[dict[str, Any]]:
        """Appelle l'API pour un sponsor donné. Retourne la liste brute."""
        params = {
            "query.spons": sponsor,
            "pageSize": self.page_size,
            "format": "json",
            "filter.advanced": (
                f"AREA[LastUpdatePostDate]RANGE["
                f"{since.date().isoformat()},{until.date().isoformat()}]"
            ),
        }
        try:
            r = requests.get(self.base_url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("[clinicaltrials] GET {} : {}", sponsor, exc)
            return []

        if r.status_code != 200:
            logger.warning(
                "[clinicaltrials] {} : HTTP {}", sponsor, r.status_code,
            )
            return []
        try:
            data = r.json()
        except ValueError:
            logger.warning("[clinicaltrials] {} : JSON invalide", sponsor)
            return []
        return list(data.get("studies") or [])

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: dict[str, Any]) -> NormalizedItem | None:
        proto = raw.get("protocolSection") or {}
        ident = proto.get("identificationModule") or {}
        status = proto.get("statusModule") or {}
        design = proto.get("designModule") or {}
        sponsors = proto.get("sponsorCollaboratorsModule") or {}
        conditions_mod = proto.get("conditionsModule") or {}
        interv_mod = proto.get("armsInterventionsModule") or {}

        nct_id = ident.get("nctId")
        if not nct_id:
            return None

        last_update_str = (
            (status.get("lastUpdatePostDateStruct") or {}).get("date")
        )
        content_at = _parse_date(last_update_str)
        if content_at is None:
            return None

        lead_sponsor = (sponsors.get("leadSponsor") or {}).get("name")
        phases = design.get("phases") or []
        # ClinicalTrials renvoie une liste de phases : on prend la plus avancée.
        phase = _most_advanced_phase(phases)

        interventions = [
            iv.get("name") for iv in (interv_mod.get("interventions") or [])
            if iv.get("name")
        ]
        conditions = list(conditions_mod.get("conditions") or [])

        payload = {
            "nct_id": nct_id,
            "ticker": raw.get("_ticker"),
            "sponsor_query": raw.get("_sponsor_query"),
            "lead_sponsor": lead_sponsor,
            "brief_title": ident.get("briefTitle"),
            "overall_status": status.get("overallStatus"),
            "phase": phase,
            "phases": phases,
            "start_date": _struct_date(status.get("startDateStruct")),
            "completion_date": _struct_date(status.get("completionDateStruct")),
            "primary_completion_date": _struct_date(
                status.get("primaryCompletionDateStruct")
            ),
            "last_update_post_date": last_update_str,
            "conditions": conditions,
            "interventions": interventions,
        }
        return NormalizedItem(
            entity_type="clinical_trial",
            entity_id=nct_id,
            content_at=content_at,
            payload=payload,
        )


# ----------------------------------------------------------------- helpers


_PHASE_RANK = {
    "EARLY_PHASE1": 0,
    "PHASE1": 1,
    "PHASE1/PHASE2": 2,
    "PHASE2": 3,
    "PHASE2/PHASE3": 4,
    "PHASE3": 5,
    "PHASE4": 6,
    "NA": -1,
}


def _most_advanced_phase(phases: list[str]) -> str | None:
    """Retourne la phase la plus avancée d'une liste, None si vide."""
    if not phases:
        return None
    return max(phases, key=lambda p: _PHASE_RANK.get(p, -2))


def _parse_date(value: str | None) -> datetime | None:
    """Parse une date YYYY-MM-DD ou YYYY-MM ; retourne None si échec."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _struct_date(struct: dict[str, Any] | None) -> str | None:
    """Extrait `date` d'une structure ClinicalTrials (peut être None)."""
    if not isinstance(struct, dict):
        return None
    return struct.get("date")
