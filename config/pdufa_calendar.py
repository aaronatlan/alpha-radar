"""Calendrier PDUFA manuellement curé pour la watchlist biotech.

Pourquoi un fichier statique ?
------------------------------
La FDA ne publie **pas d'API publique gratuite** des PDUFA dates à
venir. Les sources marchandes (BioPharma Catalyst, Evaluate Pharma)
sont payantes, et scraper ces sites n'est pas robuste pour de la
production.

À la place, on maintient ici une liste à jour manuellement à partir
des annonces officielles (press releases sponsors, SEC 8-K). Le
calendrier alimente la règle d'alerte critique `PDUFANearRule`
(SPEC §7.7 — « PDUFA <30 jours avec score >70 »).

Format : un dict par PDUFA avec :
    - `ticker` : ticker watchlist du sponsor
    - `target_action_date` : date PDUFA (YYYY-MM-DD)
    - `drug` : nom du produit / candidat
    - `indication` : indication thérapeutique
    - `application_number` : BLA/NDA si connu (optionnel)

Mode dégradé : si une date est passée, la règle d'alerte la skippe.
À nettoyer (commenter / archiver) après l'événement pour garder ce
fichier court — les approvals atterrissent ensuite dans `raw_data`
via le `FDACollector` avec leur date factuelle.
"""
from __future__ import annotations

from datetime import date
from typing import TypedDict


class PDUFAEntry(TypedDict, total=False):
    ticker: str
    target_action_date: str   # ISO YYYY-MM-DD
    drug: str
    indication: str
    application_number: str   # optionnel


# Liste vide à la sortie de Phase 4 — à peupler quand des PDUFA réelles
# sont identifiées sur les biotechs suivis. L'absence d'entrée n'est pas
# une erreur : la règle d'alerte se contentera de logger un INFO.
PDUFA_CALENDAR: list[PDUFAEntry] = []


def upcoming_pdufas(as_of: date) -> list[PDUFAEntry]:
    """PDUFAs futures (date >= `as_of`), triées par date croissante."""
    out: list[PDUFAEntry] = []
    for entry in PDUFA_CALENDAR:
        try:
            d = date.fromisoformat(entry["target_action_date"])
        except (KeyError, ValueError):
            continue
        if d >= as_of:
            out.append(entry)
    return sorted(out, key=lambda e: e["target_action_date"])
