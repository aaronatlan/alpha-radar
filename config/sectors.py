"""Définition des secteurs suivis par le système.

Chaque secteur est un périmètre thématique pour lequel un Heat Score sera
calculé en Phase 2. Les catégories arXiv permettent au collecteur
académique de Phase 1 de savoir quels flux ingérer.

La liste est volontairement réduite pour la Phase 1 — elle s'étoffera
en Phase 4 (enrichissement sectoriel : pharma, défense, spatial…).
"""
from __future__ import annotations

from typing import TypedDict


class SectorDefinition(TypedDict):
    """Structure d'un secteur suivi."""

    id: str
    name: str
    category: str
    keywords: list[str]
    arxiv_categories: list[str]


SECTORS: list[SectorDefinition] = [
    {
        "id": "quantum_computing",
        "name": "Ordinateur quantique",
        "category": "deep_tech",
        "keywords": ["quantum", "qubit", "superconducting", "ion trap", "quantum advantage"],
        "arxiv_categories": ["quant-ph"],
    },
    {
        "id": "ai_ml",
        "name": "Intelligence artificielle / ML",
        "category": "software",
        "keywords": ["large language model", "transformer", "neural network", "llm", "diffusion"],
        "arxiv_categories": ["cs.AI", "cs.LG", "cs.CL"],
    },
    {
        "id": "computer_vision",
        "name": "Vision par ordinateur",
        "category": "software",
        "keywords": ["object detection", "segmentation", "vision transformer", "3d reconstruction"],
        "arxiv_categories": ["cs.CV"],
    },
    {
        "id": "robotics",
        "name": "Robotique",
        "category": "deep_tech",
        "keywords": ["manipulation", "locomotion", "humanoid", "slam"],
        "arxiv_categories": ["cs.RO"],
    },
    {
        "id": "cybersecurity",
        "name": "Cybersécurité",
        "category": "software",
        "keywords": ["vulnerability", "exploit", "zero-day", "malware", "cryptography"],
        "arxiv_categories": ["cs.CR"],
    },
    {
        "id": "biotech",
        "name": "Biotechnologie",
        "category": "pharma",
        "keywords": ["gene therapy", "crispr", "mrna", "protein folding"],
        "arxiv_categories": ["q-bio"],
    },
    {
        "id": "space",
        "name": "Spatial",
        "category": "deep_tech",
        "keywords": ["satellite", "launcher", "orbit", "constellation"],
        "arxiv_categories": ["astro-ph.IM", "physics.space-ph"],
    },
]


SECTORS_BY_ID: dict[str, SectorDefinition] = {s["id"]: s for s in SECTORS}


def all_arxiv_categories() -> list[str]:
    """Liste dédupliquée (ordre préservé) des catégories arXiv suivies.

    Utilisée par ArxivCollector pour savoir quels flux ingérer sans
    dupliquer les requêtes quand deux secteurs partagent une catégorie.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for sector in SECTORS:
        for cat in sector["arxiv_categories"]:
            if cat not in seen:
                seen.add(cat)
                ordered.append(cat)
    return ordered
