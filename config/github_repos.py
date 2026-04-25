"""Dépôts GitHub open source surveillés, indexés par secteur.

Un dépôt est pertinent quand il est **central** pour un secteur : les
dynamiques de stars / forks / commits y sont une proxy crédible de
l'adoption ou du momentum académique. Liste volontairement courte en
Phase 2 — elle s'étoffera avec les secteurs pharma/défense en Phase 4.
"""
from __future__ import annotations

from typing import TypedDict


class RepoDefinition(TypedDict):
    owner: str
    repo: str
    sectors: list[str]


GITHUB_REPOS: list[RepoDefinition] = [
    # IA / ML
    {"owner": "huggingface", "repo": "transformers", "sectors": ["ai_ml"]},
    {"owner": "pytorch", "repo": "pytorch", "sectors": ["ai_ml"]},
    {"owner": "ggerganov", "repo": "llama.cpp", "sectors": ["ai_ml"]},
    {"owner": "langchain-ai", "repo": "langchain", "sectors": ["ai_ml"]},
    # Vision
    {"owner": "ultralytics", "repo": "ultralytics", "sectors": ["computer_vision"]},
    # Quantique
    {"owner": "Qiskit", "repo": "qiskit", "sectors": ["quantum_computing"]},
    {"owner": "quantumlib", "repo": "Cirq", "sectors": ["quantum_computing"]},
    # Robotique
    {"owner": "ros2", "repo": "ros2", "sectors": ["robotics"]},
    # Cybersécurité
    {"owner": "projectdiscovery", "repo": "nuclei", "sectors": ["cybersecurity"]},
    # Biotech / bio
    {"owner": "deepmind", "repo": "alphafold", "sectors": ["biotech"]},
]


def repo_full_name(definition: RepoDefinition) -> str:
    """`owner/repo` — format canonique GitHub."""
    return f"{definition['owner']}/{definition['repo']}"
