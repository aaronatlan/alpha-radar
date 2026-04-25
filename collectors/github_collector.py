"""Collecteur GitHub — snapshot quotidien des métriques de dépôts surveillés.

API publique : `GET https://api.github.com/repos/{owner}/{repo}`.
Pas de clé obligatoire, mais le rate limit passe de 60/h à 5000/h dès
qu'on fournit un token (`ALPHA_GITHUB_TOKEN`).

Sémantique snapshot
-------------------
GitHub ne fournit pas l'historique `stars_count` par jour via l'API
publique. On capture donc un **snapshot quotidien** : `content_at` est
fixé à la date du run (UTC, minuit). La dérivée temporelle (vélocité
stars/jour) est reconstruite côté features en comparant deux snapshots
successifs.

`entity_id = "{owner}/{repo}:{YYYY-MM-DD}"` garantit l'unicité par
(repo, jour). La contrainte UNIQUE `(source, entity_id, hash)` fait le
reste : deux runs le même jour ne doublonnent pas si rien n'a changé.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.github_repos import GITHUB_REPOS, RepoDefinition, repo_full_name
from config.settings import settings


class GitHubCollector(BaseCollector):
    """Collecte les métriques publiques de dépôts GitHub surveillés."""

    source_name = "github"
    request_delay = 0.5
    api_root = "https://api.github.com"
    timeout = 15.0

    def __init__(self, repos: list[RepoDefinition] | None = None) -> None:
        super().__init__()
        self._repos: list[RepoDefinition] = list(repos) if repos else list(GITHUB_REPOS)
        self._token = settings.github_token

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
        """Snapshot de chaque repo. `since` / `until` ignorés — l'API GitHub
        publique ne donne pas l'historique ; on collecte toujours au spot."""
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        items: list[dict[str, Any]] = []
        for definition in self._repos:
            url = f"{self.api_root}/repos/{repo_full_name(definition)}"
            try:
                r = requests.get(url, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                logger.warning(
                    "[github] GET {} a échoué : {}", repo_full_name(definition), exc
                )
                self._throttle()
                continue

            if r.status_code == 403 and "rate limit" in r.text.lower():
                logger.warning(
                    "[github] rate limit atteint — arrêt anticipé de la collecte"
                )
                break
            if r.status_code != 200:
                logger.warning(
                    "[github] GET {} : HTTP {}",
                    repo_full_name(definition), r.status_code,
                )
                self._throttle()
                continue

            data = r.json()
            items.append({"definition": definition, "payload": data})
            self._throttle()

        return items

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: dict[str, Any]) -> NormalizedItem | None:
        data = raw.get("payload")
        definition = raw.get("definition")
        if not data or not definition:
            return None

        today_utc = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )
        full_name = repo_full_name(definition)
        entity_id = f"{full_name}:{today_utc.strftime('%Y-%m-%d')}"

        payload = {
            "full_name": full_name,
            "sectors": list(definition.get("sectors", [])),
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "open_issues": data.get("open_issues_count"),
            "subscribers": data.get("subscribers_count"),
            "language": data.get("language"),
            "pushed_at": data.get("pushed_at"),
            "updated_at": data.get("updated_at"),
        }
        return NormalizedItem(
            entity_type="github_repo_snapshot",
            entity_id=entity_id,
            content_at=today_utc,
            payload=payload,
        )
