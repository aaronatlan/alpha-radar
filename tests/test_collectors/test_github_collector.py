"""Tests du `GitHubCollector` — tous les appels HTTP sont mockés."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from collectors.github_collector import GitHubCollector


def _fake_response(status: int = 200, json_data: dict | None = None, text: str = ""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    r.text = text
    return r


REPOS = [{"owner": "huggingface", "repo": "transformers", "sectors": ["ai_ml"]}]


def test_normalize_produces_canonical_item():
    c = GitHubCollector(repos=REPOS)
    raw = {
        "definition": REPOS[0],
        "payload": {
            "stargazers_count": 123000,
            "forks_count": 25000,
            "open_issues_count": 900,
            "subscribers_count": 1000,
            "language": "Python",
            "pushed_at": "2026-04-01T12:00:00Z",
            "updated_at": "2026-04-01T12:00:00Z",
        },
    }
    item = c.normalize(raw)
    assert item is not None
    assert item["entity_type"] == "github_repo_snapshot"
    assert item["entity_id"].startswith("huggingface/transformers:")
    assert item["payload"]["stars"] == 123000
    assert item["payload"]["sectors"] == ["ai_ml"]


def test_normalize_rejects_incomplete():
    c = GitHubCollector(repos=REPOS)
    assert c.normalize({"payload": None, "definition": REPOS[0]}) is None
    assert c.normalize({"payload": {}, "definition": None}) is None


def test_collect_handles_http_errors_gracefully():
    c = GitHubCollector(repos=REPOS)
    c.request_delay = 0
    with patch("collectors.github_collector.requests.get",
               return_value=_fake_response(status=500, text="oops")):
        out = c.collect(since=None, until=None)  # args ignorés
    assert out == []


def test_collect_breaks_on_rate_limit():
    """Un 403 avec 'rate limit' doit couper la boucle immédiatement."""
    two_repos = REPOS + [{"owner": "pytorch", "repo": "pytorch", "sectors": ["ai_ml"]}]
    c = GitHubCollector(repos=two_repos)
    c.request_delay = 0
    resp = _fake_response(status=403, text="API rate limit exceeded")
    with patch("collectors.github_collector.requests.get", return_value=resp) as m:
        out = c.collect(since=None, until=None)
    assert out == []
    assert m.call_count == 1  # break après le premier


def test_collect_success_round_trip():
    c = GitHubCollector(repos=REPOS)
    c.request_delay = 0
    payload = {
        "stargazers_count": 100, "forks_count": 10,
        "open_issues_count": 1, "subscribers_count": 5,
        "language": "Python", "pushed_at": None, "updated_at": None,
    }
    with patch("collectors.github_collector.requests.get",
               return_value=_fake_response(status=200, json_data=payload)):
        out = c.collect(since=None, until=None)
    assert len(out) == 1
    assert out[0]["payload"]["stargazers_count"] == 100
