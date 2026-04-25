"""Tests du `AlertsEngine` (Phase 3 étape 4)."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable

import pytest

from alerts.engine import AlertsEngine, _extract_dedupe_key
from alerts.notifier import EmailNotifier
from alerts.rules import AlertCandidate, Rule
from memory.database import Alert, session_scope


# ---------------------------------------------------------------- helpers


class _StaticRule(Rule):
    """Règle test : retourne une liste prédéfinie de candidats."""

    def __init__(
        self,
        name: str,
        candidates: list[AlertCandidate],
        severity: str = "info",
    ) -> None:
        self.name = name
        self.severity = severity
        self._candidates = candidates
        self.calls = 0

    def evaluate(self, as_of: datetime) -> Iterable[AlertCandidate]:
        self.calls += 1
        return self._candidates


class _RaisingRule(Rule):
    """Règle test qui lève une exception — vérifie la tolérance."""

    name = "boom"
    severity = "info"

    def evaluate(self, as_of: datetime) -> Iterable[AlertCandidate]:
        raise RuntimeError("boom")


class _DummyNotifier(EmailNotifier):
    """Notifier qui enregistre les envois en mémoire au lieu d'appeler SMTP."""

    def __init__(self, *, enabled: bool = True) -> None:
        # Court-circuite la lecture des settings.
        self._enabled = enabled
        self.sent: list[tuple[str, str]] = []
        self._config = object() if enabled else None  # truthy / None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def send(self, *, subject: str, body: str) -> bool:
        self.sent.append((subject, body))
        return True


def _make_candidate(
    *,
    rule_name: str = "r",
    severity: str = "info",
    dedupe_key: str = "k:1",
    message: str = "hello",
    asset_id: str | None = None,
    sector_id: str | None = None,
    thesis_id: int | None = None,
) -> AlertCandidate:
    return AlertCandidate(
        rule_name=rule_name,
        severity=severity,
        message=message,
        dedupe_key=dedupe_key,
        asset_id=asset_id,
        sector_id=sector_id,
        thesis_id=thesis_id,
    )


def _alerts_in_db() -> list[Alert]:
    with session_scope() as s:
        rows = s.query(Alert).order_by(Alert.id.asc()).all()
        for r in rows:
            s.expunge(r)
    return rows


# --------------------------------------------------------- _extract_dedupe_key


def test_extract_dedupe_key_normal():
    assert _extract_dedupe_key(json.dumps({"dedupe_key": "abc"})) == "abc"


def test_extract_dedupe_key_missing():
    assert _extract_dedupe_key(json.dumps({"x": 1})) is None


def test_extract_dedupe_key_invalid_json():
    assert _extract_dedupe_key("not json") is None


def test_extract_dedupe_key_none():
    assert _extract_dedupe_key(None) is None


def test_extract_dedupe_key_array_payload():
    """Un payload JSON valide mais pas dict → None (pas de plantage)."""
    assert _extract_dedupe_key("[1, 2, 3]") is None


# ----------------------------------------------------------- AlertsEngine


def test_engine_persists_candidates(tmp_db):
    rule = _StaticRule("r1", [
        _make_candidate(rule_name="r1", dedupe_key="k:1", asset_id="NVDA"),
        _make_candidate(rule_name="r1", dedupe_key="k:2", asset_id="AMD"),
    ])
    engine = AlertsEngine(rules=[rule], notifier=_DummyNotifier(enabled=False))
    n = engine.run(as_of=datetime(2026, 4, 25))
    assert n == 2
    rows = _alerts_in_db()
    assert len(rows) == 2
    assert {r.asset_id for r in rows} == {"NVDA", "AMD"}


def test_engine_dedupes_against_existing(tmp_db):
    rule = _StaticRule("r1", [
        _make_candidate(rule_name="r1", dedupe_key="k:1"),
        _make_candidate(rule_name="r1", dedupe_key="k:2"),
    ])
    engine = AlertsEngine(rules=[rule], notifier=_DummyNotifier(enabled=False))
    assert engine.run(as_of=datetime(2026, 4, 25)) == 2
    # Re-run identique → 0 nouveau (dédupliqué via dedupe_key).
    assert engine.run(as_of=datetime(2026, 4, 25)) == 0
    assert len(_alerts_in_db()) == 2


def test_engine_dedupes_within_same_run(tmp_db):
    """Si une règle retourne deux fois le même dedupe_key, un seul insert."""
    rule = _StaticRule("r1", [
        _make_candidate(rule_name="r1", dedupe_key="k:1"),
        _make_candidate(rule_name="r1", dedupe_key="k:1"),
    ])
    engine = AlertsEngine(rules=[rule], notifier=_DummyNotifier(enabled=False))
    n = engine.run(as_of=datetime(2026, 4, 25))
    assert n == 1
    assert len(_alerts_in_db()) == 1


def test_engine_isolates_failing_rules(tmp_db):
    good = _StaticRule("r1", [
        _make_candidate(rule_name="r1", dedupe_key="k:1"),
    ])
    bad = _RaisingRule()
    engine = AlertsEngine(
        rules=[bad, good, bad],
        notifier=_DummyNotifier(enabled=False),
    )
    n = engine.run(as_of=datetime(2026, 4, 25))
    assert n == 1
    rows = _alerts_in_db()
    assert len(rows) == 1


def test_engine_dispatches_to_notifier(tmp_db):
    rule = _StaticRule("r1", [
        _make_candidate(rule_name="r1", dedupe_key="k:1",
                        message="Test alert"),
    ])
    notifier = _DummyNotifier(enabled=True)
    engine = AlertsEngine(rules=[rule], notifier=notifier)
    engine.run(as_of=datetime(2026, 4, 25))
    assert len(notifier.sent) == 1
    subject, body = notifier.sent[0]
    assert "ALPHA RADAR" in subject
    assert "INFO" in subject
    assert "Test alert" in body


def test_engine_does_not_dispatch_when_notifier_disabled(tmp_db):
    rule = _StaticRule("r1", [
        _make_candidate(rule_name="r1", dedupe_key="k:1"),
    ])
    notifier = _DummyNotifier(enabled=False)
    engine = AlertsEngine(rules=[rule], notifier=notifier)
    engine.run(as_of=datetime(2026, 4, 25))
    assert notifier.sent == []


def test_engine_does_not_renotify_dedup_skipped(tmp_db):
    rule = _StaticRule("r1", [
        _make_candidate(rule_name="r1", dedupe_key="k:1"),
    ])
    notifier = _DummyNotifier(enabled=True)
    engine = AlertsEngine(rules=[rule], notifier=notifier)
    engine.run(as_of=datetime(2026, 4, 25))
    engine.run(as_of=datetime(2026, 4, 25))   # re-run
    assert len(notifier.sent) == 1   # pas de double notification


def test_engine_persists_dedupe_key_in_data_json(tmp_db):
    rule = _StaticRule("r1", [
        _make_candidate(rule_name="r1", dedupe_key="my:key"),
    ])
    engine = AlertsEngine(rules=[rule], notifier=_DummyNotifier(enabled=False))
    engine.run(as_of=datetime(2026, 4, 25))
    row = _alerts_in_db()[0]
    parsed = json.loads(row.data_json)
    assert parsed["dedupe_key"] == "my:key"


def test_engine_default_rules_run_without_data(tmp_db):
    """Sans aucune thèse / éval / heat, l'engine produit 0 alerte mais
    ne crashe pas avec les règles par défaut."""
    engine = AlertsEngine(notifier=_DummyNotifier(enabled=False))
    assert engine.run(as_of=datetime(2026, 4, 25)) == 0
