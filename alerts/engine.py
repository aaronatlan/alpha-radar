"""Moteur d'alertes — Phase 3 étape 4.

Orchestre :
1. l'évaluation de chaque `Rule` configurée,
2. la déduplication contre l'existant en base via `data_json.dedupe_key`,
3. l'insertion d'une ligne `alerts` par candidat retenu,
4. la notification email (si SMTP est configuré).

Tolérance aux pannes
--------------------
Une règle qui lève une exception est sautée — le batch continue avec les
autres règles. Idem pour l'envoi email : un échec SMTP est loggé et
n'empêche ni l'écriture en base ni les règles suivantes.

Idempotence
-----------
Chaque candidat porte un `dedupe_key` stable (ex. `thesis:42` ou
`heat_surge:ai_ml:2026-04-25`). Avant insertion, le moteur consulte la
table `alerts` filtrée sur `rule_name` et compare les `dedupe_key`
existants — pas de doublon créé sur re-run.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable

from loguru import logger
from sqlalchemy import select

from alerts.notifier import EmailNotifier, format_subject
from alerts.rules import DEFAULT_RULES, AlertCandidate, Rule
from memory.database import Alert, session_scope, utc_now


class AlertsEngine:
    """Évalue toutes les règles, dédupe, persiste, notifie."""

    def __init__(
        self,
        rules: Iterable[Rule] | None = None,
        notifier: EmailNotifier | None = None,
    ) -> None:
        self._rules: list[Rule] = list(rules) if rules is not None else list(DEFAULT_RULES)
        self._notifier = notifier if notifier is not None else EmailNotifier()

    def run(self, as_of: datetime | None = None) -> int:
        """Évalue toutes les règles à `as_of`. Retourne le nb d'alertes créées."""
        ts = as_of or utc_now()
        logger.info("[alerts] {} règle(s) à évaluer à {}", len(self._rules), ts)
        n_created = 0
        for rule in self._rules:
            try:
                n_created += self._run_rule(rule, ts)
            except Exception as exc:
                logger.warning(
                    "[alerts] règle '{}' a échoué : {}", rule.name, exc,
                )
        logger.info("[alerts] Terminé : {} alerte(s) créée(s)", n_created)
        return n_created

    # --- pipeline ------------------------------------------------------

    def _run_rule(self, rule: Rule, as_of: datetime) -> int:
        candidates = list(rule.evaluate(as_of))
        if not candidates:
            return 0
        existing_keys = self._existing_keys(rule.name)
        n = 0
        for c in candidates:
            if c.dedupe_key in existing_keys:
                continue
            self._persist(c, created_at=as_of)
            self._notify(c)
            n += 1
            existing_keys.add(c.dedupe_key)
            logger.info(
                "[alerts] +1 [{}][{}] {}",
                c.severity.upper(), c.rule_name, c.message,
            )
        return n

    def _existing_keys(self, rule_name: str) -> set[str]:
        """Lit les dedupe_keys déjà persistés pour cette règle."""
        stmt = (
            select(Alert.data_json)
            .where(Alert.rule_name == rule_name)
        )
        keys: set[str] = set()
        with session_scope() as session:
            for (data_json,) in session.execute(stmt).all():
                key = _extract_dedupe_key(data_json)
                if key:
                    keys.add(key)
        return keys

    def _persist(self, c: AlertCandidate, *, created_at: datetime) -> None:
        with session_scope() as session:
            session.add(Alert(
                created_at=created_at,
                rule_name=c.rule_name,
                severity=c.severity,
                asset_type=c.asset_type,
                asset_id=c.asset_id,
                sector_id=c.sector_id,
                message=c.message,
                data_json=c.data_json(),
                thesis_id=c.thesis_id,
                acknowledged=False,
            ))

    def _notify(self, c: AlertCandidate) -> None:
        """Délègue à l'email notifier si activé. Silencieux sinon."""
        if not self._notifier.is_enabled:
            return
        # Résumé court pour le sujet : on tronque le message à ~60 chars.
        summary = c.message.split("\n", 1)[0][:60]
        subject = format_subject(c.severity, c.rule_name, summary)
        body = (
            f"{c.message}\n\n"
            f"Règle : {c.rule_name}\n"
            f"Sévérité : {c.severity}\n"
            f"Données : {json.dumps(c.data, indent=2, default=str)}\n"
        )
        self._notifier.send(subject=subject, body=body)


# ---------------------------------------------------------------- helpers


def _extract_dedupe_key(data_json: str | None) -> str | None:
    """Lit `dedupe_key` dans le JSON de l'alerte, tolère le malformé."""
    if not data_json:
        return None
    try:
        d = json.loads(data_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    key = d.get("dedupe_key")
    return str(key) if key else None
