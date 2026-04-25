"""Notification email — Phase 3 étape 4.

Encapsule l'envoi SMTP via la stdlib (`smtplib` + `email.message`). Pas
de dépendance externe : on reste sur des protocoles éprouvés.

Politique
---------
- Si la config SMTP est incomplète, le notifier est inerte (`is_enabled`
  vaut False) et `send` log-and-skip silencieusement. Le moteur d'alertes
  continue d'écrire en base, l'email manque seulement.
- Une exception SMTP n'interrompt pas le batch : on log au niveau warning
  et on poursuit. La trace en base reste exploitable.

Variables d'environnement (préfixe ALPHA_)
------------------------------------------
- ALPHA_SMTP_HOST     : ex. `smtp.gmail.com`
- ALPHA_SMTP_PORT     : ex. `587` (STARTTLS) ou `465` (SSL)
- ALPHA_SMTP_USER     : adresse email d'auth
- ALPHA_SMTP_PASSWORD : mot de passe applicatif
- ALPHA_SMTP_FROM     : expéditeur affiché (par défaut = SMTP_USER)
- ALPHA_SMTP_TO       : destinataire (single recipient en v1)
- ALPHA_SMTP_USE_TLS  : `true` (STARTTLS, défaut) | `false` (SSL natif si port=465)
"""
from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

from loguru import logger

from config.settings import settings


@dataclass(frozen=True)
class SMTPConfig:
    """Snapshot des paramètres SMTP nécessaires à l'envoi."""

    host: str
    port: int
    user: str
    password: str
    sender: str
    recipient: str
    use_tls: bool = True

    @classmethod
    def from_settings(cls) -> "SMTPConfig | None":
        """Construit la config depuis `settings`. Renvoie None si incomplète."""
        host = getattr(settings, "smtp_host", None)
        user = getattr(settings, "smtp_user", None)
        password = getattr(settings, "smtp_password", None)
        recipient = getattr(settings, "smtp_to", None)
        if not (host and user and password and recipient):
            return None
        port = int(getattr(settings, "smtp_port", 587) or 587)
        sender = getattr(settings, "smtp_from", None) or user
        use_tls = bool(getattr(settings, "smtp_use_tls", True))
        return cls(
            host=host,
            port=port,
            user=user,
            password=password,
            sender=sender,
            recipient=recipient,
            use_tls=use_tls,
        )


class EmailNotifier:
    """Envoie un email par alerte. No-op si SMTP non configuré.

    L'objet est construit une fois par run du moteur ; la connexion SMTP
    est ouverte/fermée par envoi (volumes faibles, pas de pooling utile
    en Phase 3).
    """

    def __init__(self, config: SMTPConfig | None = None) -> None:
        self._config = config if config is not None else SMTPConfig.from_settings()
        if self._config is None:
            logger.info(
                "[notifier] SMTP non configuré — alertes écrites en base "
                "uniquement. Définir ALPHA_SMTP_HOST/USER/PASSWORD/TO pour "
                "activer l'email."
            )

    @property
    def is_enabled(self) -> bool:
        return self._config is not None

    def send(self, *, subject: str, body: str) -> bool:
        """Envoie un email. Retourne True si succès, False sinon."""
        if self._config is None:
            return False
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self._config.sender
        msg["To"] = self._config.recipient
        msg.set_content(body)
        try:
            self._send(msg)
            return True
        except (smtplib.SMTPException, OSError) as exc:
            # Ne tue pas le batch — l'alerte reste en base, l'email est
            # juste manqué. Le prochain run peut re-tenter via un mode
            # digest (non implémenté ici).
            logger.warning(
                "[notifier] Échec envoi email ({}) : {}",
                type(exc).__name__, exc,
            )
            return False

    # --- internals ------------------------------------------------------

    def _send(self, msg: EmailMessage) -> None:
        """Ouvre une connexion SMTP, authentifie et envoie."""
        cfg = self._config
        assert cfg is not None  # garde-fou — `send` a déjà vérifié
        if cfg.use_tls:
            with smtplib.SMTP(cfg.host, cfg.port, timeout=10) as smtp:
                smtp.starttls()
                smtp.login(cfg.user, cfg.password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP_SSL(cfg.host, cfg.port, timeout=10) as smtp:
                smtp.login(cfg.user, cfg.password)
                smtp.send_message(msg)


def format_subject(severity: str, rule_name: str, summary: str) -> str:
    """Sujet email standard : `[ALPHA RADAR][SEVERITY][rule] résumé`."""
    sev = severity.upper()
    return f"[ALPHA RADAR][{sev}][{rule_name}] {summary}"
