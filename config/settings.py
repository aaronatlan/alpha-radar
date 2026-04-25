"""Configuration centrale chargée depuis `.env` via pydantic-settings.

Ce module est la seule source de vérité pour les paths, niveaux de log et
clés API. Tout le reste du code importe depuis ici — jamais de `os.getenv`
disséminé dans les modules.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Paramètres d'application chargés depuis les variables d'environnement.

    Les variables sont préfixées `ALPHA_` dans `.env` (ex: `ALPHA_LOG_LEVEL`)
    et exposées sans préfixe sur cet objet (ex: `settings.log_level`).
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="ALPHA_",
        extra="ignore",
    )

    env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    db_path: Path = Field(default=Path("data/alpha_radar.db"))
    data_dir: Path = Field(default=Path("data"))

    # Délai par défaut (secondes) entre deux requêtes d'un même collecteur.
    # Les collecteurs peuvent surcharger via leur attribut de classe.
    default_request_delay: float = 1.0

    # --- Clés et identifiants d'API externes (tous optionnels) ---
    #
    # Toutes les variables portent le préfixe `ALPHA_` en `.env`
    # (cf. `env_prefix` ci-dessus) :
    #   - ALPHA_GITHUB_TOKEN : PAT GitHub (rate limit 5000/h avec,
    #     60/h sans). Si absent, le collecteur fonctionne en mode anonyme.
    #     Le `GITHUB_TOKEN` brut (gh CLI / Actions) n'est **pas** lu par
    #     pydantic-settings — préfixer explicitement pour l'activer.
    #   - ALPHA_NEWSAPI_KEY : clé newsapi.org (tier gratuit 100 req/jour).
    #     Si absente, le collecteur est un no-op silencieux.
    #   - ALPHA_SEC_USER_AGENT : l'API SEC EDGAR exige un User-Agent
    #     identifiant (nom + email). Format recommandé : "Nom Prénom email".
    github_token: str | None = None
    newsapi_key: str | None = None
    sec_user_agent: str = "Alpha Radar alpha-radar@example.com"

    # --- SMTP (alertes email, Phase 3 étape 4) ---
    #
    # Si l'un des champs `smtp_host`, `smtp_user`, `smtp_password`,
    # `smtp_to` est absent, le notifier est inerte : les alertes restent
    # écrites en base mais aucun email n'est envoyé. C'est volontaire —
    # le moteur d'alertes doit fonctionner même sans config email.
    #
    #   - ALPHA_SMTP_HOST     : ex. smtp.gmail.com
    #   - ALPHA_SMTP_PORT     : 587 (STARTTLS, défaut) | 465 (SSL natif)
    #   - ALPHA_SMTP_USER     : adresse email d'auth
    #   - ALPHA_SMTP_PASSWORD : mot de passe applicatif (jamais le mot de
    #     passe principal — utiliser un app password)
    #   - ALPHA_SMTP_FROM     : expéditeur affiché (par défaut = SMTP_USER)
    #   - ALPHA_SMTP_TO       : destinataire (un seul en v1)
    #   - ALPHA_SMTP_USE_TLS  : true (défaut, STARTTLS) | false (SSL natif)
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_to: str | None = None
    smtp_use_tls: bool = True

    @property
    def db_url(self) -> str:
        """URL SQLAlchemy pour la base SQLite principale.

        Le dossier parent est créé à la volée si nécessaire.
        """
        path = self.db_path
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"


settings = Settings()


_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    """Configure loguru. Idempotent (appelable plusieurs fois sans effet)."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> | {message}"
        ),
    )

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "alpha-radar-{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
    )
    _LOGGING_CONFIGURED = True
