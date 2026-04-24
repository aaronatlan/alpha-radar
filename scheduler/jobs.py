"""Scheduler minimal — Phase 1.

Deux jobs récurrents :
  - arXiv : quotidien 06:00 Europe/Paris, catégories configurées
  - yfinance : jours ouvrés 22:00 Europe/Paris, watchlist configurée

La base est initialisée au démarrage. Une erreur dans un job ne tue
pas le scheduler (mode dégradé : le prochain tick réessaiera).

Les fenêtres glissantes créent un chevauchement volontaire d'une
exécution à l'autre ; la déduplication dans `raw_data` garantit
l'idempotence, ce qui rend ce chevauchement sûr et utile (filet contre
les trous de collecte).
"""
from __future__ import annotations

from datetime import timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from collectors.arxiv_collector import ArxivCollector
from collectors.yfinance_collector import YFinanceCollector
from config.settings import configure_logging
from memory.database import init_db, utc_now


def run_arxiv_job() -> None:
    """Collecte arXiv sur une fenêtre glissante de 48 h."""
    until = utc_now()
    since = until - timedelta(days=2)
    try:
        ArxivCollector().run(since, until)
    except Exception as exc:  # pragma: no cover — BaseCollector.run avale déjà
        logger.exception("Job arXiv : exception non rattrapée : {}", exc)


def run_yfinance_job() -> None:
    """Collecte yfinance sur une fenêtre de 5 jours (absorbe week-ends / fériés)."""
    until = utc_now()
    since = until - timedelta(days=5)
    try:
        YFinanceCollector().run(since, until)
    except Exception as exc:  # pragma: no cover
        logger.exception("Job yfinance : exception non rattrapée : {}", exc)


def build_scheduler() -> BlockingScheduler:
    """Construit et configure le scheduler avec les jobs Phase 1."""
    sched = BlockingScheduler(timezone="Europe/Paris")
    sched.add_job(
        run_arxiv_job,
        trigger=CronTrigger(hour=6, minute=0),
        id="arxiv_daily",
        name="arXiv daily collection",
        replace_existing=True,
    )
    sched.add_job(
        run_yfinance_job,
        trigger=CronTrigger(day_of_week="mon-fri", hour=22, minute=0),
        id="yfinance_weekday",
        name="yfinance weekday collection",
        replace_existing=True,
    )
    return sched


def main() -> None:
    """Point d'entrée CLI : `python -m scheduler.jobs`."""
    configure_logging()
    init_db()
    sched = build_scheduler()
    logger.info(
        "Scheduler démarré. Jobs enregistrés : {}",
        [j.id for j in sched.get_jobs()],
    )
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Arrêt du scheduler demandé.")


if __name__ == "__main__":
    main()
