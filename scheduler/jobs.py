"""Scheduler — Phase 2.

Jobs de collecte récurrents et recalcul des features / scores. La base
est initialisée au démarrage. Une erreur dans un job ne tue pas le
scheduler (mode dégradé : le prochain tick réessaiera).

Les fenêtres glissantes créent un chevauchement volontaire d'une
exécution à l'autre ; la déduplication dans `raw_data` garantit
l'idempotence (idem pour les features via UNIQUE
`(feature_name,target_type,target_id,computed_at)`).

Cadences choisies (heure Europe/Paris) :

  Collectes
  ---------
  06:00 quotidien  — arXiv (48 h)
  07:00 quotidien  — CoinGecko (snapshot spot, un par jour UTC)
  07:30 quotidien  — GitHub (snapshot stars / forks, un par jour UTC)
  08:00 quotidien  — SEC EDGAR (fenêtre 7 j)
  09:00 toutes 3h  — NewsAPI (fenêtre 24 h, plusieurs passages pour
                      capter le flux de la journée ; dédup sur URL hash)
  22:00 jours ouvrés — yfinance (5 j)

  Features / scores
  -----------------
  22:30 jours ouvrés — features techniques (RSI, momentum, volume ratio)
                      juste après la clôture US, avant la mise à jour
                      des Heat Scores / stock scores.
  22:45 quotidien    — vélocités sectorielles (arxiv, github stars)
  22:50 quotidien    — sentiment news (FinBERT, log-and-skip si extra absent)
  23:00 quotidien    — Heat Scores sectoriels
  23:15 quotidien    — stock scores & crypto scores (si applicable)
  23:30 quotidien    — génération de thèses (Phase 3)
  23:45 quotidien    — évaluation des thèses aux jalons 30/90/180/365/540j
  23:55 quotidien    — post-mortem (recalcul de signal_performance)
  23:58 quotidien    — moteur d'alertes (nouvelles thèses, verdicts, surges)
"""
from __future__ import annotations

from datetime import timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from collectors.arxiv_collector import ArxivCollector
from collectors.coingecko_collector import CoinGeckoCollector
from collectors.github_collector import GitHubCollector
from collectors.news_collector import NewsAPICollector
from collectors.sec_edgar_collector import SECEdgarCollector
from collectors.yfinance_collector import YFinanceCollector
from config.settings import configure_logging
from features.sentiment import NewsSentimentSectorFeature
from features.technical import (
    Momentum30DFeature,
    RSI14Feature,
    VolumeRatio7_30Feature,
)
from features.velocity import ArxivVelocityFeature, GitHubStarsVelocityFeature
from memory.database import init_db, utc_now
from scoring.sector_heat import SectorHeatScorer
from scoring.stock_scorer import StockScorer
from alerts.engine import AlertsEngine
from thesis.evaluator import ThesisEvaluator
from thesis.generator import ThesisGenerator
from thesis.post_mortem import PostMortemAnalyzer


# --------------------------------------------------------------- collectes


def _run_collector(label: str, build, days: int) -> None:
    """Exécute `build().run(since, until)` avec garde-fou log-only."""
    until = utc_now()
    since = until - timedelta(days=days)
    try:
        build().run(since, until)
    except Exception as exc:  # pragma: no cover — BaseCollector.run avale déjà
        logger.exception("Job {} : exception non rattrapée : {}", label, exc)


def run_arxiv_job() -> None:
    _run_collector("arxiv", ArxivCollector, days=2)


def run_yfinance_job() -> None:
    _run_collector("yfinance", YFinanceCollector, days=5)


def run_coingecko_job() -> None:
    # L'endpoint /coins/markets ne renvoie que le spot, `since/until`
    # n'influent pas sur la collecte — passer 1 j suffit.
    _run_collector("coingecko", CoinGeckoCollector, days=1)


def run_github_job() -> None:
    # Snapshot : `since/until` non utilisés côté API, juste conventionnels.
    _run_collector("github", GitHubCollector, days=1)


def run_sec_edgar_job() -> None:
    # Les filings apparaissent avec un délai ; fenêtre de 7 j avec
    # chevauchement couvre les rattrapages.
    _run_collector("sec_edgar", SECEdgarCollector, days=7)


def run_newsapi_job() -> None:
    # Fenêtre de 24 h avec chevauchement (toutes les 3 h).
    _run_collector("newsapi", NewsAPICollector, days=1)


# ------------------------------------------------------------- features / scores


def _run_feature(label: str, build) -> None:
    try:
        build().run()
    except Exception as exc:  # pragma: no cover
        logger.exception("Job {} : exception non rattrapée : {}", label, exc)


def run_technical_features_job() -> None:
    _run_feature("rsi_14", RSI14Feature)
    _run_feature("momentum_30d", Momentum30DFeature)
    _run_feature("volume_ratio_7_30", VolumeRatio7_30Feature)


def run_velocity_features_job() -> None:
    _run_feature("arxiv_velocity", ArxivVelocityFeature)
    _run_feature("github_stars_velocity", GitHubStarsVelocityFeature)


def run_sentiment_features_job() -> None:
    # Le modèle FinBERT est lourd : on évite d'arrêter le scheduler si
    # l'extra `sentiment` n'est pas installé — run() log-and-skip.
    _run_feature("news_sentiment_sector", NewsSentimentSectorFeature)


def run_sector_heat_job() -> None:
    _run_feature("sector_heat_score", SectorHeatScorer)


def run_stock_scores_job() -> None:
    _run_feature("stock_score", StockScorer)


def run_thesis_generator_job() -> None:
    """Génère les thèses (Phase 3) à partir des scores du jour."""
    try:
        ThesisGenerator().run()
    except Exception as exc:  # pragma: no cover
        logger.exception("Job thesis_generator : exception non rattrapée : {}", exc)


def run_thesis_evaluator_job() -> None:
    """Évalue les thèses dues à leurs jalons (Phase 3 étape 2)."""
    try:
        ThesisEvaluator().run()
    except Exception as exc:  # pragma: no cover
        logger.exception("Job thesis_evaluator : exception non rattrapée : {}", exc)


def run_post_mortem_job() -> None:
    """Recalcule signal_performance depuis les évaluations (Phase 3 étape 3)."""
    try:
        PostMortemAnalyzer().run()
    except Exception as exc:  # pragma: no cover
        logger.exception("Job post_mortem : exception non rattrapée : {}", exc)


def run_alerts_engine_job() -> None:
    """Évalue les règles d'alerte et notifie par email (Phase 3 étape 4)."""
    try:
        AlertsEngine().run()
    except Exception as exc:  # pragma: no cover
        logger.exception("Job alerts_engine : exception non rattrapée : {}", exc)


# ---------------------------------------------------------------- scheduler


def build_scheduler() -> BlockingScheduler:
    """Construit et configure le scheduler avec les jobs Phase 2."""
    sched = BlockingScheduler(timezone="Europe/Paris")

    # Collectes
    sched.add_job(
        run_arxiv_job,
        trigger=CronTrigger(hour=6, minute=0),
        id="arxiv_daily",
        name="arXiv daily collection",
        replace_existing=True,
    )
    sched.add_job(
        run_coingecko_job,
        trigger=CronTrigger(hour=7, minute=0),
        id="coingecko_daily",
        name="CoinGecko daily snapshot",
        replace_existing=True,
    )
    sched.add_job(
        run_github_job,
        trigger=CronTrigger(hour=7, minute=30),
        id="github_daily",
        name="GitHub repos daily snapshot",
        replace_existing=True,
    )
    sched.add_job(
        run_sec_edgar_job,
        trigger=CronTrigger(hour=8, minute=0),
        id="sec_edgar_daily",
        name="SEC EDGAR daily filings",
        replace_existing=True,
    )
    sched.add_job(
        run_newsapi_job,
        trigger=CronTrigger(hour="9,12,15,18,21", minute=0),
        id="newsapi_3h",
        name="NewsAPI every 3h",
        replace_existing=True,
    )
    sched.add_job(
        run_yfinance_job,
        trigger=CronTrigger(day_of_week="mon-fri", hour=22, minute=0),
        id="yfinance_weekday",
        name="yfinance weekday collection",
        replace_existing=True,
    )

    # Features / scores (séquentiel, après les collectes du jour)
    sched.add_job(
        run_technical_features_job,
        trigger=CronTrigger(day_of_week="mon-fri", hour=22, minute=30),
        id="technical_features",
        name="Technical features (RSI, momentum, volume ratio)",
        replace_existing=True,
    )
    sched.add_job(
        run_velocity_features_job,
        trigger=CronTrigger(hour=22, minute=45),
        id="velocity_features",
        name="Sector velocity features",
        replace_existing=True,
    )
    sched.add_job(
        run_sentiment_features_job,
        trigger=CronTrigger(hour=22, minute=50),
        id="sentiment_features",
        name="News sentiment features (FinBERT)",
        replace_existing=True,
    )
    sched.add_job(
        run_sector_heat_job,
        trigger=CronTrigger(hour=23, minute=0),
        id="sector_heat_scores",
        name="Sector Heat Scores",
        replace_existing=True,
    )
    sched.add_job(
        run_stock_scores_job,
        trigger=CronTrigger(hour=23, minute=15),
        id="stock_scores",
        name="Stock composite scores",
        replace_existing=True,
    )
    sched.add_job(
        run_thesis_generator_job,
        trigger=CronTrigger(hour=23, minute=30),
        id="thesis_generator",
        name="Thesis generator",
        replace_existing=True,
    )
    sched.add_job(
        run_thesis_evaluator_job,
        trigger=CronTrigger(hour=23, minute=45),
        id="thesis_evaluator",
        name="Thesis evaluator (milestones)",
        replace_existing=True,
    )
    sched.add_job(
        run_post_mortem_job,
        trigger=CronTrigger(hour=23, minute=55),
        id="post_mortem",
        name="Post-mortem signal_performance",
        replace_existing=True,
    )
    sched.add_job(
        run_alerts_engine_job,
        trigger=CronTrigger(hour=23, minute=58),
        id="alerts_engine",
        name="Alerts engine (rules + email)",
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
