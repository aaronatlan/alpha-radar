"""Première passe sync : features → scoring → thèses → évaluations → alertes.

À lancer juste après `initial_collect.py` pour que le pipeline aval ait
de quoi mâcher dès maintenant — sinon il faut attendre les cron triggers
(22:30+) pour voir scoring et thèses se matérialiser.

Usage : `python -m scripts.initial_compute`
"""
from __future__ import annotations

from loguru import logger

from alerts.engine import AlertsEngine
from config.settings import configure_logging
from features.sentiment import NewsSentimentSectorFeature
from features.technical import (
    Momentum30DFeature,
    RSI14Feature,
    VolumeRatio7_30Feature,
)
from features.velocity import ArxivVelocityFeature, GitHubStarsVelocityFeature
from memory.database import init_db
from scoring.sector_heat import SectorHeatScorer
from scoring.stock_scorer import StockScorer
from thesis.evaluator import ThesisEvaluator
from thesis.generator import ThesisGenerator
from thesis.post_mortem import PostMortemAnalyzer


PIPELINE: list[tuple[str, type]] = [
    ("rsi_14",                RSI14Feature),
    ("momentum_30d",          Momentum30DFeature),
    ("volume_ratio_7_30",     VolumeRatio7_30Feature),
    ("arxiv_velocity",        ArxivVelocityFeature),
    ("github_stars_velocity", GitHubStarsVelocityFeature),
    ("sector_heat_score",     SectorHeatScorer),
    ("stock_score",           StockScorer),
]


def main() -> None:
    configure_logging()
    init_db()

    for label, klass in PIPELINE:
        logger.info("================ {} ================", label)
        try:
            klass().run()
        except Exception as exc:
            logger.exception(">>> {} : ÉCHEC ({})", label, exc)

    # FinBERT séparément — coûteux, l'extra `[sentiment]` peut manquer.
    logger.info("================ news_sentiment_sector ================")
    try:
        NewsSentimentSectorFeature().run()
    except Exception as exc:
        logger.warning(">>> news_sentiment_sector : skip ({})", exc)

    logger.info("================ thesis_generator ================")
    ThesisGenerator().run()

    logger.info("================ thesis_evaluator ================")
    ThesisEvaluator().run()

    logger.info("================ post_mortem ================")
    PostMortemAnalyzer().run()

    logger.info("================ alerts_engine ================")
    AlertsEngine().run()

    logger.info("[initial_compute] Pipeline complet — terminé.")


if __name__ == "__main__":
    main()
