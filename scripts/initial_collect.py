"""Première collecte sync : pour bootstrapper la DB tout de suite.

Sans ça, il faut attendre les cron triggers (06:00 → 23:58) pour voir
quelque chose en base. Ce script déclenche **une seule passe** de
chaque collecteur sur sa fenêtre par défaut.

Idempotent : re-exécutable sans risque (UNIQUE constraints en base).
À lancer **en plus** du scheduler (qui prend le relais ensuite).

Usage : `python -m scripts.initial_collect`
"""
from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger

from collectors.arxiv_collector import ArxivCollector
from collectors.clinicaltrials_collector import ClinicalTrialsCollector
from collectors.coingecko_collector import CoinGeckoCollector
from collectors.fda_collector import FDACollector
from collectors.github_collector import GitHubCollector
from collectors.news_collector import NewsAPICollector
from collectors.sec_edgar_collector import SECEdgarCollector
from collectors.semantic_scholar_collector import SemanticScholarCollector
from collectors.usaspending_collector import USASpendingCollector
from collectors.yfinance_collector import YFinanceCollector
from config.settings import configure_logging
from memory.database import init_db, utc_now


COLLECTORS: list[tuple[str, type, int]] = [
    # (label, classe, days fenêtre)
    ("arxiv",          ArxivCollector,             2),
    ("yfinance",       YFinanceCollector,          7),
    ("coingecko",      CoinGeckoCollector,         1),
    ("github",         GitHubCollector,            1),
    ("sec_edgar",      SECEdgarCollector,          7),
    ("newsapi",        NewsAPICollector,           1),
    ("clinicaltrials", ClinicalTrialsCollector,   30),
    ("fda",            FDACollector,              90),
    ("usaspending",    USASpendingCollector,      30),
    ("semantic_scholar", SemanticScholarCollector, 1),
]


def main() -> None:
    configure_logging()
    init_db()
    until = utc_now()
    total = 0
    failures = 0
    for label, klass, days in COLLECTORS:
        since = until - timedelta(days=days)
        logger.info("================ {} ================", label)
        try:
            n = klass().run(since, until)
            total += n or 0
            logger.info(">>> {} : {} insérés", label, n)
        except Exception as exc:
            failures += 1
            logger.exception(">>> {} : ÉCHEC ({})", label, exc)
    logger.info("[initial_collect] TOTAL inséré : {} (échecs : {})", total, failures)


if __name__ == "__main__":
    main()
