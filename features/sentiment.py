"""Sentiment de news basé sur FinBERT.

Lit les articles de presse collectés par `NewsAPICollector` et calcule un
score de sentiment agrégé par secteur, point-in-time. Modèle par défaut :
[`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) — BERT
fine-tuné sur des titres financiers (positive / negative / neutral).

Convention de valeur
--------------------
Sentiment par article : `p_positive − p_negative` ∈ [−1, +1]. Un neutre
strict vaut 0, donc les textes "financial-ement neutres" (pouvant dominer
un flux de news) ne biaisent pas la moyenne. On stocke la **moyenne
pondérée par la fraîcheur** (plus un article est récent, plus il pèse) :

    w(t) = 0.5^(Δjours / half_life_days)

La feature est stockée telle quelle dans [−1, +1]. Le scoreur d'actions
appliquera ensuite un mapping linéaire vers [0, 100].

Dépendances optionnelles
------------------------
`transformers` et `torch` ne sont **pas** dans les dépendances de base —
ils sont dans l'extra `[sentiment]`. Si le module n'est pas installé,
`FinBERTSentimentAnalyzer` lève `RuntimeError` à l'instanciation, mais la
feature se contente de logger et de retourner `None` (mode dégradé : on
n'écrit rien en base, on ne casse pas la pipeline).

Point-in-time
-------------
Double filtre sur `raw_data` :
  - `content_at` dans la fenêtre récente (`window_days`, défaut 7)
  - `fetched_at <= as_of`

L'inférence FinBERT est purement fonctionnelle (pas d'état, pas d'I/O).
Son résultat à `as_of` ne dépend donc que des articles matériellement
collectés avant ou à `as_of`, ce qui préserve la discipline PIT.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any, Iterable

from loguru import logger
from sqlalchemy import select

from config.sectors import SECTORS, SECTORS_BY_ID
from features.base import BaseFeature
from memory.database import RawData, session_scope


# -------------------------------------------------------------- analyzer


class FinBERTSentimentAnalyzer:
    """Wrapper autour du pipeline HuggingFace FinBERT.

    Chargement paresseux : le modèle est téléchargé/instancié à la
    première inférence. Utilise le pipeline HF `text-classification`
    avec `top_k=None` pour récupérer les trois scores (positive,
    negative, neutral), qu'on réduit à un scalaire `p_pos − p_neg`.

    En tests, on injecte `classifier=<callable>` pour éviter le
    téléchargement du modèle.
    """

    model_name = "ProsusAI/finbert"
    max_chars_per_text = 512

    def __init__(self, classifier: Any | None = None) -> None:
        self._classifier = classifier

    def _ensure_classifier(self) -> None:
        if self._classifier is not None:
            return
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — dépend de l'env
            raise RuntimeError(
                "FinBERT requiert l'extra `sentiment` : "
                "`pip install -e .[sentiment]`"
            ) from exc
        logger.info("[finbert] Chargement du modèle {} (premier appel)…", self.model_name)
        self._classifier = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,
            truncation=True,
        )

    def score_texts(self, texts: Iterable[str]) -> list[float]:
        """Retourne une liste de scores ∈ [−1, +1], un par texte.

        Textes vides ou None → 0.0 (traité comme neutre).
        """
        texts_list = [
            (t[: self.max_chars_per_text] if isinstance(t, str) and t.strip() else "")
            for t in texts
        ]
        # Inutile de charger le modèle si aucun texte non-vide.
        if not any(texts_list):
            return [0.0] * len(texts_list)

        self._ensure_classifier()
        assert self._classifier is not None

        # On passe seulement les textes non-vides au pipeline (plus rapide).
        nonempty_idx = [i for i, t in enumerate(texts_list) if t]
        nonempty = [texts_list[i] for i in nonempty_idx]
        raw_out = self._classifier(nonempty)

        scores = [0.0] * len(texts_list)
        for idx, result in zip(nonempty_idx, raw_out):
            # `result` est une liste de dicts {label, score} quand top_k=None.
            p_pos = 0.0
            p_neg = 0.0
            for entry in result:
                label = str(entry.get("label", "")).lower()
                if label.startswith("pos"):
                    p_pos = float(entry.get("score", 0.0))
                elif label.startswith("neg"):
                    p_neg = float(entry.get("score", 0.0))
            scores[idx] = p_pos - p_neg
        return scores


# --------------------------------------------------------------- feature


class NewsSentimentSectorFeature(BaseFeature):
    """Sentiment FinBERT moyen par secteur sur une fenêtre récente.

    Agrège titre + description de chaque article (tronqués). Pondération
    temporelle exponentielle (demi-vie `half_life_days`) : un article
    d'il y a un jour pèse plus qu'un d'il y a une semaine.
    """

    feature_name = "news_sentiment_sector"
    target_type = "sector"

    #: Fenêtre d'agrégation en jours.
    window_days = 7
    #: Demi-vie de la pondération temporelle (jours).
    half_life_days = 3.0
    #: Seuil minimum d'articles pour publier une valeur (évite le bruit).
    min_articles = 3

    def __init__(
        self,
        sector_ids: list[str] | None = None,
        analyzer: FinBERTSentimentAnalyzer | None = None,
    ) -> None:
        super().__init__()
        if sector_ids is not None:
            unknown = set(sector_ids) - set(SECTORS_BY_ID)
            if unknown:
                raise ValueError(f"Secteurs inconnus : {sorted(unknown)}")
            self._sector_ids = list(sector_ids)
        else:
            self._sector_ids = [s["id"] for s in SECTORS]
        self._analyzer = analyzer  # lazily built si None

    def targets(self) -> list[str]:
        return list(self._sector_ids)

    def compute(
        self, target_id: str, as_of: datetime
    ) -> tuple[float, dict[str, Any]] | None:
        articles = _load_sector_articles(
            sector_id=target_id,
            start=as_of - timedelta(days=self.window_days),
            end=as_of,
            as_of=as_of,
        )
        if len(articles) < self.min_articles:
            return None

        texts = [
            " ".join(filter(None, [a.get("title"), a.get("description")]))
            for a in articles
        ]

        try:
            analyzer = self._analyzer or FinBERTSentimentAnalyzer()
            scores = analyzer.score_texts(texts)
        except RuntimeError as exc:
            # FinBERT indisponible (extra non installé) → mode dégradé.
            logger.warning("[news_sentiment] {}", exc)
            return None

        weights: list[float] = []
        for a in articles:
            dt = a.get("content_at")
            if isinstance(dt, datetime):
                delta_days = max(0.0, (as_of - dt).total_seconds() / 86400.0)
            else:
                delta_days = self.window_days  # fallback : poids plancher
            weights.append(0.5 ** (delta_days / self.half_life_days))

        total_w = sum(weights)
        if total_w <= 0.0 or math.isnan(total_w):
            return None
        value = sum(s * w for s, w in zip(scores, weights)) / total_w

        metadata = {
            "n_articles": len(articles),
            "window_days": self.window_days,
            "half_life_days": self.half_life_days,
            "mean_raw_sentiment": (sum(scores) / len(scores)) if scores else 0.0,
        }
        return value, metadata


# ---------------------------------------------------------------- helpers


def _load_sector_articles(
    sector_id: str, start: datetime, end: datetime, as_of: datetime
) -> list[dict[str, Any]]:
    """Lit les articles `newsapi` tagués `sector_id` dans [start, end).

    Filtre PIT : `content_at ∈ [start, end)` et `fetched_at <= as_of`.
    Les payloads mal formés sont ignorés.
    """
    stmt = (
        select(RawData.content_at, RawData.payload_json)
        .where(RawData.source == "newsapi")
        .where(RawData.entity_type == "news_article")
        .where(RawData.content_at >= start)
        .where(RawData.content_at < end)
        .where(RawData.fetched_at <= as_of)
    )
    articles: list[dict[str, Any]] = []
    with session_scope() as session:
        for content_at, payload_json in session.execute(stmt):
            try:
                payload = json.loads(payload_json)
            except (TypeError, ValueError):
                continue
            if payload.get("sector_id") != sector_id:
                continue
            articles.append(
                {
                    "title": payload.get("title"),
                    "description": payload.get("description"),
                    "url": payload.get("url"),
                    "content_at": content_at,
                }
            )
    return articles
