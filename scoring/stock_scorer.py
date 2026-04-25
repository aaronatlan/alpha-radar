"""Scoreur composite d'actions.

Lit les features techniques PIT (`rsi_14`, `momentum_30d`,
`volume_ratio_7_30`) pour chaque ticker de la watchlist, les convertit
en sous-scores 0-100 ("dimensions" au sens SPEC §7.3), puis combine
selon les poids versionnés de `scoring.weights.STOCK_SCORE_WEIGHTS`.

Phase 2 v1 : une seule dimension `momentum` — le score composite est
donc égal au sous-score momentum. Quand signal_quality et sentiment
seront disponibles (étapes 4-5), on passera à `v2_...` ou `v3_...`
sans toucher au code de ce scoreur, juste en changeant les poids et
en ajoutant les `_compute_dimension_*` correspondants.

Normalisation des sous-scores
-----------------------------
Chaque dimension est bornée [0, 100] pour rester additive. Les
mappings sont volontairement simples (linéaires par morceaux) — ils
seront remplacés par des sigmoïdes calibrées en Phase 6.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from config.watchlists import STOCK_WATCHLIST, WATCHLIST_TICKERS
from features.base import BaseFeature
from memory.database import Feature, RawData, session_scope
from scoring._io import latest_feature_value
from scoring.weights import STOCK_SCORE_WEIGHTS


def _sentiment_to_score(s: float) -> float:
    """Sentiment ∈ [−1, +1] → sous-score ∈ [0, 100]. Linéaire, 0 → 50."""
    clipped = max(-1.0, min(1.0, s))
    return 50.0 * (1.0 + clipped)


# Formes SEC qui portent un signal fort ("smart money" ou levée) — elles
# boostent signal_quality quand elles apparaissent récemment.
_SIGNAL_FORMS = {"SC 13D", "SC 13D/A", "D", "8-K"}
_SEC_WINDOW_DAYS = 30


# ------------------------------------------------------- sub-score mappings


def _rsi_to_score(rsi: float) -> float:
    """RSI 14j ∈ [0, 100] → sous-score ∈ [0, 100].

    On pénalise la sur-vente (RSI < 30) et la sur-achat (RSI > 70),
    avec un optimum plateau autour de 50-65 (momentum sain, pas
    d'excès). Triangulaire simple : pic à 60, nul à 0 et à 100.
    """
    rsi = max(0.0, min(100.0, rsi))
    peak = 60.0
    if rsi <= peak:
        return 100.0 * rsi / peak
    return 100.0 * (100.0 - rsi) / (100.0 - peak)


def _momentum_to_score(m30: float) -> float:
    """Rendement 30j → [0, 100]. +20% saturé à 100, -20% à 0, 0 → 50."""
    clipped = max(-0.20, min(0.20, m30))
    return 50.0 + 50.0 * (clipped / 0.20)


def _volume_to_score(vr: float) -> float:
    """Ratio volume 7/30 → [0, 100]. 1.0 → 50, 2.0 → 100, 0 → 0."""
    clipped = max(0.0, min(2.0, vr))
    return 50.0 * clipped


# ------------------------------------------------------------- dimensions


def _compute_momentum_dimension(
    ticker: str, as_of: datetime
) -> tuple[float, dict[str, Any]] | None:
    """Agrège RSI, momentum 30j et volume ratio en un sous-score momentum.

    Moyenne pondérée fixe (intra-dimension) : 0.35 RSI + 0.45 momentum
    + 0.20 volume. Si aucune des trois features n'est disponible,
    retourne `None` (pas de fabrication ex nihilo). Si certaines sont
    disponibles, on renormalise sur les présentes.
    """
    rsi = latest_feature_value("rsi_14", "asset", ticker, as_of)
    mom = latest_feature_value("momentum_30d", "asset", ticker, as_of)
    vol = latest_feature_value("volume_ratio_7_30", "asset", ticker, as_of)

    raw_weights = {"rsi": 0.35, "momentum_30d": 0.45, "volume_ratio": 0.20}
    parts: dict[str, float] = {}
    if rsi is not None:
        parts["rsi"] = _rsi_to_score(rsi)
    if mom is not None:
        parts["momentum_30d"] = _momentum_to_score(mom)
    if vol is not None:
        parts["volume_ratio"] = _volume_to_score(vol)

    if not parts:
        return None

    total_w = sum(raw_weights[k] for k in parts)
    score = sum(raw_weights[k] * parts[k] for k in parts) / total_w

    inputs = {
        "rsi_14": rsi,
        "momentum_30d": mom,
        "volume_ratio_7_30": vol,
    }
    return score, {"subscores": parts, "inputs": inputs}


# --------------------------------------- signal_quality dimension


_SECTORS_BY_TICKER = {item["ticker"]: item["sectors"] for item in STOCK_WATCHLIST}


def _compute_signal_quality_dimension(
    ticker: str, as_of: datetime
) -> tuple[float, dict[str, Any]] | None:
    """Sous-score signal_quality : qualité des signaux externes autour du titre.

    Deux composantes pour l'instant :
      - Heat Score sectoriel moyen (attention de la communauté / marché
        sur les secteurs de l'action),
      - Densité de filings SEC "significatifs" (13D / Form D / 8-K) sur
        les 30 derniers jours, cappée à 3 filings → 100.

    Les deux sont bornées [0, 100] et moyennées à 60/40.
    """
    sectors = _SECTORS_BY_TICKER.get(ticker, [])

    # Heat Score moyen sur les secteurs de l'action
    heat_values: list[float] = []
    for sid in sectors:
        v = latest_feature_value("sector_heat_score", "sector", sid, as_of)
        if v is not None:
            heat_values.append(v)
    sector_heat = (sum(heat_values) / len(heat_values)) if heat_values else None

    # Filings SEC dans les 30 derniers jours (PIT). On distingue :
    #   - `has_sec_data` : au moins un filing observé pour ce ticker
    #     (même hors formes d'intérêt) → on a bien collecté, le score
    #     est calculable ;
    #   - `n_signal_filings` : décompte restreint aux formes "smart money".
    # Sans aucune observation, on traite le sec_score comme absent (pas
    # 0), pour ne pas faire chuter signal_quality quand sec_edgar n'a
    # simplement pas encore été collecté.
    start = as_of - timedelta(days=_SEC_WINDOW_DAYS)
    has_sec_data = False
    n_signal_filings = 0
    with session_scope() as session:
        stmt = (
            select(RawData.payload_json)
            .where(RawData.source == "sec_edgar")
            .where(RawData.entity_type == "sec_filing")
            .where(RawData.content_at >= start)
            .where(RawData.content_at < as_of)
            .where(RawData.fetched_at <= as_of)
        )
        import json as _json
        for (payload_json,) in session.execute(stmt):
            try:
                payload = _json.loads(payload_json)
            except (TypeError, ValueError):
                continue
            if payload.get("ticker") != ticker:
                continue
            has_sec_data = True
            if payload.get("form") in _SIGNAL_FORMS:
                n_signal_filings += 1
    sec_score = (
        min(100.0, (n_signal_filings / 3.0) * 100.0)
        if has_sec_data else None
    )

    # Aucune composante exploitable → signal_quality absent (pas 0).
    if sector_heat is None and sec_score is None:
        return None

    parts: dict[str, float] = {}
    raw_weights = {"sector_heat": 0.6, "sec_filings": 0.4}
    if sector_heat is not None:
        parts["sector_heat"] = sector_heat
    if sec_score is not None:
        parts["sec_filings"] = sec_score

    total_w = sum(raw_weights[k] for k in parts)
    score = sum(raw_weights[k] * parts[k] for k in parts) / total_w

    return score, {
        "subscores": parts,
        "inputs": {
            "n_signal_filings_30d": n_signal_filings,
            "has_sec_data": has_sec_data,
            "sector_heat_mean": sector_heat,
        },
    }


# ------------------------------------------ sentiment dimension


def _compute_sentiment_dimension(
    ticker: str, as_of: datetime
) -> tuple[float, dict[str, Any]] | None:
    """Sous-score sentiment : moyenne du sentiment news des secteurs de l'action.

    On réutilise la feature PIT `news_sentiment_sector` (déjà pondérée
    par la fraîcheur, bornée [−1, +1]). Si aucun secteur de l'action
    n'a de sentiment publié (< min_articles dans la fenêtre), la
    dimension est absente — pas 50 par défaut.
    """
    sectors = _SECTORS_BY_TICKER.get(ticker, [])
    if not sectors:
        return None

    values: list[float] = []
    for sid in sectors:
        v = latest_feature_value("news_sentiment_sector", "sector", sid, as_of)
        if v is not None:
            values.append(v)
    if not values:
        return None

    raw = sum(values) / len(values)
    score = _sentiment_to_score(raw)
    return score, {
        "inputs": {
            "n_sectors_with_sentiment": len(values),
            "mean_sentiment_raw": raw,
            "sectors": sectors,
        }
    }


# --------------------------------------------------------------- scorer


class StockScorer(BaseFeature):
    """Score composite 0-100 pour chaque action de la watchlist."""

    feature_name = "stock_score"
    target_type = "asset"
    model_version = "v2_mom_sigqual"

    def __init__(
        self,
        tickers: list[str] | None = None,
        model_version: str | None = None,
    ) -> None:
        super().__init__()
        self._tickers = list(tickers) if tickers else list(WATCHLIST_TICKERS)
        if model_version is not None:
            self.model_version = model_version
        self._weights = STOCK_SCORE_WEIGHTS[self.model_version]

    def targets(self) -> list[str]:
        return list(self._tickers)

    def compute(
        self, target_id: str, as_of: datetime
    ) -> tuple[float, dict[str, Any]] | None:
        dimensions: dict[str, tuple[float, dict[str, Any]]] = {}

        if "momentum" in self._weights:
            mom = _compute_momentum_dimension(target_id, as_of)
            if mom is not None:
                dimensions["momentum"] = mom

        if "signal_quality" in self._weights:
            sq = _compute_signal_quality_dimension(target_id, as_of)
            if sq is not None:
                dimensions["signal_quality"] = sq

        if "sentiment" in self._weights:
            sent = _compute_sentiment_dimension(target_id, as_of)
            if sent is not None:
                dimensions["sentiment"] = sent

        if not dimensions:
            return None

        # Renormalisation sur les dimensions effectivement calculées.
        total_w = sum(self._weights[dim] for dim in dimensions)
        score = sum(
            self._weights[dim] * value for dim, (value, _) in dimensions.items()
        ) / total_w

        metadata = {
            "model_version": self.model_version,
            "weights": dict(self._weights),
            "dimensions": {dim: value for dim, (value, _) in dimensions.items()},
            "details": {dim: detail for dim, (_, detail) in dimensions.items()},
        }
        return score, metadata
