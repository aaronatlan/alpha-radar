"""Poids experts utilisés par les scoreurs.

Chaque version de poids porte un identifiant stable (`model_version`)
snapshotté dans les métadonnées produites. Toute nouvelle calibration
crée une **nouvelle entrée** — on ne modifie jamais un set de poids
existant, pour garantir la reproductibilité d'un score historique.

Phase 2 : poids manuels. Phase 6 : remplacement progressif par des
poids appris (XGBoost) avec la même convention de versioning.
"""
from __future__ import annotations


# --- Heat Score sectoriel -------------------------------------------------
#
# Inputs possibles (à élargir au fil des sources) :
#   - arxiv_velocity      : vélocité académique
#   - (à venir) github_velocity, sec_form_d_density, news_sentiment_sector…
#
# La somme des poids vaut 1.0. Toute feature manquante fait renvoyer
# `None` à `compute()` — on ne substitue jamais 0 à une donnée absente.

SECTOR_HEAT_WEIGHTS: dict[str, dict[str, float]] = {
    "v1_velocity_only": {"arxiv_velocity": 1.0},
    # Étape 4 : la vélocité GitHub (stars) entre en jeu.
    "v2_arxiv_github": {"arxiv_velocity": 0.6, "github_stars_velocity": 0.4},
}


# --- Score actions --------------------------------------------------------
#
# SPEC §7.3 cible un score 4-dimensions (momentum / signal_quality /
# health / rehabilitation). Phase 2 démarre avec momentum seul puis
# ajoute signal_quality et sentiment. Les dimensions absentes ne
# faussent pas la moyenne : on renormalise par la somme des poids des
# dimensions effectivement disponibles (`compute()` saute une dimension
# dont aucun input n'est calculable).

STOCK_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "v1_momentum_only": {"momentum": 1.0},
    # Étape 4 : signal_quality alimenté par GitHub velocity (secteur de
    # l'action) et densité de filings SEC significatifs (13D, Form D, 8-K).
    "v2_mom_sigqual": {"momentum": 0.6, "signal_quality": 0.4},
    # Étape 5 (sentiment news) :
    "v3_mom_sigqual_sent": {
        "momentum": 0.4, "signal_quality": 0.3, "sentiment": 0.3,
    },
}


# --- Score crypto ---------------------------------------------------------
#
# SPEC §7.3 : 0.30 dev + 0.25 community + 0.25 technical + 0.20 sector.
# On démarre étape 4 avec une version réduite (dev + technical).

CRYPTO_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "v1_dev_tech": {"dev_activity": 0.5, "technical": 0.5},
}
