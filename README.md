# Alpha Radar

Système d'analyse d'investissement multi-sources qui détecte les opportunités
avant que le marché ne les valorise pleinement.

Voir [`SPEC.md`](./SPEC.md) pour le cahier des charges complet (architecture,
schéma DB, plan de développement en 6 phases).

## État du projet

**Phase 5 — Backtesting (en cours).**

- ✅ Phases 1–4 : collecteurs (arXiv, yfinance, GitHub, SEC EDGAR, NewsAPI,
  ClinicalTrials, FDA, USASpending, CoinGecko, SemanticScholar), features
  techniques + sectorielles, scoring composite, thèses + évaluations,
  mémoire/post-mortem, alertes, dashboard 7 pages.
- 🚧 Phase 5 : `PortfolioSimulator`, replay PIT, walk-forward + grid search,
  CLI runner, page Backtest. Frais & slippage paramétrables.
- ⏳ Phase 6 : ML (à partir du mois 6).

## Installation

Pré-requis : Python ≥ 3.11 (testé sur 3.12).

```bash
git clone <repo> alpha-radar && cd alpha-radar
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard]"           # base + dashboard Streamlit
# pip install -e ".[dev,dashboard,sentiment]"  # ajoute FinBERT (≈ 2 GB)
cp .env.example .env
```

Les variables d'environnement utiles sont documentées dans
[`.env.example`](./.env.example). Aucune clé API n'est obligatoire pour
démarrer (arXiv et yfinance sont publics) ; les autres collecteurs
s'activent au fur et à mesure que vous renseignez les tokens.

## Premier démarrage

Bootstrap d'une base avec quelques données réelles, puis affichage du
dashboard :

```bash
python -m scripts.initial_collect    # 1 passe synchrone des collecteurs
python -m scripts.initial_compute    # features → scoring → thèses → alertes
streamlit run dashboard/app.py       # http://localhost:8501
```

Pour explorer le dashboard sans attendre la collecte (données fictives mais
réalistes) :

```bash
python -m scripts.seed_demo
streamlit run dashboard/app.py
```

## Utilisation courante

### Scheduler (production)

```bash
python -m scheduler.jobs
```

Lance APScheduler en mode bloquant. Toutes les cadences (collectes,
features, scoring, thèses, alertes) sont définies dans
[`scheduler/jobs.py`](./scheduler/jobs.py). Une erreur sur un job ne tue
pas le scheduler — le tick suivant retentera.

### Backtest CLI

```bash
# Simulation portefeuille sur les thèses déjà en base
python -m backtesting.runner portfolio \
    --start 2026-01-01 --end 2026-04-01 \
    --benchmark SPY \
    --slippage-bps 5 --fee-bps 0

# Re-jeu PIT (re-score + re-génère les thèses sur la fenêtre, puis simule)
python -m backtesting.runner replay \
    --start 2026-01-01 --end 2026-04-01 --step-days 1

# Walk-forward + grid search sur des jeux de poids
python -m backtesting.runner walk-forward \
    --start 2024-01-01 --end 2026-01-01 \
    --folds 4 --weights v2_mom_sigqual,v3_mom_sigqual_sent
```

Les backtests respectent la discipline point-in-time — aucun close
post-`as_of` n'est lu. Frais & slippage par défaut : 5 bps de slippage par
côté (≈ 10 bps round-trip), 0 bps de commission (retail commission-free).

### Tests

```bash
pytest                       # tous
pytest tests/test_dashboard  # une suite
pytest -k portfolio          # par mot-clé
```

## Dashboard

Pages disponibles dans le menu latéral :

1. **Heat Map** — treemap des Heat Scores sectoriels.
2. **Opportunities** — classement actions par score composite.
3. **Performance** — track record (alpha moyen, taux de succès par jalon).
4. **Alerts** — alertes actives, ack en un clic.
5. **Memory** — historique complet des thèses + export CSV.
6. **Backtest** — simulation portfolio interactive (frais réglables).
7. **Thèse** — vue détaillée d'une thèse (narrative, score décomposé,
   prix avec zone d'entrée + jalons d'évaluation).

URL partageable : la page Thèse accepte `?thesis_id=N` en query string.

## Structure

- `config/` — settings, secteurs suivis, watchlists, calendrier PDUFA
- `collectors/` — un module par source de données, héritant de `BaseCollector`
- `features/` — features techniques + vélocités sectorielles + sentiment
- `scoring/` — Heat Scores sectoriels, scoring composite actions
- `thesis/` — générateur, évaluateur, post-mortem
- `alerts/` — règles + moteur (avec dédup + email optionnel)
- `backtesting/` — `PortfolioSimulator`, replay PIT, walk-forward, runner CLI
- `dashboard/` — Streamlit multi-pages (lecture seule)
- `memory/` — schéma SQLite et accès (point-in-time)
- `scheduler/` — orchestration des jobs périodiques
- `scripts/` — bootstrap, seed démo, inspection de la base
- `tests/` — tests unitaires + intégration pytest
- `data/` — données brutes et caches (gitignore)

## Principes non négociables

1. **Gratuit** — APIs publiques uniquement.
2. **Point-in-time** — chaque donnée porte son timestamp de validité ; aucun
   calcul ne doit utiliser d'information du futur. La double colonne
   `(content_at, fetched_at)` garantit qu'on peut rejouer l'historique tel
   qu'il était observable.
3. **Règles d'abord, ML ensuite** — poids experts en dur pour les 6-12 premiers
   mois, ML seulement après labellisation suffisante.
4. **Interprétable** — chaque score est décomposable en ses contributions.
5. **Mémoire immuable** — les thèses ne sont jamais modifiées, les évaluations
   sont ajoutées en append-only.
6. **Mode dégradé** — si une API tombe, le système continue sur les autres.

## Dépannage

- **Le scheduler tourne mais rien n'apparaît dans le dashboard.** La plupart
  des jobs s'exécutent le soir (22:30 → 23:58 Europe/Paris). Lancez
  `python -m scripts.initial_collect && python -m scripts.initial_compute`
  pour bootstrapper sans attendre.
- **`ModuleNotFoundError: streamlit`** — installer l'extra dashboard :
  `pip install -e ".[dashboard]"`.
- **Sentiment news vide.** FinBERT n'est pas dans les deps par défaut
  (≈ 2 GB) : `pip install -e ".[sentiment]"` puis recalculer.
- **Inspecter la base SQLite.** `python -m scripts.db_state` affiche un
  résumé des lignes par table et de la fraîcheur des features.
