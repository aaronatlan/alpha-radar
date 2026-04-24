# Alpha Radar — Cahier des Charges Technique

**Version** 1.0
**Auteur** Aaron
**Date** Avril 2026

---

## 1. Vision

Alpha Radar est un système d'analyse d'investissement multi-sources qui détecte les opportunités avant que le marché ne les valorise pleinement. Le système couvre trois classes d'actifs — actions cotées, startups non cotées, crypto-actifs — et combine des signaux hétérogènes (recherche académique, brevets, offres d'emploi, activité open source, contrats gouvernementaux, données financières) pour générer des thèses d'investissement scorées et horodatées.

Le système est conçu pour apprendre de ses propres prédictions. Chaque thèse est archivée avec tous ses paramètres, évaluée périodiquement contre la réalité, et alimente un module de machine learning qui recalibre progressivement les poids des signaux selon leur précision historique.

L'usage cible est personnel dans un premier temps, avec un horizon de professionnalisation possible à 2-3 ans.

---

## 2. Principes de Conception

**Gratuit par construction** Toutes les sources de données utilisées sont des APIs publiques ou des données scrapables gratuitement. Aucune dépendance à Bloomberg, PitchBook ou équivalent payant.

**Point-in-time rigoureux** Chaque donnée stockée est associée au timestamp où elle était effectivement disponible. Aucune feature ne peut utiliser une information du futur pour prédire le passé. Cette discipline est non négociable — c'est le principal risque technique d'un système ML financier.

**Règles d'abord, ML ensuite** Le système démarre avec des poids experts définis manuellement. Après six à douze mois de données labélisées, un modèle ML remplace progressivement les règles. Jamais l'inverse.

**Modulaire** Chaque source de données est un module indépendant avec une interface standardisée. Ajouter une nouvelle source ne doit pas impliquer de refactoring du cœur du système.

**Interprétable** Chaque score généré doit être décomposable en ses contributions individuelles. Un score de 78 doit pouvoir être expliqué par une ventilation claire entre ses signaux composants. Pas de black box.

**Mémoire centrale** Toute prédiction est immuable une fois créée. Les évaluations sont ajoutées dans des tables séparées. L'historique complet est toujours reconstructible.

---

## 3. Architecture Globale

### 3.1 Vue en trois niveaux

```
┌─────────────────────────────────────────────────────┐
│         NIVEAU 1 — RADAR SECTORIEL                  │
│         Quels secteurs chauffent en ce moment       │
│         → Heat Score par secteur (0-100)            │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│         NIVEAU 2 — SCANNER D'OPPORTUNITÉS           │
│         Qui bénéficie dans les secteurs chauds      │
│         → Liste A (cotés directs)                   │
│         → Liste B (proxies cotés)                   │
│         → Liste C (startups à surveiller)           │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│         NIVEAU 3 — THÈSE D'INVESTISSEMENT           │
│         Pourquoi maintenant, risques, catalyseurs   │
│         → Score composite par actif                 │
│         → Thèse narrative structurée                │
│         → Zone d'entrée suggérée                    │
└─────────────────────────────────────────────────────┘
```

### 3.2 Flux de données

```
Sources externes
     │
     ▼
Collecteurs (un par source)
     │
     ▼
Stockage brut point-in-time (raw_data)
     │
     ▼
Calcul des features (features)
     │
     ▼
Agrégation sectorielle (Heat Scores)
     │
     ▼
Scoring des actifs (thèses)
     │
     ▼
Mémoire (archivage immuable)
     │
     ▼
Dashboard + Alertes
     │
     ▼
Évaluation périodique → Recalibration ML
```

---

## 4. Stack Technique

### 4.1 Langage et runtime

Python 3.11+ exclusivement pour le backend. C'est le langage standard du ML, toutes les librairies nécessaires sont disponibles, et la stabilité est bonne.

### 4.2 Librairies principales

**Collecte de données**
- `requests` pour les APIs REST
- `feedparser` pour les flux RSS
- `beautifulsoup4` pour le scraping HTML
- `yfinance` pour les données financières
- `arxiv` (client officiel) pour arXiv
- `PyGithub` pour l'API GitHub
- `sec-edgar-downloader` pour les filings SEC

**Traitement et ML**
- `pandas`, `numpy` pour la manipulation
- `scikit-learn` pour les modèles classiques
- `xgboost` pour le scoring principal
- `transformers` et `torch` pour FinBERT (sentiment)
- `prophet` pour les séries temporelles

**Stockage**
- `SQLite` pour la persistance (suffisant en phase personnelle, migration PostgreSQL possible)
- `sqlalchemy` comme ORM

**Interface**
- `streamlit` pour le dashboard
- `plotly` pour les visualisations interactives

**Orchestration**
- `apscheduler` pour les tâches périodiques
- `loguru` pour le logging

**Tests et qualité**
- `pytest` pour les tests unitaires
- `black` et `ruff` pour le formatting et le linting

### 4.3 Structure du projet

```
alpha-radar/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── config/
│   ├── settings.py              Configuration centrale
│   ├── sectors.py               Définition des secteurs suivis
│   └── watchlists.py            Actifs suivis par défaut
│
├── collectors/
│   ├── base.py                  Classe abstraite Collector
│   ├── arxiv_collector.py
│   ├── semantic_scholar_collector.py
│   ├── pubmed_collector.py
│   ├── github_collector.py
│   ├── sec_edgar_collector.py
│   ├── clinical_trials_collector.py
│   ├── fda_collector.py
│   ├── usa_spending_collector.py
│   ├── patent_collector.py
│   ├── jobs_collector.py
│   ├── news_collector.py
│   ├── yfinance_collector.py
│   └── coingecko_collector.py
│
├── features/
│   ├── base.py                  Classe abstraite Feature
│   ├── velocity.py              Vélocité des signaux temporels
│   ├── technical.py             RSI, MACD, volume
│   ├── fundamental.py           Ratios financiers
│   ├── sentiment.py             NLP avec FinBERT
│   └── aggregators.py           Agrégation par secteur
│
├── scoring/
│   ├── sector_heat.py           Heat Score sectoriel
│   ├── stock_scorer.py          Score des actions
│   ├── crypto_scorer.py         Score crypto
│   ├── startup_scorer.py        Score startups (via proxies)
│   └── weights.py               Poids (manuels puis ML)
│
├── models/
│   ├── base.py                  Classe abstraite Model
│   ├── xgboost_scorer.py        Modèle principal
│   ├── sentiment_model.py       Wrapper FinBERT
│   ├── timeseries.py            Prophet pour vélocités
│   └── training.py              Pipeline d'entraînement
│
├── memory/
│   ├── database.py              Schéma SQLite
│   ├── theses.py                CRUD thèses
│   ├── evaluations.py           Évaluation périodique
│   └── performance.py           Métriques système
│
├── thesis/
│   ├── generator.py             Génération narrative
│   ├── risks.py                 Analyse des risques
│   └── catalysts.py             Détection catalyseurs
│
├── alerts/
│   ├── engine.py                Moteur d'alertes
│   ├── rules.py                 Règles de déclenchement
│   └── notifiers.py             Email, Telegram, etc.
│
├── dashboard/
│   ├── app.py                   Point d'entrée Streamlit
│   ├── pages/
│   │   ├── 1_radar.py           Heat Map sectorielle
│   │   ├── 2_opportunities.py   Liste des opportunités
│   │   ├── 3_thesis.py          Détail d'une thèse
│   │   ├── 4_performance.py     Track record du système
│   │   └── 5_memory.py          Historique des prédictions
│   └── components/
│       ├── charts.py
│       └── tables.py
│
├── backtesting/
│   ├── engine.py                Simulation historique
│   ├── data_builder.py          Reconstruction features PIT
│   └── metrics.py               Sharpe, alpha, drawdown
│
├── scheduler/
│   └── jobs.py                  Tâches périodiques
│
├── tests/
│   ├── test_collectors/
│   ├── test_features/
│   ├── test_scoring/
│   └── test_memory/
│
├── data/
│   ├── raw/                     Données brutes (gitignore)
│   ├── processed/               Features calculées (gitignore)
│   └── cache/                   Cache temporaire (gitignore)
│
└── notebooks/
    ├── exploration/             Analyses ponctuelles
    └── backtests/               Validation historique
```

---

## 5. Schéma de Base de Données

### 5.1 Tables principales

**raw_data** — toutes les données brutes horodatées

```sql
CREATE TABLE raw_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,              -- 'arxiv', 'github', 'sec_edgar', etc.
    entity_type TEXT NOT NULL,         -- 'paper', 'repo', 'filing', etc.
    entity_id TEXT NOT NULL,           -- identifiant unique côté source
    fetched_at TIMESTAMP NOT NULL,     -- quand on a récupéré la donnée
    content_at TIMESTAMP,              -- timestamp de la donnée elle-même
    payload_json TEXT NOT NULL,        -- données brutes en JSON
    hash TEXT NOT NULL,                -- hash pour déduplication
    UNIQUE(source, entity_id, hash)
);

CREATE INDEX idx_raw_source_time ON raw_data(source, content_at);
CREATE INDEX idx_raw_entity ON raw_data(entity_type, entity_id);
```

**features** — features calculées point-in-time

```sql
CREATE TABLE features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,        -- 'arxiv_velocity_cs.LG', 'rsi_14d', etc.
    target_type TEXT NOT NULL,         -- 'sector' ou 'asset'
    target_id TEXT NOT NULL,           -- 'quantum_computing' ou 'NVDA'
    computed_at TIMESTAMP NOT NULL,    -- quand la feature était valide
    value REAL NOT NULL,
    metadata_json TEXT,                -- infos complémentaires
    UNIQUE(feature_name, target_type, target_id, computed_at)
);

CREATE INDEX idx_feat_target ON features(target_type, target_id, computed_at);
CREATE INDEX idx_feat_name ON features(feature_name, computed_at);
```

**sectors** — définition des secteurs

```sql
CREATE TABLE sectors (
    id TEXT PRIMARY KEY,               -- 'quantum_computing'
    name TEXT NOT NULL,                -- 'Ordinateur quantique'
    category TEXT NOT NULL,            -- 'deep_tech', 'pharma', etc.
    keywords_json TEXT NOT NULL,       -- mots-clés pour filtrage
    arxiv_categories_json TEXT,        -- cs.LG, quant-ph, etc.
    active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**theses** — les prédictions du système (immuables)

```sql
CREATE TABLE theses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    asset_type TEXT NOT NULL,          -- 'stock', 'crypto', 'startup'
    asset_id TEXT NOT NULL,            -- 'NVDA', 'bitcoin', etc.
    sector_id TEXT NOT NULL,
    score REAL NOT NULL,               -- score composite 0-100
    score_breakdown_json TEXT NOT NULL, -- décomposition par dimension
    recommendation TEXT NOT NULL,      -- 'BUY', 'WATCH', 'AVOID'
    horizon_days INTEGER NOT NULL,     -- horizon d'investissement
    entry_price REAL,                  -- pour les cotés
    entry_conditions_json TEXT,        -- zone d'entrée, signaux
    triggers_json TEXT NOT NULL,       -- signaux déclencheurs
    risks_json TEXT NOT NULL,          -- risques identifiés
    catalysts_json TEXT,               -- événements à venir
    narrative TEXT NOT NULL,           -- thèse en texte
    model_version TEXT NOT NULL,       -- version du modèle utilisé
    weights_snapshot_json TEXT NOT NULL -- poids utilisés (pour reproductibilité)
);

CREATE INDEX idx_theses_asset ON theses(asset_type, asset_id);
CREATE INDEX idx_theses_created ON theses(created_at);
```

**evaluations** — évaluation périodique des thèses

```sql
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thesis_id INTEGER NOT NULL,
    evaluated_at TIMESTAMP NOT NULL,
    days_since_thesis INTEGER NOT NULL, -- 30, 90, 180, 365, 540...
    current_price REAL,                -- pour les cotés
    return_pct REAL,                   -- rendement depuis entrée
    benchmark_return_pct REAL,         -- rendement du benchmark
    alpha_pct REAL,                    -- alpha vs benchmark
    status TEXT NOT NULL,              -- 'active', 'success', 'failure', 'partial'
    events_occurred_json TEXT,         -- événements survenus
    predicted_catalysts_hit BOOLEAN,   -- catalyseurs anticipés se sont-ils produits
    notes TEXT,                        -- notes qualitatives
    FOREIGN KEY (thesis_id) REFERENCES theses(id)
);

CREATE INDEX idx_eval_thesis ON evaluations(thesis_id);
CREATE INDEX idx_eval_date ON evaluations(evaluated_at);
```

**signal_performance** — track record par signal

```sql
CREATE TABLE signal_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name TEXT NOT NULL,
    sector_id TEXT,                    -- NULL = tous secteurs confondus
    horizon_days INTEGER NOT NULL,
    n_predictions INTEGER NOT NULL,
    n_successes INTEGER NOT NULL,
    accuracy REAL NOT NULL,            -- taux de succès
    avg_alpha REAL,                    -- alpha moyen généré
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(signal_name, sector_id, horizon_days)
);
```

**alerts** — alertes déclenchées

```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rule_name TEXT NOT NULL,
    severity TEXT NOT NULL,            -- 'info', 'warning', 'critical'
    asset_type TEXT,
    asset_id TEXT,
    sector_id TEXT,
    message TEXT NOT NULL,
    data_json TEXT,
    thesis_id INTEGER,                 -- si l'alerte a généré une thèse
    acknowledged BOOLEAN DEFAULT 0,
    FOREIGN KEY (thesis_id) REFERENCES theses(id)
);
```

### 5.2 Principes de stockage

Les données brutes ne sont jamais modifiées ni supprimées. Les corrections sont gérées par versioning — une nouvelle entrée remplace logiquement l'ancienne sans la détruire.

Les features sont recalculables à tout moment à partir des données brutes. Si le calcul change, on peut reconstituer l'historique complet.

Les thèses et évaluations sont append-only. Ça garantit l'intégrité du track record et empêche toute réécriture de l'histoire.

---

## 6. Sources de Données Détaillées

### 6.1 Recherche académique

**arXiv** Client officiel Python, pas de clé API. Catégories suivies initialement : cs.AI, cs.LG, cs.CL, cs.CV, cs.RO, cs.CR, q-bio, quant-ph, physics.space-ph. Fréquence de collecte : quotidienne.

**Semantic Scholar** API gratuite, 100 req/s sans clé. Utilisée pour enrichir arXiv avec les métriques de citations.

**PubMed** API gratuite via E-utilities NCBI. Essentiel pour la pharma et la biotech. Fréquence : quotidienne.

**bioRxiv/medRxiv** API publique. Preprints en biologie et médecine avant peer review.

### 6.2 Développement et adoption tech

**GitHub** API officielle, 5000 req/h avec token gratuit. Signaux suivis : stars, forks, commits, contributors, langue du repo.

**Hugging Face** API publique. Nombre de downloads et Spaces pour tracker l'adoption des modèles IA.

### 6.3 Financier et régulatoire

**SEC EDGAR** API publique gratuite. Filings surveillés : 10-K, 10-Q, 8-K, 13D, 13F, Form D. Particulièrement 13D pour les activist investors et Form D pour les levées de fonds privées US.

**USASpending.gov** API REST gratuite. Tous les contrats fédéraux américains. Filtres par agence, montant, mots-clés.

**ClinicalTrials.gov** API gratuite v2. Essais cliniques mondiaux avec statut, phase, sponsor, résultats.

**FDA** Scraping des pages d'approbations et PDUFA calendar. Pas d'API officielle mais données publiques structurées.

**EU Clinical Trials Register** Données publiques, scraping nécessaire.

### 6.4 Données de marché

**yfinance** Librairie Python wrappant Yahoo Finance. Prix, volumes, fondamentaux. Gratuit mais rate-limité — utiliser avec parcimonie et cache agressif.

**CoinGecko** API publique gratuite sans clé. Prix crypto, volumes, données développeur, communauté.

**Alpha Vantage** Tier gratuit 25 req/jour. Utilisé comme backup pour les données financières.

### 6.5 Emploi et talents

**Adzuna** API gratuite 1000 req/jour après inscription. Offres d'emploi mondiales.

**HackerNews** API Firebase publique. "Who is hiring" et "Who wants to be hired" mensuels.

**Remotive** API publique pour jobs remote, utile pour tracker les startups.

### 6.6 Brevets

**Google Patents** API publique non officielle mais fonctionnelle. Recherche par entreprise, mots-clés, période.

**USPTO** API officielle gratuite. Données brevets américains.

### 6.7 News

**NewsAPI** Tier gratuit 100 req/jour. Usage parcimonieux pour les requêtes ciblées.

**RSS aggregator** Flux RSS de Reuters, Bloomberg, FT, TechCrunch, The Information, Sifted, Les Echos. Illimité et gratuit.

**Google News RSS** Flux RSS par mot-clé. Gratuit et illimité.

### 6.8 Divers

**Google Trends** Librairie pytrends. Rate-limité, utiliser avec cache.

**CoinGecko Community Data** Twitter followers, Reddit subscribers par projet crypto.

---

## 7. Modules Fonctionnels

### 7.1 Collecteurs

Chaque collecteur hérite d'une classe abstraite `BaseCollector` définissant l'interface commune :

```python
class BaseCollector:
    source_name: str

    def collect(self, since: datetime, until: datetime) -> list[dict]:
        """Récupère les données sur la période donnée."""
        raise NotImplementedError

    def normalize(self, raw: dict) -> dict:
        """Normalise en format standard."""
        raise NotImplementedError

    def store(self, items: list[dict]) -> int:
        """Stocke en base en respectant la déduplication."""
        ...
```

Règles communes à tous les collecteurs :

Gestion des rate limits avec backoff exponentiel automatique. Cache local pour éviter les requêtes redondantes. Logging détaillé des succès et échecs. Stockage systématique en table `raw_data` avec hash de déduplication.

### 7.2 Features

Une feature est une fonction pure qui prend des données brutes point-in-time et produit une valeur numérique.

**Features de vélocité** Dérivée temporelle d'un compteur. Utilisée pour arXiv, GitHub, jobs, brevets. Calculée en comparant le volume sur une fenêtre récente vs une fenêtre de référence.

**Features techniques** RSI, MACD, volume ratio, momentum 14j/30j. Calculées à partir des prix historiques.

**Features fondamentales** PE, EV/EBITDA, PEG, debt/equity, profit margin, revenue growth. Extraites de yfinance.

**Features de sentiment** Score FinBERT appliqué aux titres et résumés de news. Agrégé par actif et par secteur.

**Features d'agrégation sectorielle** Somme pondérée des features d'actifs individuels vers le niveau sectoriel.

### 7.3 Scoring

Le scoring produit un score composite 0-100 pour chaque asset. Décomposition :

**Pour les actions**
```
Score = 0.25 × momentum_score
      + 0.25 × signal_quality_score    (qui achète/investit ?)
      + 0.25 × health_score            (anti value-trap)
      + 0.25 × rehabilitation_momentum (confirmation)
```

**Pour la crypto**
```
Score = 0.30 × dev_activity_score   (GitHub, commits)
      + 0.25 × community_score      (Twitter, Reddit)
      + 0.25 × technical_score      (prix, volume)
      + 0.20 × sector_heat_score    (narrative sectorielle)
```

**Pour les startups (via proxies)**
Les startups non cotées ne sont pas scorées directement. Elles sont agrégées en signal sectoriel qui alimente le scoring des proxies cotés.

### 7.4 Génération de thèses

Une thèse est générée quand un score dépasse un seuil sectoriel. Elle contient :

- Un titre résumant l'opportunité en une phrase
- Le score composite et sa décomposition
- Les signaux déclencheurs avec leurs valeurs
- La narrative structurée en cinq sections (Pourquoi maintenant / Score / Catalyseurs / Risques / Entrée)
- Les catalyseurs datables à venir (PDUFA, conférences, résultats trimestriels)
- Les risques structurés (value trap, régulatoire, concurrentiel, macro)

La narrative est générée par template paramétrique au départ. Migration possible vers un LLM local (Mistral, Llama) pour des narratives plus naturelles.

### 7.5 Mémoire et évaluation

Le module de mémoire tourne quotidiennement et effectue les opérations suivantes :

Pour chaque thèse active, calcul du rendement depuis la date de création et du rendement du benchmark sectoriel sur la même période. Création d'une entrée d'évaluation aux jalons de 30, 90, 180, 365 et 540 jours.

Au jalon 180 jours, classification automatique du statut : success si alpha positif significatif (>5%), failure si alpha négatif significatif (<-5%), partial sinon.

Au jalon 365 jours, déclenchement d'une post-mortem automatique pour les thèses évaluées. La post-mortem alimente la table `signal_performance` qui recalcule les taux de précision par signal, secteur, horizon.

### 7.6 Apprentissage automatique

Phase 1 (mois 0-6) Règles expertes uniquement. Collecte de données labélisées.

Phase 2 (mois 6-12) Premier modèle XGBoost entraîné sur les thèses évaluées. Validation par train/test split temporel strict (jamais de shuffle aléatoire en finance).

Phase 3 (mois 12+) Modèle ML en production. Les règles expertes servent de baseline de sécurité et de fallback. Recalibration mensuelle des poids.

Pipeline d'entraînement :

Extraction des thèses évaluées comme dataset (features snapshot + label = alpha réalisé). Split temporel : entraînement sur les 80% plus anciennes, test sur les 20% plus récentes. Training XGBoost avec validation croisée temporelle. Évaluation : MAE sur l'alpha prédit, accuracy sur la classification buy/watch/avoid. Déploiement : sauvegarde du modèle dans `models/artifacts/` avec versioning.

### 7.7 Alertes

Moteur déclenché sur événements. Règles initiales :

**Critiques** Nouveau Form 13D sur un actif watchlist. PDUFA date <30 jours avec score >70. Sector Heat Score +20 points en 48h.

**Warnings** Paper avec >100 citations en 7 jours dans un secteur suivi. Contrat gouvernemental >100M$ détecté. Buyback >3% du flottant annoncé.

**Info** Nouvelle thèse générée avec score >70. Évaluation de thèse passe en success ou failure.

Notifications par email au départ. Extension Telegram ou Discord possible.

### 7.8 Backtesting

Le moteur de backtesting simule le système sur des données historiques en respectant strictement le point-in-time.

Procédure : choix d'une période historique (2018-2023 par exemple). Pour chaque jour de cette période, reconstruction de l'état des features telles qu'elles étaient ce jour-là. Génération de thèses synthétiques avec les règles et poids à valider. Simulation de l'évolution des portefeuilles théoriques. Calcul des métriques : Sharpe ratio, alpha vs benchmark, maximum drawdown, hit rate.

Le backtesting permet de valider une configuration avant de la passer en production. Risque majeur : overfitting sur les données historiques. Atténuation par walk-forward validation — on teste sur des périodes successives plutôt qu'une seule fois.

### 7.9 Dashboard

Application Streamlit multi-pages.

**Page 1 — Radar** Heat Map sectorielle sous forme de treemap avec couleur selon le Heat Score. Évolution historique des top 5 secteurs. Signaux en cours par secteur.

**Page 2 — Opportunités** Liste triable des thèses actives avec score, secteur, asset, horizon. Filtres par type d'actif, score minimum, horizon. Lien vers détail.

**Page 3 — Thèse** Vue détaillée d'une thèse : narrative, décomposition du score, catalyseurs calendrier, risques, graphique de prix avec zone d'entrée.

**Page 4 — Performance** Track record global du système. Performance par secteur, par signal, par horizon. Courbe equity curve d'un portefeuille théorique basé sur les recommandations.

**Page 5 — Mémoire** Historique complet des thèses avec leur statut actuel. Filtres temporels et par statut. Export CSV.

---

## 8. Ordre de Développement

Ce système est trop gros pour être construit d'un bloc. Ordre recommandé :

### Phase 1 — Fondations (2-3 semaines)

Setup du projet et structure des dossiers. Configuration, logging, gestion des secrets via .env. Schéma de base SQLite complet. Classe `BaseCollector` et deux collecteurs simples : arXiv et yfinance. Tests unitaires sur ces collecteurs. Script de scheduler minimal.

À la fin de cette phase, le système collecte des données brutes et les stocke proprement en base.

### Phase 2 — Features et Scoring minimal (3-4 semaines)

Ajout des collecteurs GitHub, SEC EDGAR, CoinGecko, NewsAPI. Module de features de base : vélocité arXiv, signaux techniques, sentiment simple. Calcul des Heat Scores sectoriels (règles manuelles). Premier scoring d'actions sur une watchlist réduite (20 tickers). Dashboard Streamlit basique affichant Heat Map et scores.

À la fin de cette phase, le système produit des premiers scores exploitables.

### Phase 3 — Mémoire et Thèses (2-3 semaines)

Génération de thèses structurées avec narrative templatée. Stockage des thèses en base. Module d'évaluation automatique aux jalons. Page dashboard Performance avec premières métriques. Système d'alertes email basique.

À la fin de cette phase, le système génère des prédictions archivées et évaluables.

### Phase 4 — Enrichissement sectoriel (4-6 semaines)

Ajout des collecteurs spécialisés : ClinicalTrials.gov, FDA, USASpending.gov, PubMed, Semantic Scholar, patents. Modules de scoring sectoriel spécialisés (pharma, défense, spatial). Extension de la watchlist. Amélioration des templates de thèses par secteur.

À la fin de cette phase, le système couvre les secteurs spécialisés identifiés.

### Phase 5 — Backtesting (3-4 semaines)

Moteur de backtesting avec reconstruction PIT. Validation sur période 2018-2023. Ajustement des poids basé sur les résultats. Walk-forward validation. Documentation des performances historiques.

À la fin de cette phase, on a une mesure objective de la valeur du système.

### Phase 6 — Machine Learning (en continu à partir du mois 6)

Extraction du dataset depuis les thèses évaluées. Premier modèle XGBoost. Pipeline d'entraînement et versioning des modèles. A/B testing règles vs ML. Bascule progressive vers ML si validation concluante.

Cette phase n'a pas de fin — c'est une amélioration continue.

---

## 9. Mesures de Succès

Le système est considéré comme fonctionnel si, après douze mois d'opération, les critères suivants sont remplis :

Au moins 50 thèses générées et évaluées.

Taux de succès global supérieur à 55% (baseline aléatoire = 50% pour buy/avoid).

Alpha moyen annualisé positif vs benchmark sectoriel.

Sharpe ratio du portefeuille simulé supérieur à 1.0.

Pas de bug critique non résolu en production.

Temps de génération d'une thèse inférieur à 30 secondes.

---

## 10. Risques et Mitigations

**Data leakage** Risque principal. Mitigation par discipline stricte du point-in-time, tests unitaires dédiés sur le backtesting, revue systématique des features avant intégration.

**Overfitting** Risque en phase ML. Mitigation par walk-forward validation, train/test splits temporels, régularisation des modèles, préférence pour la simplicité.

**Biais de survivance** Les données historiques ne contiennent que les boîtes qui existent encore. Mitigation par utilisation de datasets historiques corrigés quand disponibles, conservation explicite des boîtes disparues dans la watchlist initiale.

**Rate limits et bannissements** Les APIs gratuites ont des limites. Mitigation par cache agressif, rotation des user agents pour le scraping, respect des robots.txt, fallbacks entre sources.

**Dette technique** Un projet aussi large accumule facilement de la dette. Mitigation par tests unitaires systématiques, code review avant chaque merge, refactoring programmé tous les deux mois.

**Burnout personnel** C'est un projet ambitieux construit en solo à côté des études. Mitigation par découpage en phases livrables, pas de pression sur les délais, priorité à la qualité sur la vitesse.

---

## 11. Évolutions Futures

Intégration LLM local pour génération de thèses naturelles (Mistral 7B quantized, tournant en local).

Ajout d'analyse de flux (order flow) si des données gratuites émergent.

Extension à des marchés émergents (Asie, Amérique latine).

API publique pour partager les thèses avec des collaborateurs.

Migration SQLite vers PostgreSQL si volume dépasse 10 GB.

Conteneurisation Docker pour déploiement sur serveur dédié.

---

## 12. Philosophie de Développement

Le système n'a pas besoin d'être parfait pour être utile. Un prototype qui tourne et collecte des données depuis six mois est plus précieux qu'un système parfait qui n'existe que sur papier.

La première priorité est de construire la **boucle complète** — collecte, scoring, thèse, évaluation — même avec peu de sources au départ. Une boucle complète avec trois collecteurs vaut mieux qu'une collecte riche sans scoring.

Chaque module doit pouvoir tourner en mode dégradé si une source est indisponible. Le système ne doit jamais tomber entièrement à cause d'une API externe qui change.

La documentation dans le code est obligatoire. Chaque fonction publique a une docstring. Chaque module a un README expliquant son rôle dans l'architecture.

Les tests unitaires sont non négociables pour les modules de scoring, de features et de mémoire. Les collecteurs peuvent avoir des tests d'intégration moins stricts.

Le système est construit pour être utilisé, pas pour impressionner. Les métriques qui comptent sont le nombre de thèses générées et leur précision — pas la complexité du code ou l'élégance de l'architecture.
