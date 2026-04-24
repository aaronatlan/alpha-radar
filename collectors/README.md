# collectors/

Un module par source de données. Chaque collecteur hérite de
`BaseCollector` (voir `base.py`) et implémente deux méthodes :

- `collect(since, until)` — récupère les données brutes sur la période.
- `normalize(raw)` — convertit un item natif en format canonique.

La méthode `run()` (définie dans la classe mère) orchestre
collect → normalize → hashing → insertion en `raw_data` avec
déduplication via la contrainte UNIQUE `(source, entity_id, hash)`.

## Mode dégradé

Toute exception pendant `collect()` ou `normalize()` est loggée et
n'interrompt jamais le scheduler : un collecteur planté laisse les
autres continuer. C'est une règle absolue du projet.

## Point-in-time

Chaque item normalisé porte un `content_at` (timestamp intrinsèque de la
donnée, p. ex. date de publication d'un papier) distinct de
`fetched_at` (instant de l'appel). Les features en aval doivent filtrer
sur `content_at <= T_prédiction` **et** `fetched_at <= T_prédiction`
pour empêcher toute fuite d'information du futur.

## Phase 1

- `arxiv_collector.py` — preprints arXiv
- `yfinance_collector.py` — OHLCV quotidiens Yahoo Finance

Les autres collecteurs (`github`, `sec_edgar`, `coingecko`, `news`,
`clinical_trials`, `fda`, `usa_spending`, `patents`, `jobs`…) arriveront
en Phase 2 et Phase 4.
