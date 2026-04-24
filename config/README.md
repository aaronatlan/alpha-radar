# config/

Configuration centrale du projet. Aucune logique métier ici — uniquement
des paramètres et définitions statiques.

- `settings.py` — paramètres chargés depuis `.env` via pydantic-settings
  (paths, niveau de log, clés API). Également la fonction
  `configure_logging()` qui initialise loguru.
- `sectors.py` — liste des secteurs suivis (id, mots-clés, catégories
  arXiv associées). Réduite en Phase 1, enrichie en Phase 4.
- `watchlists.py` — tickers cotés suivis par défaut, avec leur
  rattachement sectoriel.
