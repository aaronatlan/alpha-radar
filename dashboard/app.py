"""Page d'accueil du dashboard — vue d'ensemble + santé des collectes.

Lancement :

    streamlit run dashboard/app.py

Les pages dédiées (Heat Map, Opportunités) sont dans `dashboard/pages/`,
chargées automatiquement par le routage multi-page de Streamlit.
"""
from __future__ import annotations

import streamlit as st

from dashboard._data import (
    get_collector_health,
    get_feature_freshness,
)
from memory.database import init_db


st.set_page_config(page_title="Alpha Radar", page_icon=":signal_strength:", layout="wide")

# La base peut ne pas exister au premier lancement (dev local). On la
# crée de façon idempotente avant toute lecture.
init_db()

st.title("Alpha Radar")
st.caption(
    "Système d'analyse d'investissement multi-sources — "
    "détection d'opportunités avant le marché."
)

st.subheader("Navigation")
st.write(
    "Utiliser le menu latéral pour accéder aux vues :\n"
    "- **Heat Map** — intensité par secteur (arXiv + GitHub).\n"
    "- **Opportunities** — classement actions avec dimensions."
)


# -------------------------------------------------------------- health


st.subheader("Santé des collectes")
health = get_collector_health()
if health.empty:
    st.info("Aucune donnée collectée pour le moment. Lancez `python -m scheduler.jobs`.")
else:
    st.dataframe(health, hide_index=True, use_container_width=True)


st.subheader("Fraîcheur des features")
freshness = get_feature_freshness()
if freshness.empty:
    st.info("Aucune feature calculée. Les jobs techniques tournent à 22:30 / 22:45 / 23:00.")
else:
    st.dataframe(freshness, hide_index=True, use_container_width=True)


with st.expander("Conventions"):
    st.markdown(
        """
        - Tous les timestamps sont naïfs, implicitement **UTC**.
        - Les valeurs affichées sont **point-in-time** à l'instant courant.
        - Les Heat Scores et stock scores sont dans l'intervalle **[0, 100]**.
        - Le sentiment brut est dans [−1, +1]. Il est mappé linéairement
          vers [0, 100] pour le scoring d'actions.
        """
    )
