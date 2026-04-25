"""Page Opportunities — classement d'actions par score composite.

Table triée décroissante par `stock_score`. Filtres basiques (secteur,
score minimum). Les dimensions manquantes (sentiment avant que FinBERT
ait tourné) apparaissent vides plutôt que 0 — elles ont été sautées,
pas rabaissées.
"""
from __future__ import annotations

import streamlit as st

from config.sectors import SECTORS
from dashboard._data import get_stock_scores
from memory.database import init_db


st.set_page_config(page_title="Opportunities", page_icon="📈", layout="wide")
init_db()

st.title("Opportunités — actions")
st.caption(
    "Score composite [0, 100] par ticker. Dimensions : "
    "momentum (technique) · signal_quality (heat + filings SEC) · "
    "sentiment (FinBERT sur news sectorielles)."
)


df = get_stock_scores()

if df.empty or df["stock_score"].notna().sum() == 0:
    st.info(
        "Aucun score d'action pour le moment. "
        "Les scores sont recalculés à 23:15 par le scheduler."
    )
else:
    # --- filtres ---------------------------------------------------------
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        sector_ids = ["(tous)"] + [s["id"] for s in SECTORS]
        chosen = st.selectbox("Filtrer par secteur", sector_ids)
    with col_f2:
        min_score = st.slider("Score minimum", 0, 100, 0, step=5)

    filtered = df.copy()
    if chosen != "(tous)":
        filtered = filtered[filtered["sectors"].str.contains(chosen, na=False)]
    filtered = filtered[filtered["stock_score"].fillna(-1) >= min_score]

    st.dataframe(
        filtered,
        hide_index=True,
        use_container_width=True,
        column_config={
            "stock_score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.1f",
            ),
            "momentum": st.column_config.NumberColumn("Momentum", format="%.1f"),
            "signal_quality": st.column_config.NumberColumn("SigQual", format="%.1f"),
            "sentiment": st.column_config.NumberColumn("Sentiment", format="%.1f"),
        },
    )

    st.caption(
        f"{len(filtered)} ticker(s) affiché(s) sur {len(df)}. "
        "Une dimension vide signifie qu'aucun input n'était disponible — "
        "elle a été sautée (renormalisation sur les dimensions présentes)."
    )
