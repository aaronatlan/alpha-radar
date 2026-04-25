"""Page Heat Map — intensité d'activité par secteur.

Une treemap Plotly est la vue principale : chaque rectangle = un
secteur, couleur ∝ Heat Score [0, 100]. Les secteurs sans score sont
affichés grisés (couleur neutre) pour rester visibles dans la carte.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard._data import get_sector_heat_scores
from memory.database import init_db


st.set_page_config(page_title="Heat Map", page_icon="🔥", layout="wide")
init_db()

st.title("Heat Map sectoriel")
st.caption(
    "Intensité d'activité par secteur (0 = froid, 100 = brûlant). "
    "Combinaison vélocité arXiv + vélocité GitHub (v2)."
)


df = get_sector_heat_scores()

col_left, col_right = st.columns([2, 1])

with col_left:
    if df["heat_score"].notna().sum() == 0:
        st.info("Aucun Heat Score encore calculé.")
    else:
        fig = px.treemap(
            df.fillna({"heat_score": 0.0}),
            path=["category", "sector_name"],
            values=[1] * len(df),  # rectangles de taille égale
            color="heat_score",
            range_color=[0, 100],
            color_continuous_scale="RdYlGn_r",
            hover_data={
                "sector_id": True,
                "heat_score": ":.1f",
                "computed_at": True,
            },
        )
        fig.update_layout(margin=dict(t=10, b=10, l=0, r=0), height=500)
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Top 5")
    top = (
        df.dropna(subset=["heat_score"])
        .sort_values("heat_score", ascending=False)
        .head(5)
        [["sector_name", "heat_score", "computed_at"]]
    )
    if top.empty:
        st.write("—")
    else:
        st.dataframe(
            top.rename(columns={"sector_name": "Secteur", "heat_score": "Score"}),
            hide_index=True,
            use_container_width=True,
        )


st.subheader("Détail par secteur")
display = df[["sector_id", "sector_name", "category", "heat_score", "computed_at"]]
st.dataframe(
    display.rename(
        columns={
            "sector_id": "ID",
            "sector_name": "Secteur",
            "category": "Catégorie",
            "heat_score": "Heat Score",
            "computed_at": "Calculé à",
        }
    ),
    hide_index=True,
    use_container_width=True,
)
