"""Page 3 — Performance (SPEC §7.9 Page 4).

Track record global du système : KPIs, tableau signal_performance,
distribution de l'alpha par horizon.
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard._data import (
    get_alpha_by_horizon,
    get_performance_summary,
    get_signal_performance,
)
from memory.database import init_db

init_db()

st.set_page_config(page_title="Performance — Alpha Radar", layout="wide")
st.title("📊 Performance")
st.caption("Track record global du système de thèses.")

# ---------------------------------------------------------------- KPIs

summary = get_performance_summary()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Thèses générées", summary["n_theses"])
c2.metric("Thèses évaluées", summary["n_evaluated"])

if summary["success_rate"] is not None:
    c3.metric("Taux de succès", f"{summary['success_rate'] * 100:.1f}%")
else:
    c3.metric("Taux de succès", "—")

if summary["avg_alpha"] is not None:
    c4.metric("Alpha moyen", f"{summary['avg_alpha'] * 100:+.2f}%")
else:
    c4.metric("Alpha moyen", "—")

n_t = summary["n_success"] + summary["n_failure"] + summary["n_partial"]
c5.metric("Évaluations terminales", n_t)

st.divider()

# ---------------------------------------------------------------- alpha par horizon

st.subheader("Alpha & taux de succès par jalon")
df_horizon = get_alpha_by_horizon()
if df_horizon.empty:
    st.info("Pas encore d'évaluations terminales.")
else:
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        df_horizon["horizon_label"] = df_horizon["horizon_days"].apply(
            lambda d: f"J+{d}"
        )
        fig = px.bar(
            df_horizon,
            x="horizon_label",
            y="mean_alpha",
            color="success_rate",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            text=df_horizon["mean_alpha"].map(
                lambda v: f"{v * 100:+.1f}%" if v is not None else "—"
            ),
            labels={
                "horizon_label": "Jalon",
                "mean_alpha": "Alpha moyen",
                "success_rate": "Taux de succès",
            },
            title="Alpha moyen par jalon (couleur = taux de succès)",
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with col_table:
        disp = df_horizon.copy()
        disp["mean_alpha"] = disp["mean_alpha"].map(
            lambda v: f"{v * 100:+.2f}%" if v is not None else "—"
        )
        disp["success_rate"] = disp["success_rate"].map(
            lambda v: f"{v * 100:.1f}%" if v is not None else "—"
        )
        disp = disp.rename(columns={
            "horizon_days": "Jalon (j)",
            "mean_alpha": "Alpha moy.",
            "success_rate": "Succès %",
            "n_predictions": "N",
        })
        st.dataframe(
            disp[["Jalon (j)", "Alpha moy.", "Succès %", "N"]],
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# ---------------------------------------------------------------- signal_performance

st.subheader("Performance par signal")

sectors_options = ["Tous secteurs"] + sorted({
    row.sector_id
    for row in []  # populated below
})

sp_df = get_signal_performance(sector_id=None)  # tous secteurs agrégés

if sp_df.empty:
    st.info("Aucune donnée signal_performance. Le post-mortem s'exécute après "
            "les premières évaluations à 365 jours.")
else:
    # Formattage
    disp_sp = sp_df.copy()
    disp_sp["accuracy"] = disp_sp["accuracy"].map(
        lambda v: f"{v * 100:.1f}%" if v is not None else "—"
    )
    disp_sp["avg_alpha"] = disp_sp["avg_alpha"].map(
        lambda v: f"{v * 100:+.2f}%" if v is not None else "—"
    )
    disp_sp = disp_sp.rename(columns={
        "signal_name": "Signal",
        "horizon_days": "Jalon (j)",
        "n_predictions": "N",
        "n_successes": "Succès",
        "accuracy": "Précision",
        "avg_alpha": "Alpha moy.",
        "last_updated": "Mis à jour",
    })
    st.dataframe(
        disp_sp[[
            "Signal", "Jalon (j)", "N", "Succès", "Précision",
            "Alpha moy.", "Mis à jour",
        ]],
        use_container_width=True,
        hide_index=True,
    )
