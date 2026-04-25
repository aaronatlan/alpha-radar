"""Page 4 — Alertes (SPEC §7.7).

Timeline des alertes filtrables par sévérité et secteur.
Bouton d'accusé de réception par alerte.
"""
from __future__ import annotations

import streamlit as st

from dashboard._data import acknowledge_alert, get_alerts
from memory.database import init_db

init_db()

st.set_page_config(page_title="Alertes — Alpha Radar", layout="wide")
st.title("🔔 Alertes")

# ---------------------------------------------------------------- filtres

col_sev, col_sec, col_ack = st.columns(3)

with col_sev:
    sev_options = ["Toutes", "critical", "warning", "info"]
    sev = st.selectbox("Sévérité", sev_options)
    sev_filter = None if sev == "Toutes" else sev

with col_sec:
    sector_input = st.text_input("Secteur (laisser vide = tous)", value="")
    sector_filter = sector_input.strip() or None

with col_ack:
    ack_options = {"Toutes": None, "Non traitées": False, "Traitées": True}
    ack_label = st.selectbox("Statut", list(ack_options.keys()))
    ack_filter = ack_options[ack_label]

df = get_alerts(
    severity=sev_filter,
    sector_id=sector_filter,
    acknowledged=ack_filter,
    limit=200,
)

st.caption(f"{len(df)} alerte(s) affichée(s).")

if df.empty:
    st.success("Aucune alerte correspondant aux filtres.")
    st.stop()

# ---------------------------------------------------------------- timeline

_SEVERITY_COLORS = {
    "critical": "🔴",
    "warning": "🟠",
    "info": "🔵",
}

for _, row in df.iterrows():
    icon = _SEVERITY_COLORS.get(str(row["severity"]), "⚪")
    ack_badge = " ✅" if row["acknowledged"] else ""
    ts = row["created_at"]
    ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "—"

    with st.expander(
        f"{icon} [{str(row['severity']).upper()}] {row['rule_name']} — "
        f"{ts_str}{ack_badge}",
        expanded=not row["acknowledged"],
    ):
        st.markdown(row["message"])

        meta_cols = st.columns(4)
        meta_cols[0].markdown(f"**Asset** : {row['asset_id'] or '—'}")
        meta_cols[1].markdown(f"**Secteur** : {row['sector_id'] or '—'}")
        meta_cols[2].markdown(
            f"**Thèse** : #{row['thesis_id']}" if row["thesis_id"] else "**Thèse** : —"
        )
        meta_cols[3].markdown(f"**Règle** : `{row['rule_name']}`")

        if not row["acknowledged"]:
            if st.button("Marquer comme traitée", key=f"ack_{row['id']}"):
                acknowledge_alert(int(row["id"]))
                st.success("Alerte traitée.")
                st.rerun()
