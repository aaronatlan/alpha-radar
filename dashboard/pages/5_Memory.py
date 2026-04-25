"""Page 5 — Mémoire (SPEC §7.9 Page 5).

Historique complet des thèses avec statut, filtres temporels,
filtre par statut et export CSV.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import streamlit as st

from dashboard._data import get_theses_history
from memory.database import init_db

init_db()

st.set_page_config(page_title="Mémoire — Alpha Radar", layout="wide")
st.title("🗂️ Mémoire")
st.caption("Historique complet des thèses générées et leur évaluation courante.")

# ---------------------------------------------------------------- filtres

col_status, col_from, col_to = st.columns(3)

with col_status:
    status_options = {
        "Tous": None,
        "Actives": ["active"],
        "Succès": ["success"],
        "Échecs": ["failure"],
        "Partiels": ["partial"],
        "Non évaluées": ["none"],
    }
    status_label = st.selectbox("Statut", list(status_options.keys()))
    status_filter = status_options[status_label]

with col_from:
    d_from = st.date_input(
        "Depuis", value=date.today() - timedelta(days=365)
    )
    date_from = datetime.combine(d_from, datetime.min.time()) if d_from else None

with col_to:
    d_to = st.date_input("Jusqu'au", value=date.today())
    date_to = datetime.combine(d_to, datetime.max.time()) if d_to else None

# "Non évaluées" = status is None → ne filtrons pas sur latest_status
raw_status = None if status_label in ("Tous", "Non évaluées") else status_filter

df = get_theses_history(
    status_filter=raw_status,
    date_from=date_from,
    date_to=date_to,
)

# Pour "Non évaluées", on garde uniquement les lignes sans évaluation.
if status_label == "Non évaluées":
    df = df[df["latest_status"].isna()]

st.caption(f"{len(df)} thèse(s) affichée(s).")

if df.empty:
    st.info("Aucune thèse correspondant aux filtres.")
    st.stop()

# ---------------------------------------------------------------- tableau

_STATUS_EMOJI = {
    "success": "✅",
    "failure": "❌",
    "partial": "🔶",
    "active": "🔵",
}

disp = df.copy()
disp["statut"] = disp["latest_status"].map(
    lambda s: f"{_STATUS_EMOJI.get(str(s), '—')} {s}" if s else "—"
)
disp["alpha"] = disp["latest_alpha"].map(
    lambda v: f"{v * 100:+.2f}%" if v is not None else "—"
)
disp["score"] = disp["score"].map(lambda v: f"{v:.1f}" if v is not None else "—")
disp["entry_price"] = disp["entry_price"].map(
    lambda v: f"{v:.2f}" if v is not None else "—"
)
disp["jalon"] = disp["latest_days"].map(
    lambda v: f"J+{int(v)}" if v is not None else "—"
)
disp["created_at"] = disp["created_at"].map(
    lambda v: v.strftime("%Y-%m-%d") if v is not None else "—"
)

disp = disp.rename(columns={
    "thesis_id": "ID",
    "asset_id": "Ticker",
    "sector_id": "Secteur",
    "recommendation": "Reco",
    "score": "Score",
    "created_at": "Créée le",
    "entry_price": "Prix entrée",
    "statut": "Statut",
    "jalon": "Jalon",
    "alpha": "Alpha",
})

columns_display = [
    "ID", "Ticker", "Secteur", "Reco", "Score",
    "Créée le", "Prix entrée", "Statut", "Jalon", "Alpha",
]

st.dataframe(
    disp[columns_display],
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------- export CSV

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Exporter en CSV",
    data=csv,
    file_name="theses_history.csv",
    mime="text/csv",
)
