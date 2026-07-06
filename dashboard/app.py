"""Streamlit monitoring dashboard: data quality, drift, and model metrics over runs.

Run:  streamlit run dashboard/app.py     (or `make dashboard`)
Reads the `runs` table written by src/monitoring/runs.py. This is the honest 'visualization' layer
(answers the trivago 'stand out with: Looker/BI' line with your own dashboard).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import load_settings
from src.db.database import get_engine
from src.monitoring.runs import load_runs

st.set_page_config(page_title="Used-Car Pipeline -- Data Quality Monitor", layout="wide")
st.title("Used-Car Pipeline -- Data Quality & Drift Monitor")


@st.cache_data(ttl=60)
def _load() -> pd.DataFrame:
    settings = load_settings()
    return load_runs(get_engine(), settings["database"]["runs_table"])


try:
    runs = _load()
except Exception as exc:  # noqa: BLE001
    st.warning(f"No run history yet -- run `make backfill` first.\n\nDetails: {exc}")
    st.stop()

if runs.empty:
    st.info("The runs table is empty. Run `make backfill` to populate it.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Runs recorded", len(runs))
c2.metric("Latest DQ pass rate", f"{runs['dq_pass_rate'].iloc[-1]:.0%}")
c3.metric(
    "Latest model MAE",
    f"{runs['mae'].iloc[-1]:,.0f}" if runs["mae"].notna().any() else "n/a",
)

st.subheader("Data-quality pass rate over runs")
st.line_chart(runs.set_index("batch_label")["dq_pass_rate"])

if runs["mae"].notna().any():
    st.subheader("Model error (MAE) over runs")
    st.line_chart(runs.set_index("batch_label")["mae"])

st.subheader("Rows ingested per run")
st.bar_chart(runs.set_index("batch_label")["n_rows"])

st.subheader("Data freshness (days) per run")
st.line_chart(runs.set_index("batch_label")["freshness_days"])

st.subheader("Run history")
st.dataframe(runs, use_container_width=True)

# TODO: add a PSI-over-runs chart once per-run drift results are persisted (monitoring.drift).
