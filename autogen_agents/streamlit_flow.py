from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


LOG_DIR = Path("data/logs")


def load_logs() -> pd.DataFrame:
    records: list[dict] = []
    if not LOG_DIR.exists():
        return pd.DataFrame()
    for path in LOG_DIR.glob("*.jsonl"):
        agent_name = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["agent_name"] = agent_name
                metadata = record.get("metadata", {}) or {}
                record["session_id"] = metadata.get("session_id")
                record["role"] = metadata.get("role")
                record["task_type"] = metadata.get("task_type")
                record["route"] = ", ".join(metadata.get("route", []))
                record["actual_sec"] = metadata.get("actual_sec")
                record["cost"] = metadata.get("cost")
                record["tts_path"] = metadata.get("tts_path")
                records.append(record)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df.sort_values("ts")


st.set_page_config(page_title="Call Center Flow Viewer", layout="wide")
st.title("📞 Call Center Flow Viewer")
st.caption("Inspect which agent handled each step, with timing, routing, and outputs.")

data = load_logs()
if data.empty:
    st.info("No logs found yet. Run `python main.py --mode cli` to generate data.")
    st.stop()

session_ids = [sid for sid in data["session_id"].dropna().unique()]
selected_session = st.selectbox("Session", options=session_ids, index=0)

filtered = data[data["session_id"] == selected_session]

st.subheader("Flow Timeline")
timeline_cols = [
    "ts",
    "agent_name",
    "role",
    "task_type",
    "route",
    "actual_sec",
    "cost",
]
st.dataframe(filtered[timeline_cols], use_container_width=True)

st.subheader("Agent Responses")
for _, row in filtered.iterrows():
    header = f"{row['ts']} • {row['agent_name']} ({row.get('role') or 'n/a'})"
    with st.expander(header, expanded=False):
        st.write(row.get("response", ""))
        if row.get("tts_path"):
            st.audio(row["tts_path"])

st.subheader("Summary")
summary = filtered.groupby("agent_name").agg(
    interactions=("agent_name", "count"),
    avg_seconds=("actual_sec", "mean"),
    avg_cost=("cost", "mean"),
).reset_index()
st.dataframe(summary, use_container_width=True)
