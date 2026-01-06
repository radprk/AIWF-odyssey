from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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
                record["industry"] = metadata.get("industry")
                judge = metadata.get("judge", {}) or {}
                record["judge_relevance"] = judge.get("relevance")
                record["judge_faithfulness"] = judge.get("faithfulness")
                record["judge_helpfulness"] = judge.get("helpfulness")
                record["judge_rationale"] = judge.get("rationale")
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
    "industry",
    "route",
    "actual_sec",
    "cost",
    "judge_relevance",
    "judge_faithfulness",
    "judge_helpfulness",
]
st.dataframe(filtered[timeline_cols], use_container_width=True)

st.subheader("Flow Graph")
unique_agents = list(filtered["agent_name"].fillna("Unknown").unique())
agent_positions = {name: idx for idx, name in enumerate(unique_agents)}

nodes_x = []
nodes_y = []
node_text = []
for _, row in filtered.iterrows():
    agent_name = row.get("agent_name", "Unknown")
    nodes_x.append(row["ts"])
    nodes_y.append(agent_positions.get(agent_name, 0))
    node_text.append(
        f"{agent_name}<br>Role: {row.get('role', 'n/a')}<br>Task: {row.get('task_type', 'n/a')}"
    )

edges_x = []
edges_y = []
sorted_rows = filtered.sort_values("ts").reset_index(drop=True)
for idx in range(len(sorted_rows) - 1):
    current = sorted_rows.loc[idx]
    nxt = sorted_rows.loc[idx + 1]
    edges_x.extend([current["ts"], nxt["ts"], None])
    edges_y.extend([
        agent_positions.get(current.get("agent_name", "Unknown"), 0),
        agent_positions.get(nxt.get("agent_name", "Unknown"), 0),
        None,
    ])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=edges_x,
    y=edges_y,
    mode="lines",
    line=dict(color="#A3A3A3", width=1),
    hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=nodes_x,
    y=nodes_y,
    mode="markers",
    marker=dict(size=12, color="#3B82F6"),
    text=node_text,
    hoverinfo="text",
))

fig.update_layout(
    height=350,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="Timestamp",
    yaxis=dict(
        tickmode="array",
        tickvals=list(agent_positions.values()),
        ticktext=list(agent_positions.keys()),
        title="Agent",
    ),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Agent Responses")
for _, row in filtered.iterrows():
    header = f"{row['ts']} • {row['agent_name']} ({row.get('role') or 'n/a'})"
    with st.expander(header, expanded=False):
        st.write(row.get("response", ""))
        if row.get("judge_rationale"):
            st.markdown("**Judge rationale**")
            st.write(row.get("judge_rationale"))
        if row.get("tts_path"):
            st.audio(row["tts_path"])

st.subheader("Summary")
summary = filtered.groupby("agent_name").agg(
    interactions=("agent_name", "count"),
    avg_seconds=("actual_sec", "mean"),
    avg_cost=("cost", "mean"),
    avg_relevance=("judge_relevance", "mean"),
    avg_faithfulness=("judge_faithfulness", "mean"),
    avg_helpfulness=("judge_helpfulness", "mean"),
).reset_index()
st.dataframe(summary, use_container_width=True)
