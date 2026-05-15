"""
dashboard.py — Streamlit eval dashboard.

Run with:
    streamlit run dashboard.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd

from src.eval_runner import load_recent_runs, load_run_by_id

st.set_page_config(page_title="Eval Dashboard", layout="wide", page_icon="🔬")

st.title("🔬 Model Regression Dashboard")
st.caption("Email classifier · prompt regression detection")


runs = load_recent_runs(20)

if not runs:
    st.warning("No eval runs found. Run `python -m src.ci_run` to create the first one.")
    st.stop()


st.subheader("Pass rate over time")

df = pd.DataFrame(runs)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df["moving_avg"] = df["pass_rate"].rolling(7, min_periods=1).mean()

chart_df = df[["timestamp", "pass_rate", "moving_avg"]].set_index("timestamp")
st.line_chart(chart_df, color=["#3b82f6", "#f59e0b"])
st.caption("Blue = per-run pass rate · Amber = 7-run moving average")


latest = runs[0]
prev   = runs[1] if len(runs) > 1 else None

col1, col2, col3, col4 = st.columns(4)
delta = (latest["pass_rate"] - prev["pass_rate"]) if prev else 0
col1.metric("Pass rate",    f"{latest['pass_rate']:.1%}", f"{delta:+.1%}")
col2.metric("Avg latency",  f"{latest['avg_latency_ms']:.0f}ms")
col3.metric("Total tokens", f"{latest['total_tokens']:,}")
col4.metric("Cases",        latest["total_cases"])

st.divider()


st.subheader("Per-category accuracy (latest run)")
cat_data = latest["per_category_accuracy"]
cat_df = pd.DataFrame(
    [{"Category": k, "Accuracy": v} for k, v in cat_data.items()]
).set_index("Category")
st.bar_chart(cat_df)

st.divider()


st.subheader("All runs")
table_df = pd.DataFrame(runs)[
    ["run_id", "prompt_version", "model", "timestamp", "pass_rate", "passed", "failed", "avg_latency_ms", "total_tokens"]
].rename(columns={
    "run_id": "Run",
    "prompt_version": "Prompt",
    "model": "Model",
    "timestamp": "Time",
    "pass_rate": "Pass rate",
    "passed": "✓",
    "failed": "✗",
    "avg_latency_ms": "Latency (ms)",
    "total_tokens": "Tokens",
})
table_df["Pass rate"] = table_df["Pass rate"].map("{:.1%}".format)
st.dataframe(table_df, use_container_width=True, hide_index=True)


st.divider()
st.subheader("Drill into a run")

run_ids = [r["run_id"] for r in runs]
selected = st.selectbox("Select run", run_ids, index=0)

if selected:
    run = load_run_by_id(selected)
    if run and run.case_scores:
        case_df = pd.DataFrame([
            {
                "Case ID": s.case_id,
                "Passed": "✓" if s.passed else "✗",
                "Category match": s.category_match,
                "Predicted": s.predicted_category or "—",
                "Summary relevance": s.summary_relevance,
                "Latency (ms)": s.latency_ms,
                "Error": s.error or "",
            }
            for s in run.case_scores
        ])

        failed_only = st.checkbox("Show failures only", value=False)
        if failed_only:
            case_df = case_df[case_df["Passed"] == "✗"]

        st.dataframe(case_df, use_container_width=True, hide_index=True)
    else:
        st.info("No case scores available for this run.")