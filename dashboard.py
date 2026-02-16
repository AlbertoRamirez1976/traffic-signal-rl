import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Traffic Signal RL Dashboard", layout="wide")
st.title("Reinforcement Learning Traffic Signal Optimization")

scenario = st.selectbox("Scenario", ["single", "grid"])
csv_path = Path("results") / "tables" / f"eval_{scenario}.csv"

st.subheader("Evaluation Results")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True)

    st.subheader("Summary (mean by policy)")
    summary = df.groupby("policy")[["avg_queue", "max_queue", "total_reward"]].mean().reset_index()
    st.dataframe(summary, use_container_width=True)

    st.subheader("Average queue by policy")
    st.bar_chart(summary.set_index("policy")["avg_queue"])
else:
    st.info(f"No evaluation CSV found at: {csv_path}. Run evaluation first:\n\n"
            f"python -m src.eval.evaluate --scenario {scenario} --seeds 0 1 2 3 4")
