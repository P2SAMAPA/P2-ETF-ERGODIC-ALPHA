import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import re
from config import HF_OUTPUT_REPO, UNIVERSES, ACTIVE_UNIVERSE, HF_TOKEN

st.set_page_config(layout="wide")
st.title("🌀 Ergodic Theory Alpha Engine")
st.markdown("Test ergodicity of ETF returns using Birkhoff ergodic theorem.")

with st.expander("📖 How to use this dashboard for trading", expanded=False):
    st.markdown("""
    **What is ergodicity?**  
    - A process is ergodic if time average equals ensemble average.  
    - Non‑ergodic returns imply path dependency → momentum.  

    **Trading strategy:**  
    - Signal > 0 → long equal‑weight portfolio.  
    - Signal ≤ 0 → cash.  
    - Rebalance weekly.  
    """)

st.sidebar.header("Configuration")
universe_options = list(UNIVERSES.keys())
default_index = universe_options.index(ACTIVE_UNIVERSE) if ACTIVE_UNIVERSE in universe_options else 0
selected_universe = st.sidebar.selectbox("Select Universe", universe_options, index=default_index)

@st.cache_data(ttl=3600)
def get_latest_run_folder():
    if not HF_TOKEN:
        return None
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset", token=HF_TOKEN)
        run_folders = set()
        for f in files:
            match = re.match(r"(\d{8}_\d{6})/", f)
            if match:
                run_folders.add(match.group(1))
        if not run_folders:
            return None
        return sorted(run_folders, reverse=True)[0]
    except Exception as e:
        st.warning(f"Could not list repo files: {e}")
        return None

@st.cache_data
def load_results_for_universe(run_folder, universe):
    """Load results for a specific universe."""
    try:
        daily = pd.read_parquet(
            hf_hub_download(repo_id=HF_OUTPUT_REPO, filename=f"{run_folder}/{universe}_daily_returns.parquet", repo_type="dataset", token=HF_TOKEN)
        )
        cum_strat = pd.read_parquet(
            hf_hub_download(repo_id=HF_OUTPUT_REPO, filename=f"{run_folder}/{universe}_cumulative_strategy.parquet", repo_type="dataset", token=HF_TOKEN)
        )
        cum_bench = pd.read_parquet(
            hf_hub_download(repo_id=HF_OUTPUT_REPO, filename=f"{run_folder}/{universe}_cumulative_benchmark.parquet", repo_type="dataset", token=HF_TOKEN)
        )
        latest_scores = pd.read_parquet(
            hf_hub_download(repo_id=HF_OUTPUT_REPO, filename=f"{run_folder}/{universe}_latest_etf_scores.parquet", repo_type="dataset", token=HF_TOKEN)
        )
        return daily, cum_strat, cum_bench, latest_scores
    except Exception as e:
        st.warning(f"Could not load results for {universe}: {e}")
        return None, None, None, None

latest_run = get_latest_run_folder()
if latest_run is None:
    st.info("No results found. Run the backtest first.")
    st.stop()

daily, cum_strat, cum_bench, latest_scores = load_results_for_universe(latest_run, selected_universe)
if daily is None:
    st.error(f"No results available for universe '{selected_universe}'. Ensure backtest has run for all universes.")
    st.stop()

# Ensure indices are datetime
daily.index = pd.to_datetime(daily.index)
cum_strat.index = pd.to_datetime(cum_strat.index)
cum_bench.index = pd.to_datetime(cum_bench.index)

st.subheader("Non‑Ergodicity Signal (Market Timing)")
fig_signal = go.Figure()
fig_signal.add_trace(go.Scatter(x=daily.index, y=daily['signal'], mode='lines', name='Non‑Ergodicity Score'))
fig_signal.add_hline(y=0, line_dash="dash", line_color="red")
fig_signal.update_layout(yaxis_title="Score", xaxis_title="Date")
st.plotly_chart(fig_signal, use_container_width=True)

st.subheader("Cumulative Returns")
fig_ret = go.Figure()
fig_ret.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat['strategy'], mode='lines', name='Strategy'))
fig_ret.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench['benchmark'], mode='lines', name='Benchmark'))
fig_ret.update_layout(yaxis_title="Cumulative Return", xaxis_title="Date", yaxis_type="log")
st.plotly_chart(fig_ret, use_container_width=True)

strat_total = cum_strat['strategy'].iloc[-1] - 1
bench_total = cum_bench['benchmark'].iloc[-1] - 1
excess = strat_total - bench_total
st.metric("Strategy Total Return", f"{strat_total:.2%}", delta=f"vs benchmark {excess:.2%}")

st.subheader(f"Top ETFs by Non‑Ergodicity Score – {selected_universe}")
if latest_scores is not None and not latest_scores.empty:
    scores_series = latest_scores['non_ergodicity_score'].sort_values(ascending=False)
    top_etfs = scores_series.head(10)
    fig_etf = px.bar(x=top_etfs.index, y=top_etfs.values, title="Highest Non‑Ergodicity (most path‑dependent)")
    fig_etf.update_layout(yaxis_title="Score", xaxis_title="ETF")
    st.plotly_chart(fig_etf, use_container_width=True)
    st.dataframe(scores_series.to_frame(name="Score").style.format("{:.4f}"))
else:
    st.info("No ETF scores available.")

st.subheader("Daily Returns & Drawdown")
fig_daily = go.Figure()
fig_daily.add_trace(go.Scatter(x=daily.index, y=daily['strategy_return'], mode='lines', name='Strategy Daily Return'))
fig_daily.update_layout(yaxis_title="Return")
st.plotly_chart(fig_daily, use_container_width=True)

st.subheader("Signal Distribution")
fig_hist = px.histogram(daily, x='signal', nbins=50, title="Histogram of Non‑Ergodicity Score")
st.plotly_chart(fig_hist, use_container_width=True)

st.caption("All data pre‑computed in daily backtest for each universe.")
