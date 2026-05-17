import pandas as pd
import numpy as np
from data_manager import DataManager
from signal_generator import compute_ergodic_signal
from ergodic_tests import ergodicity_score
from config import UNIVERSES

def run_backtest_for_universe(universe_name, tickers):
    """Run ergodic backtest for a single universe. Returns results dict."""
    dm = DataManager(tickers)
    returns = dm.returns
    dates = returns.index

    print(f"Computing ergodic signal for {universe_name}...")
    signal_series = compute_ergodic_signal(returns, lookback=252, n_components=3, ma_window=21)

    benchmark_returns = returns.mean(axis=1)
    signal_aligned = signal_series.reindex(dates, method='ffill').fillna(0)
    in_market = signal_aligned > 0
    strategy_returns = pd.Series(0.0, index=dates)
    strategy_returns[in_market] = benchmark_returns[in_market]

    cum_strat = (1 + strategy_returns).cumprod()
    cum_bench = (1 + benchmark_returns).cumprod()

    # Per‑ETF ergodicity scores
    window = min(252, len(returns)//2)
    per_etf_scores = ergodicity_score(returns, window=window)
    latest_scores = per_etf_scores.iloc[-1].sort_values(ascending=False)
    per_etf_scores_history = per_etf_scores

    daily_df = pd.DataFrame({
        'strategy_return': strategy_returns,
        'benchmark_return': benchmark_returns,
        'signal': signal_aligned
    }, index=dates)

    return {
        f"{universe_name}_daily_returns": daily_df,
        f"{universe_name}_cumulative_strategy": cum_strat.to_frame(name="strategy"),
        f"{universe_name}_cumulative_benchmark": cum_bench.to_frame(name="benchmark"),
        f"{universe_name}_latest_etf_scores": latest_scores.to_frame(name="non_ergodicity_score"),
        f"{universe_name}_per_etf_scores_history": per_etf_scores_history
    }
