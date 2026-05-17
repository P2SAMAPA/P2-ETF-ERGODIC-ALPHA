import pandas as pd
import numpy as np
from data_manager import DataManager
from signal_generator import compute_ergodic_signal
from ergodic_tests import ergodicity_score
from config import UNIVERSES

def run_backtest_for_universe(universe_name, tickers):
    """Run backtest for a single universe and return results dict."""
    dm = DataManager(tickers)
    returns = dm.returns
    dates = returns.index

    print(f"Computing ergodic signal for {universe_name}...")
    signal_series = compute_ergodic_signal(returns, lookback=252, n_components=3, ma_window=21)

    benchmark_returns = returns.mean(axis=1)
    strategy_returns = pd.Series(0.0, index=dates)
    signal_aligned = signal_series.reindex(dates, method='ffill').fillna(0)
    in_market = signal_aligned > 0
    strategy_returns[in_market] = benchmark_returns[in_market]

    cum_strat = (1 + strategy_returns).cumprod()
    cum_bench = (1 + benchmark_returns).cumprod()

    window = min(252, len(returns)//2)
    per_etf_scores = ergodicity_score(returns, window=window)
    latest_scores = per_etf_scores.iloc[-1].sort_values(ascending=False)

    results_df = pd.DataFrame({
        'strategy_return': strategy_returns,
        'benchmark_return': benchmark_returns,
        'signal': signal_aligned
    }, index=dates)

    return {
        'daily_returns': results_df,
        'cumulative_strategy': cum_strat.to_frame(name="strategy"),
        'cumulative_benchmark': cum_bench.to_frame(name="benchmark"),
        'latest_etf_scores': latest_scores.to_frame(name="non_ergodicity_score"),
        'per_etf_scores_history': per_etf_scores
    }

def run_backtest_all_universes():
    """Run backtest for all universes and return a dict keyed by universe name."""
    all_results = {}
    for name, tickers in UNIVERSES.items():
        print(f"\n{'='*50}\nRunning for universe: {name}\n{'='*50}")
        all_results[name] = run_backtest_for_universe(name, tickers)
    return all_results
