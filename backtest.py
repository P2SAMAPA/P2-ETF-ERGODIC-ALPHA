import pandas as pd
import numpy as np
from data_manager import DataManager
from signal_generator import compute_ergodic_signal
from ergodic_tests import ergodicity_score
from config import UNIVERSES, ACTIVE_UNIVERSE

def run_backtest():
    tickers = UNIVERSES[ACTIVE_UNIVERSE]
    dm = DataManager(tickers)
    returns = dm.returns
    dates = returns.index

    print(f"Computing ergodic signal for {len(returns)} days...")
    signal_series = compute_ergodic_signal(returns, lookback=252, n_components=3, ma_window=21)

    # Equal-weight benchmark returns
    benchmark_returns = returns.mean(axis=1)

    # Generate strategy returns: long when signal > 0, else cash
    strategy_returns = pd.Series(0.0, index=dates)
    signal_aligned = signal_series.reindex(dates, method='ffill').fillna(0)
    in_market = signal_aligned > 0
    strategy_returns[in_market] = benchmark_returns[in_market]

    # Cumulative returns
    cum_strat = (1 + strategy_returns).cumprod()
    cum_bench = (1 + benchmark_returns).cumprod()

    # --- Pre‑compute per‑ETF non‑ergodicity scores (latest rolling window) ---
    print("Computing per‑ETF ergodicity scores...")
    # Use a 252-day window for final scores
    window = min(252, len(returns)//2)
    per_etf_scores = ergodicity_score(returns, window=window)
    # Take the latest available score for each ETF
    latest_scores = per_etf_scores.iloc[-1].sort_values(ascending=False)
    # Also store the full history of per‑ETF scores for potential time‑series display
    per_etf_scores_history = per_etf_scores

    # Daily results DataFrame
    results_df = pd.DataFrame({
        'strategy_return': strategy_returns,
        'benchmark_return': benchmark_returns,
        'signal': signal_aligned
    }, index=dates)

    return results_df, cum_strat, cum_bench, latest_scores, per_etf_scores_history
