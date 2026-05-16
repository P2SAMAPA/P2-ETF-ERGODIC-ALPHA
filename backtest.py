import pandas as pd
import numpy as np
from data_manager import DataManager
from signal_generator import compute_ergodic_signal
from config import UNIVERSES, ACTIVE_UNIVERSE, TRAIN_WINDOW, REBALANCE_FREQ, COMMISSION

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
    # Align signal with dates (forward fill)
    signal_aligned = signal_series.reindex(dates, method='ffill').fillna(0)

    # Simple threshold: go long if signal > 0
    in_market = signal_aligned > 0
    strategy_returns[in_market] = benchmark_returns[in_market]

    # Cumulative returns
    cum_strat = (1 + strategy_returns).cumprod()
    cum_bench = (1 + benchmark_returns).cumprod()

    # Daily results DataFrame
    results_df = pd.DataFrame({
        'strategy_return': strategy_returns,
        'benchmark_return': benchmark_returns,
        'signal': signal_aligned
    }, index=dates)

    return results_df, cum_strat, cum_bench

if __name__ == "__main__":
    # Quick test
    df, cs, cb = run_backtest()
    print(f"Strategy final return: {cs.iloc[-1]-1:.2%}")
    print(f"Benchmark final return: {cb.iloc[-1]-1:.2%}")
