def run_backtest():
    tickers = UNIVERSES[ACTIVE_UNIVERSE]
    dm = DataManager(tickers)
    returns = dm.returns
    
    signal_series, per_etf_nonerg = compute_ergodic_signal(returns, lookback=252, n_components=3, ma_window=21)
    
    # ... (rest of backtest logic unchanged)
    return results_df, cum_strat, cum_bench, per_etf_nonerg
