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
    
    signal_series = compute_ergodic_signal(returns, lookback=252, n_components=3, ma_window=21)
    
    # Generate positions: long the ETFs with positive signal? Actually signal is per day, not per ETF.
    # For simplicity, we create a market timing signal: long the portfolio when signal > 0, else cash.
    # A more sophisticated approach: allocate across ETFs based on individual ergodicity scores.
    # Here we use signal as overall risk-on/off.
    
    positions = []
    trade_dates = []
    in_market = False
    
    for i in range(TRAIN_WINDOW, len(dates), REBALANCE_FREQ):
        current_date = dates[i]
        if current_date in signal_series.index:
            sig = signal_series.loc[current_date]
            if sig > 0:
                if not in_market:
                    # Enter long
                    positions.append({'date': current_date, 'position': 'long'})
                    in_market = True
                    trade_dates.append(current_date)
            else:
                if in_market:
                    # Exit to cash
                    positions.append({'date': current_date, 'position': 'cash'})
                    in_market = False
                    trade_dates.append(current_date)
        else:
            # No signal, stay as is
            pass
    
    # For backtest, compute daily portfolio returns: when in market, return = equal-weight portfolio return
    port_returns = returns.mean(axis=1)  # equal weight
    strategy_returns = pd.Series(0.0, index=dates)
    current_position = 'cash'
    pos_idx = 0
    for i, d in enumerate(dates):
        if pos_idx < len(trade_dates) and d >= trade_dates[pos_idx]:
            current_position = positions[pos_idx]['position']
            pos_idx += 1
        if current_position == 'long':
            strategy_returns.loc[d] = port_returns.loc[d]
        else:
            strategy_returns.loc[d] = 0.0
    
    # Compute cumulative returns
    cum_strat = (1 + strategy_returns).cumprod()
    cum_bench = (1 + port_returns).cumprod()
    
    results_df = pd.DataFrame({
        'strategy_return': strategy_returns,
        'benchmark_return': port_returns,
        'signal': signal_series.reindex(dates, method='ffill')
    })
    return results_df, cum_strat, cum_bench
