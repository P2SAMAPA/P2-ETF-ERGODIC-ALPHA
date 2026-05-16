import numpy as np
import pandas as pd
from scipy import stats

def birkhoff_ergodic_statistic(series, num_blocks=10):
    """
    Test ergodicity for a 1D series.
    Returns: difference (time avg - ensemble avg) and p-value.
    """
    series = np.asarray(series).flatten()
    n = len(series)
    if n < num_blocks * 2:
        return np.nan, np.nan
    
    # Time average
    time_avg = np.mean(series)
    
    # Ensemble average: split into blocks
    block_size = n // num_blocks
    blocks = [series[i*block_size:(i+1)*block_size] for i in range(num_blocks)]
    block_means = [np.mean(block) for block in blocks]
    ensemble_avg = np.mean(block_means)
    diff = time_avg - ensemble_avg
    
    # Bootstrap significance
    boot_diffs = []
    for _ in range(500):
        boot_series = np.random.choice(series, size=n, replace=True)
        boot_time = np.mean(boot_series)
        boot_blocks = np.array_split(boot_series, num_blocks)
        boot_ensemble = np.mean([np.mean(block) for block in boot_blocks])
        boot_diffs.append(boot_time - boot_ensemble)
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(diff))
    
    return diff, p_value

def ergodicity_score(returns_df, window=252, num_blocks=10):
    """
    Rolling ergodicity test for each asset separately.
    Returns: DataFrame of non-ergodicity scores (abs diff) per asset.
    """
    dates = returns_df.index
    scores = pd.DataFrame(index=dates[window:], columns=returns_df.columns)
    
    for col in returns_df.columns:
        col_vals = returns_df[col].values
        col_scores = []
        for i in range(window, len(returns_df)):
            train = col_vals[i-window:i]
            diff, _ = birkhoff_ergodic_statistic(train, num_blocks)
            col_scores.append(np.abs(diff) if not np.isnan(diff) else 0)
        scores[col] = col_scores
    return scores
