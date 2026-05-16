import numpy as np
from scipy import stats

def birkhoff_ergodic_statistic(series, num_blocks=10):
    """
    Test ergodicity by comparing time average vs ensemble average.
    Returns: test statistic (difference) and p-value (if significantly different from zero).
    """
    n = len(series)
    # Time average
    time_avg = np.mean(series)
    
    # Ensemble average: split into blocks, compute mean of each block, then average
    block_size = n // num_blocks
    if block_size < 2:
        return np.nan, np.nan
    blocks = [series[i*block_size:(i+1)*block_size] for i in range(num_blocks)]
    block_means = [np.mean(block) for block in blocks]
    ensemble_avg = np.mean(block_means)
    
    # Difference as test statistic
    diff = time_avg - ensemble_avg
    
    # Bootstrap for significance
    # Null: diff should be zero if ergodic
    boot_diffs = []
    for _ in range(500):
        boot_series = np.random.choice(series, size=n, replace=True)
        boot_time = np.mean(boot_series)
        boot_blocks = np.array_split(boot_series, num_blocks)
        boot_ensemble = np.mean([np.mean(block) for block in boot_blocks])
        boot_diffs.append(boot_time - boot_ensemble)
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(diff))
    
    return diff, p_value

def ergodicity_score(returns, window=252, num_blocks=10):
    """
    Rolling ergodicity test.
    Returns: Series of test statistics (absolute diff) – higher = more non-ergodic.
    """
    dates = returns.index
    scores = []
    for i in range(window, len(returns)):
        train = returns.iloc[i-window:i].values
        diff, p = birkhoff_ergodic_statistic(train, num_blocks)
        scores.append(np.abs(diff) if not np.isnan(diff) else 0)
    return pd.Series(scores, index=dates[window:], name='non_ergodicity')
