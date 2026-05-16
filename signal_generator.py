import pandas as pd
import numpy as np
from ergodic_tests import ergodicity_score
from ergodic_decomposition import ergodic_decomposition, extract_path_dependency

def compute_ergodic_signal(returns, lookback=252, n_components=3, ma_window=21):
    """
    Compute final trading signal: non-ergodicity score (normalised).
    High score = path dependency = momentum signal.
    """
    # 1. Rolling ergodicity test
    non_erg = ergodicity_score(returns, window=lookback)
    
    # 2. Ergodic decomposition (on full period for components, then rolling path dependency)
    comps, _ = ergodic_decomposition(returns, n_components=n_components)
    path_dep = extract_path_dependency(returns, comps)
    
    # Align indices
    common_idx = non_erg.index.intersection(path_dep.index)
    non_erg = non_erg.loc[common_idx]
    path_dep = path_dep.loc[common_idx]
    
    # Combine: average of normalised signals
    signal = (non_erg - non_erg.mean()) / non_erg.std()
    signal += (path_dep - path_dep.mean()) / path_dep.std()
    signal = signal / 2  # average
    
    # Smooth
    signal = signal.rolling(ma_window, min_periods=1).mean()
    return signal
