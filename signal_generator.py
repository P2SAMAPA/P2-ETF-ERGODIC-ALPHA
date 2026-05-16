import pandas as pd
import numpy as np
from ergodic_tests import ergodicity_score
from ergodic_decomposition import ergodic_decomposition, extract_path_dependency

def compute_ergodic_signal(returns_df, lookback=252, n_components=3, ma_window=21):
    """
    Compute final trading signal (scalar per day) by averaging non-ergodicity across assets.
    """
    # 1. Per‑asset rolling ergodicity score
    non_erg_df = ergodicity_score(returns_df, window=lookback)
    # Average across assets to get a single time series
    non_erg = non_erg_df.mean(axis=1)
    
    # 2. Ergodic decomposition (on full period for components, then path dependency)
    comps, _ = ergodic_decomposition(returns_df, n_components=n_components)
    path_dep = extract_path_dependency(returns_df, comps)  # returns Series
    
    # Align indices
    common_idx = non_erg.index.intersection(path_dep.index)
    non_erg = non_erg.loc[common_idx]
    path_dep = path_dep.loc[common_idx]
    
    # Normalise and combine
    signal = (non_erg - non_erg.mean()) / non_erg.std()
    signal += (path_dep - path_dep.mean()) / path_dep.std()
    signal = signal / 2
    
    # Smooth
    signal = signal.rolling(ma_window, min_periods=1).mean()
    return signal
