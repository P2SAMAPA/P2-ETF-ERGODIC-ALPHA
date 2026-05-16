import pandas as pd
import numpy as np
from ergodic_tests import ergodicity_score
from ergodic_decomposition import ergodic_decomposition, extract_path_dependency

def compute_ergodic_signal(returns_df, lookback=252, n_components=3, ma_window=21):
    # Per-asset non-ergodicity scores
    non_erg_df = ergodicity_score(returns_df, window=lookback)
    non_erg_avg = non_erg_df.mean(axis=1)
    
    # Path dependency
    comps, _ = ergodic_decomposition(returns_df, n_components=n_components)
    path_dep = extract_path_dependency(returns_df, comps)
    
    common_idx = non_erg_avg.index.intersection(path_dep.index)
    non_erg_avg = non_erg_avg.loc[common_idx]
    path_dep = path_dep.loc[common_idx]
    
    signal = (non_erg_avg - non_erg_avg.mean()) / non_erg_avg.std()
    signal += (path_dep - path_dep.mean()) / path_dep.std()
    signal = signal / 2
    signal = signal.rolling(ma_window, min_periods=1).mean()
    
    return signal, non_erg_df   # also return per-ETF scores
