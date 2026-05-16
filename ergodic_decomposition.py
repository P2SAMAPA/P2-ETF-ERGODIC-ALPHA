import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def ergodic_decomposition(returns, n_components=3):
    """
    Decompose return series into ergodic components using PCA on covariance matrix.
    Returns: components (DataFrame of principal components), explained variance ratio.
    """
    cov_matrix = returns.cov()
    pca = PCA(n_components=n_components)
    pca.fit(cov_matrix)
    # Project returns onto principal components
    components = pca.transform(returns)
    component_names = [f'ergodic_comp_{i+1}' for i in range(n_components)]
    comp_df = pd.DataFrame(components, index=returns.index, columns=component_names)
    return comp_df, pca.explained_variance_ratio_

def extract_path_dependency(returns, ergodic_components):
    """
    Path dependency signal = residual variance after removing ergodic components.
    Returns: Series of variance of residuals across assets (cross-sectional).
    """
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
    for col in returns.columns:
        y = returns[col].values
        X = ergodic_components.values
        # Add constant term
        X = np.column_stack([np.ones(len(X)), X])
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = X @ coeffs
            residuals[col] = y - pred
        except:
            residuals[col] = 0.0
    # Cross-sectional variance of residuals at each time step
    path_dep = residuals.var(axis=1, skipna=True)
    path_dep.name = 'path_dependency'
    return path_dep
