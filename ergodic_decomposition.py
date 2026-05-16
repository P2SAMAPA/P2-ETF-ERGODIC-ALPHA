import numpy as np
from sklearn.decomposition import PCA

def ergodic_decomposition(returns, n_components=3):
    """
    Decompose return series into ergodic components using PCA on rolling moments.
    Returns: components (DataFrame of principal components)
    """
    # Use PCA on the covariance matrix of returns (assumes ergodic components are orthogonal)
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
    Path dependency signal = residual after removing ergodic components.
    High residual variance indicates non-ergodicity.
    """
    # Use regression to remove ergodic components from each asset return
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
    for col in returns.columns:
        y = returns[col].values
        X = ergodic_components.values
        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = X @ coeffs
            residuals[col] = y - pred
        except:
            residuals[col] = 0
    # Path dependency measure = variance of residuals over rolling window
    path_dep = residuals.var(axis=1)  # cross-sectional variance of residuals
    return path_dep
