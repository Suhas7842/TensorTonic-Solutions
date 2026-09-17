import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    X = np.array(X, dtype=float)

    # Center each feature
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    return (X_centered.T @ X_centered) / (X.shape[0] - 1)