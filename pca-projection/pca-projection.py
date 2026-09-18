import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Write code here
    X = np.array(X, dtype=float)

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov = np.cov(X_centered, rowvar=False)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvectors by descending eigenvalues
    indices = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, indices[:k]]

    # Project onto top k components
    projected = X_centered @ components

    return projected.tolist()