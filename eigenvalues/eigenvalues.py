import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    # Write code here
    eigenvalues = np.linalg.eigvals(matrix)

    # Remove tiny imaginary parts caused by floating-point precision
    eigenvalues = np.real_if_close(eigenvalues)

    return np.sort(eigenvalues.real)