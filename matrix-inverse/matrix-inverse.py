import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    # Write code here
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None