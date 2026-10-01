import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Avoid log(0) and division by zero
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    return float(np.sum(p * np.log(p / q)))