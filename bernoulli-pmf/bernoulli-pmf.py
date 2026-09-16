import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    # Write code here
    pmf = np.where(np.array(x) == 1, p, 1 - p)
    return {"pmf":pmf,"mean":float(p),"variance":float(p*(1-p))}