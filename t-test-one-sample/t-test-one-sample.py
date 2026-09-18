import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    # Write code here
    x = np.array(x, dtype=float)

    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)

    t_stat = (mean - mu0) / (std / np.sqrt(n))

    return float(t_stat)