import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    # Write code here
    x = np.array(x, dtype=float)

    rng = np.random.default_rng(seed)

    bootstrap_samples = rng.choice(
        x,
        size=(n_bootstrap, len(x)),
        replace=True
    )

    bootstrap_means = np.mean(bootstrap_samples, axis=1)

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        "bootstrap_mean": float(np.mean(bootstrap_means)),
        "lower": float(lower),
        "upper": float(upper)
    }