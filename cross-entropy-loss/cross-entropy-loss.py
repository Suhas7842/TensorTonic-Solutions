import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get the predicted probability for the correct class
    correct_probs = y_pred[np.arange(len(y_true)), y_true]

    # Avoid log(0)
    correct_probs = np.clip(correct_probs, 1e-15, 1 - 1e-15)

    # Cross-entropy for each sample, then mean
    loss = -np.mean(np.log(correct_probs))

    return float(loss)