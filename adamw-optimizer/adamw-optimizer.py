import numpy as np

def adamw_step(w: list, m: list, v: list, grad: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    # Write code here
    # Convert to numpy arrays
    w = np.array(w, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)

    # Update first moment
    new_m = beta1 * m + (1 - beta1) * grad

    # Update second moment
    new_v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Adam parameter update
    adam_update = new_m / (np.sqrt(new_v) + eps)

    # AdamW: decoupled weight decay
    new_w = w - lr * adam_update - lr * weight_decay * w

    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v
    }