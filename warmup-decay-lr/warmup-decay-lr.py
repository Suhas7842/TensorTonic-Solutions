def warmup_decay_schedule(base_lr: float, warmup_steps: int, total_steps: int, current_step: int) -> float:
    """
    Returns the learning rate for the requested training step.
    """
    # Write code here
    if current_step < warmup_steps:
        # Linear warmup: 0 -> base_lr
        return base_lr * current_step / warmup_steps

    # Linear decay: base_lr -> 0
    decay_steps = total_steps - warmup_steps
    return base_lr * (total_steps - current_step) / decay_steps