TARGET_RELATIVE_IMPROVEMENT = 0.10


def compute_target_mape(preliminary_val_mape: float,
                         relative_improvement: float = TARGET_RELATIVE_IMPROVEMENT) -> float:
    """The MAPE a tuned model must reach on VALIDATION to count as having met the target."""
    return preliminary_val_mape * (1 - relative_improvement)
