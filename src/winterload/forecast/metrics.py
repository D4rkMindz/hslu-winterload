import numpy as np


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error as a fraction (0.0615, not 6.15)."""
    return float(np.mean(np.abs((actual - predicted) / actual)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error in the target's unit (MW)."""
    return float(np.mean(np.abs(actual - predicted)))
