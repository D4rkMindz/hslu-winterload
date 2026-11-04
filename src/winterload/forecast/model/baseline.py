"""The project's baseline forecast: load at the same hour one week earlier (docs/proposal.pdf). The
assembled frame already holds it as the 168h lag, so a comparison needs no extra data fetch."""

import numpy as np
import pandas as pd

BASELINE_COLUMN = "load_lag_168h"


def seasonal_naive_forecast(frame: pd.DataFrame) -> np.ndarray:
    return frame[BASELINE_COLUMN].to_numpy()
