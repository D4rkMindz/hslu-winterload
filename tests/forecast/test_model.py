import numpy as np
import pandas as pd
from winterload.forecast.dataset import TARGET_COLUMN
from winterload.forecast.model.preliminary import PreliminaryModel


def test_fit_then_evaluate_on_a_learnable_relationship():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "event_timestamp": pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC"),
        "temperature_2m": rng.uniform(-5, 20, size=200),
        "load_lag_48h": rng.uniform(100, 200, size=200),
    })
    train[TARGET_COLUMN] = train["load_lag_48h"] * 2

    model = PreliminaryModel().fit(train)
    mape = model.evaluate(train)

    # Fitting and evaluating on the same, noise-free, learnable relationship should be near-exact.
    assert mape < 0.05
