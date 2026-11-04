import numpy as np
import pandas as pd
import pytest
from winterload.forecast.features.dataset import TARGET_COLUMN
from winterload.forecast.model.preliminary import EARLY_STOPPING_SETTINGS, ModelNotFitted, PreliminaryModel


def _learnable_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "event_timestamp": pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC"),
        "temperature_2m": rng.uniform(-5, 20, size=200),
        "load_lag_48h": rng.uniform(100, 200, size=200),
    })
    frame[TARGET_COLUMN] = frame["load_lag_48h"] * 2
    return frame


def test_fit_then_evaluate_on_a_learnable_relationship():
    train = _learnable_frame()

    # Scoring on the training rows of a noise-free relationship should be near-exact.
    assert PreliminaryModel().fit(train).evaluate(train) < 0.05


def test_from_estimator_predicts_the_same_as_the_fitted_model():
    train = _learnable_frame()
    model = PreliminaryModel().fit(train)

    restored = PreliminaryModel.from_estimator(model.estimator)

    assert restored.evaluate(train) == model.evaluate(train)


def test_unfitted_model_refuses_to_forecast():
    frame = _learnable_frame()

    with pytest.raises(ModelNotFitted):
        PreliminaryModel().predict(frame)
    with pytest.raises(ModelNotFitted):
        PreliminaryModel().evaluate(frame)


def test_default_settings_match_the_preliminary_hyperparameters():
    params = PreliminaryModel().estimator.get_params()

    assert params["n_estimators"] == 100
    assert params["learning_rate"] == 0.1
    assert params["num_leaves"] == 31
    assert params["max_depth"] == -1
    assert params["min_child_samples"] == 20


def test_early_stopping_stops_before_the_tree_limit():
    train = _learnable_frame()

    model = PreliminaryModel(EARLY_STOPPING_SETTINGS).fit(train)

    assert 1 <= model.estimator.best_iteration_ < 2000
    assert model.evaluate(train) < 0.05


def test_from_estimator_keeps_an_early_stopped_model():
    train = _learnable_frame()
    model = PreliminaryModel(EARLY_STOPPING_SETTINGS).fit(train)

    restored = PreliminaryModel.from_estimator(model.estimator)

    assert np.array_equal(restored.predict(train), model.predict(train))
