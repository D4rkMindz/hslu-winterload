import numpy as np
import pandas as pd
import pytest

from winterload.forecast.evaluation.tuning import (
    DEFAULT_PARAMS,
    baseline_fold_mapes,
    score_folds,
    settings_from_params,
    settings_params,
    suggest_settings,
    tuned_settings_snippet,
    tuning_folds,
)
from winterload.forecast.features.dataset import TARGET_COLUMN
from winterload.forecast.model.preliminary import PRELIMINARY_SETTINGS, LightGBMSettings, PreliminaryModel


class FixedTrial:
    """Answers every suggestion from `params`, and checks it lies inside the search space."""

    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, low, high, *, log=False):
        assert low <= self.params[name] <= high, name
        return self.params[name]

    def suggest_float(self, name, low, high, *, log=False):
        assert low <= self.params[name] <= high, name
        return self.params[name]


def _frame(rows: int = 9000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "event_timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
        "load_lag_48h": rng.uniform(100, 200, size=rows),
        "load_lag_168h": rng.uniform(100, 200, size=rows),
    })
    frame[TARGET_COLUMN] = frame["load_lag_48h"] * 2
    return frame


def test_the_first_trial_is_the_preliminary_model():
    settings = suggest_settings(FixedTrial(DEFAULT_PARAMS))

    for name in ("n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples"):
        assert getattr(settings, name) == getattr(PRELIMINARY_SETTINGS, name)
    # Bagging with subsample=1.0 draws every row, so it equals no bagging.
    assert settings.subsample == 1.0 and settings.colsample_bytree == 1.0


def test_settings_survive_the_round_trip_through_params():
    settings = suggest_settings(FixedTrial(DEFAULT_PARAMS | {"learning_rate": 0.0371, "num_leaves": 63}))

    assert settings_from_params(DEFAULT_PARAMS | {"learning_rate": 0.0371, "num_leaves": 63}) == settings


def test_the_snippet_rebuilds_the_settings_and_fits_ruff():
    settings = LightGBMSettings(n_estimators=731, learning_rate=0.0371, num_leaves=63, max_depth=-1,
                                min_child_samples=12, subsample=0.81, subsample_freq=1, colsample_bytree=0.66,
                                reg_lambda=0.42)

    snippet = tuned_settings_snippet(settings, "abc123")
    namespace = {"LightGBMSettings": LightGBMSettings}
    exec(snippet, namespace)

    assert namespace["TUNED_SETTINGS"] == settings
    assert namespace["TUNING_RUN_ID"] == "abc123"
    assert max(len(line) for line in snippet.splitlines()) <= 120


def test_settings_params_hold_every_field():
    assert set(settings_params(PRELIMINARY_SETTINGS)) >= {"n_estimators", "subsample", "reg_lambda"}


def test_score_folds_reports_the_running_mean_after_each_fold():
    folds = tuning_folds(_frame())
    reported = []

    scores = score_folds(PRELIMINARY_SETTINGS, folds, lambda fold, mean: reported.append((fold, mean)))

    assert len(scores) == len(folds) == 6
    assert [fold for fold, _ in reported] == list(range(6))
    assert reported[-1][1] == pytest.approx(np.mean(scores))


def test_a_report_that_raises_stops_the_folds():
    folds = tuning_folds(_frame())
    seen = []

    def prune_after_two(fold, mean):
        seen.append(fold)
        if fold == 1:
            raise RuntimeError("pruned")

    with pytest.raises(RuntimeError):
        score_folds(PRELIMINARY_SETTINGS, folds, prune_after_two)
    assert seen == [0, 1]


def test_folds_stay_in_time_order_even_for_unsorted_history():
    shuffled = _frame().sample(frac=1, random_state=0)

    folds = tuning_folds(shuffled)

    for train, validation in folds:
        assert train["event_timestamp"].max() < validation["event_timestamp"].min()


def test_baseline_scores_the_168h_lag():
    frame = _frame()
    frame[TARGET_COLUMN] = frame["load_lag_168h"]

    assert baseline_fold_mapes(tuning_folds(frame)) == pytest.approx([0.0] * 6)


def test_tuned_settings_change_the_model():
    frame = _frame(2000)
    default = PreliminaryModel().fit(frame).predict(frame)
    tuned = PreliminaryModel(LightGBMSettings(n_estimators=50, learning_rate=0.05, num_leaves=7, max_depth=-1,
                                              min_child_samples=40, subsample=0.6, subsample_freq=1,
                                              colsample_bytree=0.5, reg_lambda=5.0)).fit(frame).predict(frame)

    assert not np.allclose(default, tuned)
