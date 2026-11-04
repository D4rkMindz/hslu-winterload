import numpy as np
import pandas as pd
import pytest
from winterload.forecast.evaluation.feature_study import chronological_folds, feature_sets, out_of_fold_forecast, \
    summarize_folds
from winterload.forecast.features.dataset import CALENDAR_COLUMNS, LAG_COLUMNS, TARGET_COLUMN, WEATHER_COLUMNS


def _learnable_frame(rows: int = 8500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "event_timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
        "load_lag_48h": rng.uniform(100, 200, size=rows),
        "noise": rng.uniform(0, 1, size=rows),
    })
    frame[TARGET_COLUMN] = frame["load_lag_48h"] * 2
    return frame


def test_feature_sets_hold_every_group_combination():
    sets = feature_sets()

    assert len(sets) == 7
    assert sets["load"] == LAG_COLUMNS
    assert set(sets["load+calendar+weather"]) == set(LAG_COLUMNS) | set(CALENDAR_COLUMNS) | set(WEATHER_COLUMNS)


def test_chronological_folds_validate_right_after_training():
    frame = pd.DataFrame({"event_timestamp": pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")})

    folds = chronological_folds(frame, fold_count=3, fold_hours=10)

    assert len(folds) == 3
    for train, validation in folds:
        assert len(validation) == 10
        assert train.index[-1] + 1 == validation.index[0]
    assert folds[-1][1].index[-1] == frame.index[-1]


def test_out_of_fold_forecast_scores_every_fold():
    scored = out_of_fold_forecast(_learnable_frame())

    assert len(scored) == 6 * 1344
    assert sorted(scored["fold"].unique()) == [0, 1, 2, 3, 4, 5]
    assert not scored["forecast"].isna().any()


def test_out_of_fold_forecast_ignores_columns_it_was_not_given():
    frame = _learnable_frame()
    other = frame.assign(noise=np.random.default_rng(1).uniform(0, 1, size=len(frame)))

    first = out_of_fold_forecast(frame, columns=["load_lag_48h"])
    second = out_of_fold_forecast(other, columns=["load_lag_48h"])

    assert first["forecast"].equals(second["forecast"])


def test_summarize_folds_counts_every_fold_the_same():
    scored = pd.DataFrame({
        "fold": [0, 1, 1],
        TARGET_COLUMN: [100.0, 200.0, 200.0],
        "forecast": [110.0, 180.0, 180.0],
    })

    summary = summarize_folds("x", scored)

    assert summary.mean_mape == pytest.approx(0.10)
    assert summary.mean_mae == pytest.approx(15.0)
    assert len(summary.fold_mapes) == 2
