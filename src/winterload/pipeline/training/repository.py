"""Thin storage-backed repository for the forecasting domain: fetches raw load/weather data from
storage and hands it to winterload.forecast.dataset to build a model-ready frame. The only module
in winterload.pipeline.training that imports winterload.storage.
"""

import datetime as dt

import pandas as pd
from winterload.forecast.dataset import LAG_HOURS, TARGET_COLUMN, assemble_training_frame
from winterload.forecast.splits import TEST_START, TEST_END, TRAIN_END, VALIDATION_START, TRAIN_START, VALIDATION_END
from winterload.source.openmeteo import HOURLY_VARIABLES
from winterload.storage import Storage, get_storage


def _hourly_load_series(storage: Storage, start: dt.date, end: dt.date) -> pd.Series:
    load = storage.read_range("energy_load", start, end)
    load = load.groupby("event_timestamp", as_index=False)[TARGET_COLUMN].mean()
    series = load.set_index("event_timestamp")[TARGET_COLUMN].sort_index()
    full_index = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="h",
                                inclusive="left")
    return series.reindex(full_index)


def _weather_frame(storage: Storage, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Weather forecast features, averaged (unweighted) across the four cities per hour -- see
    reviewer point E2 on the weighting choice."""
    weather = storage.read_range("weather_forecast", start, end)
    return weather.groupby("event_timestamp", as_index=False)[HOURLY_VARIABLES].mean()


def load_dataframe(storage: Storage, start: dt.date, end: dt.date) -> pd.DataFrame:
    """One row per UTC hour in [start, end): the load target, its lags, weather forecast features
    and calendar fields -- see winterload.forecast.dataset.assemble_training_frame."""
    lookback_start = start - dt.timedelta(hours=max(LAG_HOURS))
    load = _hourly_load_series(storage, lookback_start, end)
    weather = _weather_frame(storage, start, end)
    return assemble_training_frame(load, weather, start)


def load_train_and_validation(storage: Storage | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The train and validation dataframes for the B1/A4 preliminary-vs-tuned comparison. Never
    touches the final test year -- see winterload.pipeline.training.final_test_guard for that."""
    storage = storage or get_storage()
    train = load_dataframe(storage, TRAIN_START, TRAIN_END)
    validation = load_dataframe(storage, VALIDATION_START, VALIDATION_END)
    return train, validation

def load_test(storage: Storage | None = None) -> pd.DataFrame:
    storage = storage or get_storage()
    return load_dataframe(storage, TEST_START, TEST_END)