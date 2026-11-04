"""Assembles the model-ready frame from the load and weather repositories, see
winterload.forecast.features.dataset.
"""

import datetime as dt
from typing import TYPE_CHECKING

import pandas as pd
from winterload.forecast.evaluation.splits import TEST_END, TEST_START, TRAIN_END, TRAIN_START, VALIDATION_END, \
    VALIDATION_START
from winterload.forecast.features.dataset import DEFAULT_CITY_WEIGHTS, LAG_HOURS, aggregate_weather, \
    assemble_training_frame
from winterload.repository.load import LoadRepository
from winterload.repository.weather import WeatherRepository

if TYPE_CHECKING:
    from winterload.storage import Storage


def load_dataframe(storage: "Storage", start: dt.date, end: dt.date,
                   city_weights: dict[str, float] = DEFAULT_CITY_WEIGHTS) -> pd.DataFrame:
    """One row per UTC hour in [start, end) that has load and weather: the target, its lags, weather
    forecast features and calendar fields. Hours inside a gap drop out, see
    winterload.forecast.features.dataset.assemble_training_frame."""
    lookback_start = start - dt.timedelta(hours=max(LAG_HOURS))
    load = LoadRepository(storage).hourly(lookback_start, end)
    weather = aggregate_weather(WeatherRepository(storage).forecast(start, end), city_weights)
    return assemble_training_frame(load, weather, start)


def load_train_and_validation(storage: "Storage") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The train and validation dataframes for the B1/A4 preliminary-vs-tuned comparison. Never
    touches the final test year: see winterload.pipeline.training.final_test_guard for that."""
    train = load_dataframe(storage, TRAIN_START, TRAIN_END)
    validation = load_dataframe(storage, VALIDATION_START, VALIDATION_END)
    return train, validation


def load_test(storage: "Storage") -> pd.DataFrame:
    return load_dataframe(storage, TEST_START, TEST_END)
