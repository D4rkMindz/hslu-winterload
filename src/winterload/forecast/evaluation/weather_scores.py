"""Scores a weather forecast against a reference, reviewer point A1."""

from dataclasses import dataclass

import pandas as pd
from winterload.forecast.features.dataset import WEATHER_COLUMNS
from winterload.forecast.metrics import mae, mape

KELVIN_OFFSET = 273.15


@dataclass(frozen=True)
class WeatherScore:
    hours: int
    temperature_2m_mae: float
    cloud_cover_mae: float
    shortwave_radiation_mae: float
    temperature_2m_mape_kelvin: float


def score_forecast(forecast: pd.DataFrame, truth: pd.DataFrame) -> WeatherScore:
    """Both frames hold one row per hour and city, and rows without a value on either side drop out.

    MAPE divides by the truth. Celsius crosses zero in winter, so temperature MAPE uses Kelvin. Cloud
    cover and radiation sit at zero too often for a MAPE, so they only get a MAE.
    """
    joined = forecast.merge(truth, on=["event_timestamp", "location_id"], suffixes=("", "_truth"))
    joined = joined.dropna(subset=[*WEATHER_COLUMNS, *[f"{column}_truth" for column in WEATHER_COLUMNS]])

    def column_mae(column: str) -> float:
        return mae(joined[f"{column}_truth"].to_numpy(), joined[column].to_numpy())

    return WeatherScore(
        hours=len(joined),
        temperature_2m_mae=column_mae("temperature_2m"),
        cloud_cover_mae=column_mae("cloud_cover"),
        shortwave_radiation_mae=column_mae("shortwave_radiation"),
        temperature_2m_mape_kelvin=mape(joined["temperature_2m_truth"].to_numpy() + KELVIN_OFFSET,
                                        joined["temperature_2m"].to_numpy() + KELVIN_OFFSET),
    )
