import numpy as np
import pandas as pd
import pytest
from winterload.forecast.evaluation.weather_scores import score_forecast


def _frame(values: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "event_timestamp": pd.to_datetime([row[0] for row in values], utc=True),
        "location_id": [row[1] for row in values],
        "temperature_2m": [row[2] for row in values],
        "cloud_cover": [row[3] for row in values],
        "shortwave_radiation": [row[4] for row in values],
    })


FORECAST = [("2025-01-01 00:00", "a", 10.0, 50.0, 100.0), ("2025-01-01 00:00", "b", 20.0, 0.0, 0.0),
            ("2025-01-01 01:00", "a", 11.0, 60.0, 200.0), ("2025-01-01 01:00", "b", 15.0, 20.0, 10.0)]
TRUTH = [("2025-01-01 00:00", "a", 12.0, 40.0, 80.0), ("2025-01-01 00:00", "b", 17.0, 10.0, 0.0),
         ("2025-01-01 01:00", "a", 10.0, 70.0, 150.0), ("2025-01-01 01:00", "b", 15.0, 20.0, 20.0)]


def test_scores_two_cities_and_two_hours():
    score = score_forecast(_frame(FORECAST), _frame(TRUTH))
    assert score.hours == 4
    assert score.temperature_2m_mae == pytest.approx(1.5)  # |2, 3, 1, 0|
    assert score.cloud_cover_mae == pytest.approx(7.5)  # |10, 10, 10, 0|
    assert score.shortwave_radiation_mae == pytest.approx(20.0)  # |20, 0, 50, 10|
    expected = (2 / 285.15 + 3 / 290.15 + 1 / 283.15 + 0) / 4
    assert score.temperature_2m_mape_kelvin == pytest.approx(expected)


def test_nan_on_either_side_drops_the_row_from_every_score():
    forecast = _frame(FORECAST)
    truth = _frame(TRUTH)
    forecast.loc[0, "cloud_cover"] = np.nan
    truth.loc[1, "shortwave_radiation"] = np.nan
    score = score_forecast(forecast, truth)
    assert score.hours == 2
    assert score.temperature_2m_mae == pytest.approx(0.5)  # |1, 0|
    assert score.cloud_cover_mae == pytest.approx(5.0)  # |10, 0|
    assert score.shortwave_radiation_mae == pytest.approx(30.0)  # |50, 10|
    assert score.temperature_2m_mape_kelvin == pytest.approx((1 / 283.15 + 0) / 2)


def test_hour_present_in_only_one_frame_drops_out():
    score = score_forecast(_frame(FORECAST), _frame(TRUTH[:2]))
    assert score.hours == 2
    assert score.temperature_2m_mae == pytest.approx(2.5)
    assert score.cloud_cover_mae == pytest.approx(10.0)
    assert score.shortwave_radiation_mae == pytest.approx(10.0)
