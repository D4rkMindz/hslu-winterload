import datetime as dt

import pandas as pd
from winterload.forecast.dataset import LAG_COLUMNS, TARGET_COLUMN, assemble_training_frame


def _hourly_series(start: dt.date, hours: int, values: list[float]) -> pd.Series:
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=hours, freq="h")
    return pd.Series(values, index=index)


def test_assembles_lags_weather_and_calendar_fields():
    start = dt.date(2025, 1, 8)  # a Wednesday
    # One week of lookback (168h, the largest lag) plus 3 target hours of load.
    load = _hourly_series(start - dt.timedelta(hours=168), 168 + 3, [float(i) for i in range(168 + 3)])
    weather = pd.DataFrame({
        "event_timestamp": pd.date_range(pd.Timestamp(start, tz="UTC"), periods=3, freq="h"),
        "temperature_2m": [1.0, 2.0, 3.0],
    })

    frame = assemble_training_frame(load, weather, start)

    assert len(frame) == 3
    assert list(frame[TARGET_COLUMN]) == [168.0, 169.0, 170.0]
    # load_lag_48h at hour 0 of `start` is the value 48 hours earlier: index 168-48=120.
    assert frame["load_lag_48h"].iloc[0] == 120.0
    assert frame["load_lag_168h"].iloc[0] == 0.0
    assert list(frame["temperature_2m"]) == [1.0, 2.0, 3.0]
    # UTC 00:00 on 2025-01-08 is 01:00 Europe/Zurich (CET, UTC+1 in winter).
    assert list(frame["hour"]) == [1, 2, 3]
    assert frame["day_of_week"].iloc[0] == 2  # Wednesday
    assert frame["is_weekend"].iloc[0] == 0


def test_drops_rows_with_missing_lag_or_weather_values():
    start = dt.date(2025, 1, 8)
    # Only 168 + 2 hours of load for 3 target hours -- the last target hour's lags are incomplete
    # relative to a full lookback, but more directly: give weather for only 2 of the 3 hours so
    # the third row has a missing weather value and must be dropped.
    load = _hourly_series(start - dt.timedelta(hours=168), 168 + 3, [1.0] * (168 + 3))
    weather = pd.DataFrame({
        "event_timestamp": pd.date_range(pd.Timestamp(start, tz="UTC"), periods=2, freq="h"),
        "temperature_2m": [1.0, 2.0],
    })

    frame = assemble_training_frame(load, weather, start)

    assert len(frame) == 2


def test_lag_columns_cover_at_least_48h():
    assert all(hours >= 48 for hours in (int(c.removeprefix("load_lag_").removesuffix("h")) for c in LAG_COLUMNS))
