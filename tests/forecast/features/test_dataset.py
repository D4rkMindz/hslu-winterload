import datetime as dt

import numpy as np
import pandas as pd
import pytest
from winterload.forecast.evaluation.backtest import WEATHER_RUN_DELAY, expected_target_hours, issue_time
from winterload.forecast.features.dataset import EQUAL_CITY_WEIGHTS, LAG_COLUMNS, POPULATION_CITY_WEIGHTS, \
    TARGET_COLUMN, WEATHER_COLUMNS, aggregate_weather, assemble_training_frame, local_day_bounds, weather_run
from winterload.source.openmeteo import HOURLY_VARIABLES


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


def test_local_day_bounds_handle_clock_changes():
    for day, hours in [(dt.date(2025, 7, 2), 24), (dt.date(2025, 3, 30), 23), (dt.date(2024, 10, 27), 25)]:
        start, end = local_day_bounds(day)

        assert end - start == pd.Timedelta(hours, unit="hours")


def test_weather_columns_are_the_open_meteo_variables():
    assert WEATHER_COLUMNS == HOURLY_VARIABLES


def test_weather_run_is_two_days_before_the_target():
    assert weather_run(dt.date(2025, 7, 2)) == dt.datetime(2025, 6, 30, 18)
    assert weather_run(dt.date(2024, 5, 10)) == dt.datetime(2024, 5, 8, 12)


def test_weather_run_switches_to_18_utc_at_the_archive_boundary():
    assert weather_run(dt.date(2024, 8, 7)) == dt.datetime(2024, 8, 5, 12)
    assert weather_run(dt.date(2024, 8, 8)) == dt.datetime(2024, 8, 6, 18)


@pytest.mark.parametrize("day", [dt.date(2025, 7, 2), dt.date(2025, 1, 15), dt.date(2025, 3, 30), dt.date(2025, 10, 26)])
def test_target_hours_lie_28_to_52_hours_after_the_run(day):
    run = pd.Timestamp(weather_run(day), tz="UTC")

    lead = (expected_target_hours(day) - run) / pd.Timedelta(1, unit="hours")

    assert lead.min() >= 28
    assert lead.max() <= 52


def test_the_run_is_published_before_every_issue_time():
    day = dt.date(2024, 3, 16)
    while day <= dt.date(2026, 10, 31):
        assert pd.Timestamp(weather_run(day), tz="UTC") + WEATHER_RUN_DELAY <= issue_time(day), day
        day += dt.timedelta(days=1)


def _two_city_weather(temperatures: dict[int, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for hour, (a, b) in temperatures.items():
        for city, value in (("a", a), ("b", b)):
            row = {column: 0.0 for column in WEATHER_COLUMNS}
            row.update(event_timestamp=pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hour, unit="hours"),
                       location_id=city, temperature_2m=value)
            rows.append(row)
    return pd.DataFrame(rows)


def test_aggregate_weather_weights_the_cities():
    weather = _two_city_weather({1: (10.0, 20.0)})

    result = aggregate_weather(weather, {"a": 0.75, "b": 0.25})

    assert result["temperature_2m"].iloc[0] == pytest.approx(12.5)


def test_aggregate_weather_renormalizes_over_missing_cities():
    weather = _two_city_weather({2: (10.0, np.nan), 3: (np.nan, np.nan)})

    result = aggregate_weather(weather, {"a": 0.75, "b": 0.25}).set_index("event_timestamp")["temperature_2m"]

    assert result.iloc[0] == pytest.approx(10.0)
    assert np.isnan(result.iloc[1])


def test_city_weights_sum_to_one_over_the_same_cities():
    assert sum(EQUAL_CITY_WEIGHTS.values()) == pytest.approx(1)
    assert set(EQUAL_CITY_WEIGHTS) == set(POPULATION_CITY_WEIGHTS)


def test_holiday_share_follows_the_local_date():
    start = dt.date(2025, 7, 31)
    load = _hourly_series(start - dt.timedelta(hours=168), 168 + 48, [1.0] * (168 + 48))
    hours = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=48, freq="h")
    weather = pd.DataFrame({"event_timestamp": hours, **{column: 1.0 for column in WEATHER_COLUMNS}})

    frame = assemble_training_frame(load, weather, start).set_index("event_timestamp")

    assert frame.loc[pd.Timestamp("2025-07-31 21:00", tz="UTC"), "holiday_share_ch"] == 0.0
    assert frame.loc[pd.Timestamp("2025-07-31 22:00", tz="UTC"), "holiday_share_ch"] == pytest.approx(1.0)
    assert "is_holiday_ch" not in frame.columns
