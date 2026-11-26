import datetime as dt

import numpy as np
import pandas as pd
import pytest

from winterload.forecast.features.dataset import (
    LAG_COLUMNS,
    TARGET_COLUMN,
    WEATHER_COLUMNS,
    assemble_inference_frame,
)


def _load(start: str, end: str) -> pd.Series:
    index = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    return pd.Series(np.arange(len(index), dtype="float64"), index=index)


def _weather(target_day: dt.date) -> pd.DataFrame:
    hours = pd.date_range(pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC"),
                          pd.Timestamp(target_day + dt.timedelta(days=1), tz="UTC"), freq="h", inclusive="left")
    return pd.DataFrame({"event_timestamp": hours} | {column: 1.0 for column in WEATHER_COLUMNS})


@pytest.mark.parametrize(("target_day", "hours"), [
    (dt.date(2026, 11, 27), 24),
    (dt.date(2026, 3, 29), 23),   # clocks go forward
    (dt.date(2026, 10, 25), 25),  # clocks go back
])
def test_one_row_per_hour_of_the_local_day(target_day, hours):
    load = _load(str(target_day - dt.timedelta(days=8)), str(target_day + dt.timedelta(days=1)))
    frame = assemble_inference_frame(load, _weather(target_day), target_day)
    assert len(frame) == hours
    assert frame["event_timestamp"].iloc[0] == pd.Timestamp(target_day, tz="Europe/Zurich")
    assert TARGET_COLUMN not in frame.columns
    assert not frame[LAG_COLUMNS + WEATHER_COLUMNS].isna().any().any()


def test_an_hour_without_its_48h_lag_keeps_its_row_with_nan():
    target_day = dt.date(2026, 11, 27)
    # Load ends at 2026-11-25 17:00 UTC, so the 48h lag of 27 Nov from 17:00 UTC on is missing.
    load = _load("2026-11-19", "2026-11-25 17:00")
    load = load.reindex(pd.date_range("2026-11-19", "2026-11-28", freq="h", tz="UTC", inclusive="left"))
    frame = assemble_inference_frame(load, _weather(target_day), target_day)
    missing = frame.loc[frame["load_lag_48h"].isna(), "event_timestamp"]
    assert list(missing) == list(pd.date_range("2026-11-27 17:00", "2026-11-27 22:00", freq="h", tz="UTC"))
    assert not frame["load_lag_168h"].isna().any()
