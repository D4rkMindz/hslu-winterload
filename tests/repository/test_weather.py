import datetime as dt

import pandas as pd
from winterload.repository.weather import WeatherRepository


class FakeStorage:
    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def read_range(self, table, start, end):
        self.calls.append((table, start, end))
        return self.tables.get(table, pd.DataFrame())


START = dt.date(2025, 7, 1)
END = dt.date(2025, 7, 3)


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def _frame(rows: list[tuple[str, str, str | None]]) -> pd.DataFrame:
    """Rows of (timestamp, location_id, model)."""
    frame = pd.DataFrame({
        "event_timestamp": [_ts(row[0]) for row in rows],
        "location_id": [row[1] for row in rows],
        "temperature_2m": [float(index) for index in range(len(rows))],
        "run_timestamp": [_ts(row[0]) - pd.Timedelta(14, unit="hours") for row in rows],
    })
    if any(row[2] is not None for row in rows):
        frame["model"] = [row[2] for row in rows]
    return frame


def test_forecast_sorts_by_timestamp_then_location():
    rows = _frame([("2025-07-01 01:00", "b", "ecmwf_ifs"), ("2025-07-01 00:00", "b", "ecmwf_ifs"),
                   ("2025-07-01 01:00", "a", "ecmwf_ifs"), ("2025-07-01 00:00", "a", "ecmwf_ifs")])
    result = WeatherRepository(FakeStorage({"weather_forecast": rows})).forecast(START, END)
    assert list(zip(result["event_timestamp"], result["location_id"])) == [
        (_ts("2025-07-01 00:00"), "a"), (_ts("2025-07-01 00:00"), "b"),
        (_ts("2025-07-01 01:00"), "a"), (_ts("2025-07-01 01:00"), "b")]
    assert list(result.index) == [0, 1, 2, 3]


def test_forecast_reads_the_forecast_table_and_keeps_ecmwf_rows():
    rows = _frame([("2025-07-01 00:00", "a", "ecmwf_ifs"), ("2025-07-01 00:00", "b", "era5")])
    storage = FakeStorage({"weather_forecast": rows})
    result = WeatherRepository(storage).forecast(START, END)
    assert storage.calls == [("weather_forecast", START, END)]
    assert list(result["location_id"]) == ["a"]


def test_short_lead_and_reanalysis_split_the_measurements_table():
    rows = _frame([("2025-07-01 00:00", "a", "ecmwf_ifs"), ("2025-07-01 00:00", "b", "era5"),
                   ("2025-07-01 01:00", "a", "era5")])
    storage = FakeStorage({"weather_measurements": rows})
    repository = WeatherRepository(storage)
    short_lead = repository.short_lead(START, END)
    reanalysis = repository.reanalysis(START, END)
    assert [call[0] for call in storage.calls] == ["weather_measurements", "weather_measurements"]
    assert set(short_lead["model"]) == {"ecmwf_ifs"} and len(short_lead) == 1
    assert set(reanalysis["model"]) == {"era5"} and len(reanalysis) == 2


def test_forecast_00utc_reads_the_backup_table_without_a_model_column():
    rows = _frame([("2025-07-01 00:00", "a", None), ("2025-07-01 01:00", "a", None)])
    storage = FakeStorage({"weather_forecast_backup_00utc": rows})
    result = WeatherRepository(storage).forecast_00utc(START, END)
    assert storage.calls == [("weather_forecast_backup_00utc", START, END)]
    assert len(result) == 2 and "model" not in result.columns


def test_empty_reads_give_empty_frames():
    repository = WeatherRepository(FakeStorage({}))
    assert repository.forecast(START, END).empty
    assert repository.reanalysis(START, END).empty


def test_runs_of_day_keeps_only_rows_inside_the_local_day():
    # 2025-03-30 is the spring clock change in Zurich: the local day is 23:00 UTC on the 29th to 22:00 UTC on the 30th.
    rows = _frame([("2025-03-29 12:00", "a", "ecmwf_ifs"), ("2025-03-29 22:00", "a", "ecmwf_ifs"),
                   ("2025-03-29 23:00", "a", "ecmwf_ifs"), ("2025-03-30 10:00", "a", "ecmwf_ifs"),
                   ("2025-03-30 21:00", "a", "ecmwf_ifs"), ("2025-03-30 22:00", "a", "ecmwf_ifs")])
    runs = WeatherRepository(FakeStorage({"weather_forecast": rows})).runs_of_day(dt.date(2025, 3, 30))
    assert list(runs) == [_ts("2025-03-29 09:00"), _ts("2025-03-29 20:00"), _ts("2025-03-30 07:00")]
