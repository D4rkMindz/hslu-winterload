import datetime as dt

import numpy as np
import pandas as pd

from winterload.forecast.inference import ForecastRun, ModelForecast
from winterload.repository.forecast import FORECAST_TABLE, RUN_TABLE, ForecastRepository


class AppendStorage:
    def __init__(self):
        self.appends = []

    def append(self, table, df):
        self.appends.append((table, df))
        return table


RUN = ForecastRun("run-1", dt.datetime(2026, 11, 26, 6, tzinfo=dt.UTC), dt.datetime(2026, 11, 26, 6, 1, tzinfo=dt.UTC),
                  dt.date(2026, 11, 27), run_type="manual", kestra_execution_id=None)
HOURS = pd.Series(pd.date_range("2026-11-26 23:00", periods=3, freq="h", tz="UTC"), index=[5, 6, 7])


def test_forecasts_go_first_and_the_run_last():
    storage = AppendStorage()
    written = ForecastRepository(storage).save(RUN, [
        ModelForecast("winterload-load-forecast/5", "champion", HOURS, np.array([1.0, 2.0, 3.0])),
        ModelForecast("seasonal_naive_168h", "baseline", HOURS.iloc[:2], np.array([4.0, 5.0])),
    ])
    assert written == 5
    assert [table for table, _ in storage.appends] == [FORECAST_TABLE, RUN_TABLE]
    forecasts = storage.appends[0][1]
    assert list(forecasts.columns) == ["id", "prediction_time", "model_id", "model_role", "result", "run_id"]
    assert forecasts["id"].is_unique
    assert set(forecasts["run_id"]) == {"run-1"}
    assert list(forecasts["result"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(forecasts["prediction_time"]) == list(HOURS) + list(HOURS.iloc[:2])
    run = storage.appends[1][1]
    assert run.to_dict("records") == [{"id": "run-1", "started_at": RUN.started_at,
                                       "completed_at": RUN.completed_at, "target_day": RUN.target_day,
                                       "run_type": "manual", "kestra_execution_id": None}]


def test_a_run_without_forecasts_still_records_the_run():
    storage = AppendStorage()
    empty = ModelForecast("seasonal_naive_168h", "baseline", HOURS.iloc[:0], np.array([]))
    assert ForecastRepository(storage).save(RUN, [empty]) == 0
    assert [table for table, _ in storage.appends] == [RUN_TABLE]


def _ts(day, hour=0):
    return pd.Timestamp(dt.datetime(2026, 11, day, hour, tzinfo=dt.UTC))


class ReadStorage:
    """Serves prepared frames, filtered on the requested column like Storage.read_appended."""

    def __init__(self, runs, forecasts):
        self.frames = {RUN_TABLE: runs, FORECAST_TABLE: forecasts}

    def read_appended(self, table, column, start, end):
        frame = self.frames[table]
        values = pd.to_datetime(frame[column], utc=True)
        return frame[(values >= pd.Timestamp(start, tz="UTC")) & (values < pd.Timestamp(end, tz="UTC"))]


def _forecasts(*rows):
    return pd.DataFrame(rows, columns=["run_id", "prediction_time", "model_role", "result"])


def test_scheduled_takes_the_newest_scheduled_run_per_hour_and_ignores_manual_runs():
    runs = pd.DataFrame({"id": ["old", "new", "manual"], "run_type": ["scheduled", "scheduled", "manual"],
                         "started_at": [_ts(25, 6), _ts(26, 6), _ts(26, 12)]})
    forecasts = _forecasts(("old", _ts(27, 0), "champion", 1.0), ("new", _ts(27, 0), "champion", 2.0),
                           ("manual", _ts(27, 0), "champion", 9.0), ("old", _ts(27, 1), "champion", 3.0),
                           ("new", _ts(27, 0), "baseline", 7.0))
    result = ForecastRepository(ReadStorage(runs, forecasts)).scheduled(dt.date(2026, 11, 27), dt.date(2026, 11, 28),
                                                                         "champion")
    assert list(result) == [2.0, 3.0]
    assert list(result.index) == [_ts(27, 0), _ts(27, 1)]
    assert result.index.name == "prediction_time"


def test_scheduled_without_a_run_type_column_is_empty():
    runs = pd.DataFrame({"id": ["r"], "started_at": [_ts(26, 6)]})
    forecasts = _forecasts(("r", _ts(27, 0), "champion", 1.0))
    result = ForecastRepository(ReadStorage(runs, forecasts)).scheduled(dt.date(2026, 11, 27), dt.date(2026, 11, 28),
                                                                         "champion")
    assert result.empty
