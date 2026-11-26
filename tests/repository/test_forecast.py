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
                  dt.date(2026, 11, 27))
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
                                       "completed_at": RUN.completed_at, "target_day": RUN.target_day}]


def test_a_run_without_forecasts_still_records_the_run():
    storage = AppendStorage()
    empty = ModelForecast("seasonal_naive_168h", "baseline", HOURS.iloc[:0], np.array([]))
    assert ForecastRepository(storage).save(RUN, [empty]) == 0
    assert [table for table, _ in storage.appends] == [RUN_TABLE]
