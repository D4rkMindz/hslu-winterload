"""The forecasts of winterload.pipeline.inference, append-only: a rerun adds rows and never replaces any.
`winterload_forecasts` holds one row per model and hour, `winterload_forecast_runs` one row per run. On BigQuery
the view `winterload_forecasts_latest` (see BigQueryStorage.setup) holds the newest result per model and hour.
"""

import uuid
from typing import TYPE_CHECKING

import pandas as pd

from winterload.forecast.inference import ForecastRun, ModelForecast

if TYPE_CHECKING:
    from winterload.storage import Storage

FORECAST_TABLE = "winterload_forecasts"
RUN_TABLE = "winterload_forecast_runs"


class ForecastRepository:
    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    def save(self, run: ForecastRun, forecasts: list[ModelForecast]) -> int:
        """Appends the forecasts, then the run. The run goes last: the latest view joins the two, so it never
        shows the rows of a run that broke off halfway. Returns the number of forecast rows written."""
        rows = [self._rows(run, forecast) for forecast in forecasts if len(forecast.result)]
        if rows:
            self._storage.append(FORECAST_TABLE, pd.concat(rows, ignore_index=True))
        self._storage.append(RUN_TABLE, pd.DataFrame([{
            "id": run.id, "started_at": run.started_at, "completed_at": run.completed_at,
            "target_day": run.target_day,
        }]))
        return sum(len(frame) for frame in rows)

    @staticmethod
    def _rows(run: ForecastRun, forecast: ModelForecast) -> pd.DataFrame:
        return pd.DataFrame({
            "id": [str(uuid.uuid4()) for _ in range(len(forecast.result))],
            "prediction_time": forecast.prediction_time.reset_index(drop=True),
            "model_id": forecast.model_id,
            "model_role": forecast.model_role,
            "result": forecast.result.astype("float64"),
            "run_id": run.id,
        })
