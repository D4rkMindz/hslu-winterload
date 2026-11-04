import datetime as dt
from typing import TYPE_CHECKING

import pandas as pd
from winterload.forecast.features.dataset import local_day_bounds
from winterload.source.openmeteo import FORECAST_MODEL, REANALYSIS_MODEL

if TYPE_CHECKING:
    from winterload.storage import Storage

FORECAST_TABLE = "weather_forecast"
# The 00 UTC runs the store held before it moved to the 18 UTC run of D-2. They have no model column,
# and nothing writes this table any more.
FORECAST_00UTC_TABLE = "weather_forecast_backup_00utc"
MEASUREMENTS_TABLE = "weather_measurements"


def _of_model(rows: pd.DataFrame, model: str) -> pd.DataFrame:
    if rows.empty:
        return rows
    # read_range orders by hour only. A fixed city order keeps later sums equal to the last bit.
    return rows.loc[rows["model"] == model].sort_values(["event_timestamp", "location_id"]).reset_index(drop=True)


class WeatherRepository:
    def __init__(self, storage: "Storage") -> None:
        self._storage = storage

    def forecast(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """The forecast rows the model trains on, one row per hour and city."""
        return _of_model(self._storage.read_range(FORECAST_TABLE, start, end), FORECAST_MODEL)

    def forecast_00utc(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """The 00 UTC run of D-1, which misses 08:00 in summer. It only serves comparisons."""
        return self._storage.read_range(FORECAST_00UTC_TABLE, start, end)

    def short_lead(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """The Historical Forecast API's rows. Each stitches the first hours of model runs together,
        so it tracks the weather closer than any 08:00 forecast can."""
        return _of_model(self._storage.read_range(MEASUREMENTS_TABLE, start, end), FORECAST_MODEL)

    def reanalysis(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """ERA5 rows, the independent truth for scoring forecasts. Only backfilled days have them."""
        return _of_model(self._storage.read_range(MEASUREMENTS_TABLE, start, end), REANALYSIS_MODEL)

    def runs_of_day(self, target_day: dt.date) -> pd.Series:
        """The `run_timestamp` behind each forecast row of a local day. Empty on a gap day."""
        rows = self.forecast(target_day - dt.timedelta(days=1), target_day + dt.timedelta(days=1))
        start, end = local_day_bounds(target_day)
        return rows.loc[(rows["event_timestamp"] >= start) & (rows["event_timestamp"] < end), "run_timestamp"]
