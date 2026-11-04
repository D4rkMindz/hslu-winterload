import datetime as dt
from typing import TYPE_CHECKING

import pandas as pd
from winterload.forecast.features.dataset import TARGET_COLUMN

if TYPE_CHECKING:
    from winterload.storage import Storage

ENERGY_LOAD_TABLE = "energy_load"


class LoadRepository:
    def __init__(self, storage: "Storage") -> None:
        self._storage = storage

    def hourly(self, start: dt.date, end: dt.date) -> pd.Series:
        """One value per UTC hour in `[start, end)`, NaN where no reading exists."""
        load = self._storage.read_range(ENERGY_LOAD_TABLE, start, end)
        load = load.groupby("event_timestamp", as_index=False)[TARGET_COLUMN].mean()
        series = load.set_index("event_timestamp")[TARGET_COLUMN].sort_index()
        full_index = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="h",
                                   inclusive="left")
        return series.reindex(full_index)
