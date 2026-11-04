import datetime as dt

import numpy as np
import pandas as pd
from winterload.repository.load import LoadRepository


class FakeStorage:
    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def read_range(self, table, start, end):
        self.calls.append((table, start, end))
        return self.tables.get(table, pd.DataFrame())


def test_hourly_averages_duplicates_and_fills_gaps_with_nan():
    start, end = dt.date(2025, 7, 1), dt.date(2025, 7, 2)
    rows = pd.DataFrame({
        "event_timestamp": pd.to_datetime(["2025-07-01 00:00", "2025-07-01 00:00", "2025-07-01 02:00"], utc=True),
        "quantity": [100.0, 200.0, 300.0],
    })
    storage = FakeStorage({"energy_load": rows})
    series = LoadRepository(storage).hourly(start, end)
    assert storage.calls == [("energy_load", start, end)]
    assert len(series) == 24
    assert series.index[0] == pd.Timestamp("2025-07-01 00:00", tz="UTC")
    assert series.index[-1] == pd.Timestamp("2025-07-01 23:00", tz="UTC")
    assert series.iloc[0] == 150.0
    assert np.isnan(series.iloc[1])
    assert series.iloc[2] == 300.0
    assert series.isna().sum() == 22
