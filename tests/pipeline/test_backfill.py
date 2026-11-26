import datetime as dt
import logging

import pandas as pd
import pytest

from winterload.pipeline.features import DaysRejected, energy_load_backfill
from winterload.pipeline.features.energy_load_backfill import EnergyLoadBackfill


class FakeStorage:
    def __init__(self):
        self.written = []

    def day_exists(self, table, day):
        return False

    def write_day(self, table, df, day):
        self.written.append((day, df))
        return "fake"


def _run(monkeypatch, day, hours):
    records = [{"timestamp": pd.Timestamp(day) + pd.Timedelta(hours=h), "bidding_zone": "CH",
                "quantity": 6000.0, "unit": "MW"} for h in range(hours)]

    class FakeEntsoE:
        def get_load(self, start_date, end_date):
            return records

    monkeypatch.setattr(energy_load_backfill, "EntsoE", FakeEntsoE)
    storage = FakeStorage()
    EnergyLoadBackfill(logger=logging.getLogger("test"), storage=storage).run(day, day)
    return storage


def _days_back(n):
    return dt.datetime.now(dt.UTC).date() - dt.timedelta(days=n)


def test_a_partial_final_day_raises_but_is_still_written(monkeypatch):
    with pytest.raises(DaysRejected):
        _run(monkeypatch, _days_back(5), 10)


def test_a_partial_day_from_yesterday_passes(monkeypatch):
    assert len(_run(monkeypatch, _days_back(1), 10).written) == 1


def test_a_complete_final_day_passes(monkeypatch):
    assert len(_run(monkeypatch, _days_back(5), 24).written) == 1
