import datetime as dt
import logging

import numpy as np
import pandas as pd
import pytest

from winterload.pipeline import inference
from winterload.pipeline.inference import InferenceIncomplete, InferencePipeline
from winterload.repository.champion import ModelNotRegistered
from winterload.repository.forecast import FORECAST_TABLE, RUN_TABLE

TARGET_DAY = dt.date(2026, 11, 27)
HOURS = pd.date_range("2026-11-26 23:00", periods=4, freq="h", tz="UTC")


class AppendStorage:
    def __init__(self):
        self.appends = []

    def append(self, table, df):
        self.appends.append((table, df))
        return table


class FakeModel:
    feature_columns = ["load_lag_48h", "load_lag_168h"]

    def predict(self, frame):
        return frame["load_lag_48h"].to_numpy() + 1


class FakeRegistry:
    def __init__(self, registered):
        self.registered = registered

    def load_model(self, alias):
        if alias not in self.registered:
            raise ModelNotRegistered(f"no alias {alias!r}")
        return FakeModel(), f"winterload-load-forecast/{self.registered[alias]}"


@pytest.fixture
def frame(monkeypatch):
    frame = pd.DataFrame({"event_timestamp": HOURS, "load_lag_48h": [1.0, 2.0, np.nan, np.nan],
                          "load_lag_168h": [10.0, 20.0, 30.0, 40.0]})
    monkeypatch.setattr(inference, "load_inference_frame", lambda storage, target_day: frame)
    return frame


def _pipeline(storage):
    return InferencePipeline(logger=logging.getLogger("test"), storage=storage)


def test_a_baseline_run_needs_no_registry_and_writes_every_hour(frame, monkeypatch):
    def no_registry():
        raise AssertionError("baseline must not touch MLflow")
    monkeypatch.setattr(inference, "champion_repository", no_registry)
    storage = AppendStorage()
    _pipeline(storage).run(TARGET_DAY, ["baseline"])
    forecasts = storage.appends[0][1]
    assert list(forecasts["result"]) == [10.0, 20.0, 30.0, 40.0]
    assert set(forecasts["model_id"]) == {"seasonal_naive_168h"}
    assert storage.appends[1][0] == RUN_TABLE
    assert storage.appends[1][1]["target_day"].iloc[0] == TARGET_DAY


def test_skipped_hours_are_stored_as_far_as_possible_and_fail_the_stage(frame, monkeypatch, caplog):
    monkeypatch.setattr(inference, "champion_repository", lambda: FakeRegistry({"champion": "5"}))
    storage = AppendStorage()
    with caplog.at_level(logging.WARNING), pytest.raises(InferenceIncomplete, match="champion skipped 2 hour"):
        _pipeline(storage).run(TARGET_DAY, ["champion", "baseline"])
    forecasts = storage.appends[0][1]
    champion = forecasts[forecasts["model_role"] == "champion"]
    assert list(champion["prediction_time"]) == list(HOURS[:2])
    assert list(champion["result"]) == [2.0, 3.0]
    assert set(champion["model_id"]) == {"winterload-load-forecast/5"}
    assert len(forecasts[forecasts["model_role"] == "baseline"]) == 4
    assert HOURS[2].isoformat() in caplog.text and HOURS[3].isoformat() in caplog.text


def test_a_model_that_fails_to_load_doesnt_stop_the_others(frame, monkeypatch):
    monkeypatch.setattr(inference, "champion_repository", lambda: FakeRegistry({}))
    storage = AppendStorage()
    with pytest.raises(InferenceIncomplete, match="challenger failed"):
        _pipeline(storage).run(TARGET_DAY, ["challenger", "baseline"])
    assert [table for table, _ in storage.appends] == [FORECAST_TABLE, RUN_TABLE]
    assert set(storage.appends[0][1]["model_role"]) == {"baseline"}


def test_a_model_that_wants_a_column_the_frame_lacks_fails_alone(frame, monkeypatch):
    class NewFeatureModel(FakeModel):
        feature_columns = ["load_lag_48h", "a_feature_the_frame_lacks"]

    class NewFeatureRegistry(FakeRegistry):
        def load_model(self, alias):
            return NewFeatureModel(), "winterload-load-forecast/9"

    monkeypatch.setattr(inference, "champion_repository", lambda: NewFeatureRegistry({}))
    storage = AppendStorage()
    with pytest.raises(InferenceIncomplete, match=r"champion failed \(KeyError\)"):
        _pipeline(storage).run(TARGET_DAY, ["champion", "baseline"])
    assert set(storage.appends[0][1]["model_role"]) == {"baseline"}


def test_a_scheduled_run_stores_its_type_and_kestra_execution_id(frame):
    storage = AppendStorage()
    _pipeline(storage).run(TARGET_DAY, ["baseline"], scheduled_at="2026-11-26T06:00:00Z", kestra_execution_id="abc")
    run = storage.appends[1][1].iloc[0]
    assert run["run_type"] == "scheduled"
    assert run["kestra_execution_id"] == "abc"


def test_a_run_without_schedule_or_execution_id_is_manual(frame):
    storage = AppendStorage()
    _pipeline(storage).run(TARGET_DAY, ["baseline"])
    run = storage.appends[1][1].iloc[0]
    assert run["run_type"] == "manual"
    assert run["kestra_execution_id"] is None
