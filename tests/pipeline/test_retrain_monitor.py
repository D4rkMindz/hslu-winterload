import contextlib
import datetime as dt
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from winterload.forecast.lifecycle import retrain_trigger
from winterload.pipeline.monitoring import retrain_monitor
from winterload.pipeline.monitoring.retrain_monitor import RetrainMonitorPipeline

REFERENCE = dt.date(2027, 6, 8)
START = dt.date(2027, 6, 1)


class FakeRegistry:
    def __init__(self, champion="champion"):
        self.champion = champion
        self.logged = []

    def load_champion(self):
        return self.champion

    @contextlib.contextmanager
    def run(self, name, experiment):
        yield

    def log(self, params=None, metrics=None):
        self.logged.append((params, metrics))


class FakeStorage:
    def read_marker(self, key):
        return None


def _series(values, hours):
    index = pd.date_range("2027-06-01", periods=hours, freq="h", tz="UTC")
    return pd.Series(np.full(hours, values, dtype="float64"), index=index)


@pytest.fixture
def setup(monkeypatch):
    state = SimpleNamespace(registry=FakeRegistry(), challenger=[], decide_args=[])
    hours = 7 * 24

    class Load:
        def __init__(self, storage):
            pass

        def hourly(self, start, end):
            return _series(100.0, hours)

    class Forecasts:
        def __init__(self, storage):
            pass

        def scheduled(self, start, end, role):
            return _series(110.0 if role == "champion" else 200.0, hours)

    class Challenger:
        def __init__(self, **kwargs):
            pass

        def run(self, reference_date):
            state.challenger.append(reference_date)

    real_decide = retrain_trigger.RetrainTriggerPolicy.decide

    def decide(self, model_mape, baseline_mape):
        state.decide_args.append((model_mape, baseline_mape))
        return real_decide(self, model_mape, baseline_mape)

    monkeypatch.setattr(retrain_monitor, "champion_repository", lambda: state.registry)
    monkeypatch.setattr(retrain_monitor, "LoadRepository", Load)
    monkeypatch.setattr(retrain_monitor, "ForecastRepository", Forecasts)
    monkeypatch.setattr(retrain_monitor, "load_dataframe", lambda *a: pd.DataFrame())
    monkeypatch.setattr(retrain_monitor, "report_drift", lambda *a: None)
    monkeypatch.setattr(retrain_monitor, "ChampionChallengerPipeline", Challenger)
    monkeypatch.setattr(retrain_monitor.RetrainTriggerPolicy, "decide", decide)
    state.hours = hours
    return state


def _run():
    RetrainMonitorPipeline(logger=logging.getLogger("t"), storage=FakeStorage()).run(REFERENCE)


def test_skips_with_a_warning_below_the_minimum(setup, monkeypatch, caplog):
    class Short:
        def __init__(self, storage):
            pass

        def scheduled(self, start, end, role):
            return _series(110.0, 24)

    monkeypatch.setattr(retrain_monitor, "ForecastRepository", Short)
    with caplog.at_level(logging.WARNING):
        _run()
    assert "below the minimum" in caplog.text
    assert setup.decide_args == []
    assert setup.registry.logged == []


def test_passes_the_stored_forecast_mapes_to_the_policy(setup):
    _run()
    (model, baseline), = setup.decide_args
    assert model == pytest.approx(0.10)
    assert baseline == pytest.approx(1.0)
    params, metrics = setup.registry.logged[0]
    assert params["retrain"] is False
    assert metrics["model_mape_7d"] == pytest.approx(0.10)
    assert setup.challenger == []


def test_retrains_when_the_model_is_worse_than_the_baseline(setup, monkeypatch):
    class Bad:
        def __init__(self, storage):
            pass

        def scheduled(self, start, end, role):
            return _series(300.0 if role == "champion" else 110.0, setup.hours)

    monkeypatch.setattr(retrain_monitor, "ForecastRepository", Bad)
    _run()
    assert setup.registry.logged[0][0]["mape_trigger"] is True
    assert setup.challenger == [REFERENCE]
