"""The stages against a real MLflow: a throwaway SQLite store in tmp_path, synthetic frames instead of the
feature store."""

import dataclasses
import datetime as dt
import logging

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from winterload.errors import StageError
from winterload.forecast.features.dataset import TARGET_COLUMN
from winterload.forecast.lifecycle.final_test import FinalTestGate, NoTunedSettings
from winterload.forecast.model.preliminary import LightGBMSettings
from winterload.pipeline import run_all
from winterload.pipeline.training import final_test, tuning
from winterload.repository.champion import (
    GIT_SHA_TAG,
    TRAINING_EXPERIMENT,
    TUNING_EXPERIMENT,
    ChampionRepository,
    RunNotFound,
    TrackingUnavailable,
)


class FakeStorage:
    def __init__(self):
        self.markers = {}

    def read_marker(self, key):
        return self.markers.get(key)

    def write_marker(self, key, data):
        self.markers[key] = data


def _frame(start: str, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "event_timestamp": pd.date_range(start, periods=rows, freq="h", tz="UTC"),
        "load_lag_48h": rng.uniform(6000, 9000, size=rows),
        "load_lag_168h": rng.uniform(6000, 9000, size=rows),
    })
    frame[TARGET_COLUMN] = frame["load_lag_48h"] + rng.normal(0, 50, size=rows)
    return frame


@pytest.fixture
def registry(tmp_path, monkeypatch):
    # The SQLite store keeps artifacts under ./mlruns, so run from tmp_path.
    monkeypatch.chdir(tmp_path)
    yield ChampionRepository(f"sqlite:///{tmp_path}/mlflow.db", git_sha="abc123")
    mlflow.end_run()


def test_an_unreachable_server_is_a_stage_error(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")

    with pytest.raises(TrackingUnavailable, match="not reachable"):
        ChampionRepository("http://127.0.0.1:9")


def test_a_stage_error_ends_the_run_with_one_error_line_and_exit_code_1(caplog):
    def failing_stage():
        raise StageError("MLflow is down")

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exit_info:
        run_all([failing_stage])

    assert exit_info.value.code == 1
    assert [record.getMessage() for record in caplog.records] == ["StageError: MLflow is down"]


def test_every_run_carries_the_commit(registry):
    with registry.run("champion_challenger", TRAINING_EXPERIMENT):
        run_id = registry.active_run_id()

    tags = MlflowClient().get_run(run_id).data.tags
    assert tags[GIT_SHA_TAG] == "abc123"
    assert tags["mlflow.source.git.commit"] == "abc123"


def test_a_finished_run_in_the_interval_is_found(registry):
    before = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1)
    with registry.run("champion_challenger", TRAINING_EXPERIMENT):
        run_id = registry.active_run_id()

    assert registry.finished_run_since("champion_challenger", TRAINING_EXPERIMENT, before) == run_id
    assert registry.finished_run_since("champion_challenger", TRAINING_EXPERIMENT,
                                       dt.datetime.now(dt.UTC) + dt.timedelta(minutes=1)) is None
    assert registry.finished_run_since("tuning", TRAINING_EXPERIMENT, before) is None


def test_an_unknown_run_is_a_stage_error(registry):
    with pytest.raises(RunNotFound):
        registry.get_run("0123456789abcdef0123456789abcdef")


def test_tuning_logs_nested_trials_the_gate_and_a_snippet_the_final_test_accepts(registry, monkeypatch):
    train, validation = _frame("2024-03-16", 9000, 0), _frame("2025-07-01", 2016, 1)
    monkeypatch.setattr(tuning, "champion_repository", lambda: registry)
    monkeypatch.setattr(tuning, "load_train_and_validation", lambda storage: (train, validation))

    tuning.TuningPipeline(storage=FakeStorage()).run(trials=3)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(TUNING_EXPERIMENT)
    [parent] = client.search_runs([experiment.experiment_id], "attributes.run_name = 'tuning'")
    trials = client.search_runs([experiment.experiment_id], f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    assert len(trials) == 3
    assert {"validation_mape", "target_mape", "preliminary_validation_mape", "best_fold_mape",
            "baseline_fold_mape"} <= set(parent.data.metrics)
    assert parent.data.params["mode"] == tuning.FIXED_FOLDS
    assert parent.data.tags["gate_passed"] in ("True", "False")

    snippet = mlflow.artifacts.load_text(f"runs:/{parent.info.run_id}/tuned_settings.py")
    namespace = {"LightGBMSettings": LightGBMSettings}
    exec(snippet, namespace)
    assert namespace["TUNING_RUN_ID"] == parent.info.run_id

    # The committed snippet matches the run as MLflow returns it (params as strings). The gate itself
    # depends on the random data, so it is forced open here.
    tracked = registry.get_run(parent.info.run_id)
    FinalTestGate().check(namespace["TUNED_SETTINGS"],
                          dataclasses.replace(tracked, metrics=tracked.metrics | {"validation_mape": 0.0}))


def test_the_final_test_refuses_to_run_without_committed_settings(monkeypatch):
    storage = FakeStorage()
    monkeypatch.setattr(final_test, "TUNED_SETTINGS", None)

    with pytest.raises(NoTunedSettings):
        final_test.FinalTestPipeline(storage=storage).run()
    assert storage.markers == {}


def test_the_final_test_keeps_the_year_closed_when_the_run_is_missing(registry, monkeypatch):
    storage = FakeStorage()
    monkeypatch.setattr(final_test, "TUNED_SETTINGS", LightGBMSettings(100, 0.1, 31, -1, 20))
    monkeypatch.setattr(final_test, "TUNING_RUN_ID", "0123456789abcdef0123456789abcdef")
    monkeypatch.setattr(final_test, "champion_repository", lambda: registry)

    with pytest.raises(RunNotFound):
        final_test.FinalTestPipeline(storage=storage).run()
    assert storage.markers == {}


def test_an_alias_nobody_set_is_a_stage_error(registry):
    from winterload.repository.champion import ModelNotRegistered

    with pytest.raises(ModelNotRegistered, match="challenger"):
        registry.load_model("challenger")


def test_the_challenger_alias_loads_with_its_version(registry):
    from winterload.forecast.model.preliminary import PreliminaryModel

    model = PreliminaryModel().fit(_frame("2026-01-01", 200, seed=1))
    with registry.run("champion_challenger", TRAINING_EXPERIMENT):
        version = registry.register_challenger(model)
    loaded, model_id = registry.load_model("challenger")
    assert model_id == f"winterload-load-forecast/{version}"
    assert loaded.feature_columns == model.feature_columns


def test_after_a_promotion_the_challenger_alias_names_the_champion(registry):
    from winterload.forecast.model.preliminary import PreliminaryModel

    model = PreliminaryModel().fit(_frame("2026-01-01", 200, seed=1))
    with registry.run("champion_challenger", TRAINING_EXPERIMENT):
        serving_version = registry.replace_serving_model(model)
        registry.point_challenger_at(serving_version)
    assert registry.load_model("challenger")[1] == registry.load_model("champion")[1]
