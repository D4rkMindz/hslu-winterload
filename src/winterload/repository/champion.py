"""The champion on the MLflow Model Registry, and run tracking. winterload.forecast.lifecycle.champion says what
the two aliases mean. Run tracking lives here too, because `promote` logs its models into the active run.

Each stage logs to its own experiment (TRAINING_EXPERIMENT, ...), and every run carries the commit of the
code that ran it. MLFLOW_TRACKING_URI picks the server: the local docker compose MLflow, ml.darker.dev, or a
proxy in front of it on atc.darker.dev and note.darker.dev.
"""

import contextlib
import datetime as dt
import json
import sys
import warnings
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from winterload.errors import StageError
from winterload.forecast.features.dataset import TARGET_COLUMN
from winterload.forecast.lifecycle.champion import ChampionRecord
from winterload.forecast.lifecycle.final_test import TrackedRun
from winterload.forecast.model.preliminary import PreliminaryModel

# Logged datasets only record their schema, never serve as a model signature, so MLflow's hint about integer
# columns that can't hold missing values doesn't apply. It would show up as a warning in every Kestra run.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column", category=UserWarning)

TRAINING_EXPERIMENT = "winterload/training"
TUNING_EXPERIMENT = "winterload/tuning"
MONITORING_EXPERIMENT = "winterload/monitoring"
STUDIES_EXPERIMENT = "winterload/studies"

REGISTERED_MODEL_NAME = "winterload-load-forecast"
EVALUATION_ALIAS = "champion_eval"
SERVING_ALIAS = "champion"
# MLflow's error codes for an unknown alias and an unknown registered model.
NOT_REGISTERED_YET = ("INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST")
SKOPS_TRUSTED_TYPES = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMRegressor"]
GIT_SHA_TAG = "git_sha"
# The tag the MLflow UI shows as the run's source commit
SOURCE_COMMIT_TAG = "mlflow.source.git.commit"
# On champion_eval versions: the end of the data it trained on, the drift report's reference
TRAINING_END_TAG = "training_end"


class TrackingUnavailable(StageError):
    """The MLflow server can't be reached, or refuses the credentials."""


class RunNotFound(StageError):
    """The MLflow server is up, but holds no run with that ID."""


class ChampionRepository:
    def __init__(self, tracking_uri: str, git_sha: str = "local") -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._tracking_uri = tracking_uri
        self._client = MlflowClient()
        self._git_sha = git_sha
        self._check_reachable()

    def _check_reachable(self) -> None:
        try:
            self._client.search_experiments(max_results=1)
        except (MlflowException, OSError) as error:
            raise TrackingUnavailable(
                f"MLflow at {self._tracking_uri} is not reachable ({type(error).__name__}: {error}). Start it "
                "(`docker compose ... up -d mlflow`) or check MLFLOW_TRACKING_URI and, for ml.darker.dev, "
                "MLFLOW_TRACKING_USERNAME/MLFLOW_TRACKING_PASSWORD."
            ) from error

    def run(self, name: str, experiment: str, nested: bool = False) -> mlflow.ActiveRun:
        """Use as a context manager around `log`, `promote` and `replace_serving_model`. A nested run stays in
        its parent's experiment."""
        if not nested:
            mlflow.set_experiment(experiment)
        tags = {GIT_SHA_TAG: self._git_sha}
        if self._git_sha != "local":
            tags[SOURCE_COMMIT_TAG] = self._git_sha
        return mlflow.start_run(run_name=name, nested=nested, tags=tags)

    @staticmethod
    def active_run_id() -> str:
        return mlflow.active_run().info.run_id

    def log(self, params: dict[str, Any], metrics: dict[str, float],
            datasets: dict[str, pd.DataFrame] | None = None) -> None:
        """`datasets` shows up in the run's Datasets: name, row count, schema and digest per frame, with the
        name as its context (train, validation, ...). The rows themselves stay in the offline store."""
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for name, frame in (datasets or {}).items():
            mlflow.log_input(mlflow.data.from_pandas(frame, name=name, targets=TARGET_COLUMN), context=name)

    def tag(self, tags: dict[str, Any]) -> None:
        mlflow.set_tags({key: str(value) for key, value in tags.items()})

    def log_text(self, text: str, artifact_file: str) -> None:
        mlflow.log_text(text, artifact_file)

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        # default=str: Evidently's dicts hold numpy numbers and dates
        mlflow.log_text(json.dumps(data, indent=2, default=str), artifact_file)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def log_unregistered_model(self, model: PreliminaryModel, name: str) -> None:
        mlflow.lightgbm.log_model(model.estimator, name=name, serialization_format="skops",
                                  skops_trusted_types=SKOPS_TRUSTED_TYPES)

    def get_run(self, run_id: str) -> TrackedRun:
        try:
            run = self._client.get_run(run_id)
        except MlflowException as error:
            if error.error_code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
                raise RunNotFound(f"MLflow at {self._tracking_uri} has no run {run_id!r}.") from error
            raise
        return TrackedRun(run_id=run.info.run_id, params=dict(run.data.params), metrics=dict(run.data.metrics),
                          tags=dict(run.data.tags))

    def finished_run_since(self, name: str, experiment: str, since: dt.datetime) -> str | None:
        """The ID of a finished run called `name` that started at or after `since`, or None."""
        found = self._client.get_experiment_by_name(experiment)
        if found is None:
            return None
        runs = self._client.search_runs(
            [found.experiment_id],
            filter_string=f"attributes.run_name = '{name}' AND attributes.status = 'FINISHED' "
                          f"AND attributes.start_time >= {int(since.timestamp() * 1000)}",
            max_results=1,
        )
        return runs[0].info.run_id if runs else None

    def load_champion(self) -> ChampionRecord | None:
        """The current champion, or None if no run has promoted one yet."""
        try:
            version = self._client.get_model_version_by_alias(REGISTERED_MODEL_NAME, EVALUATION_ALIAS)
        except MlflowException as error:
            if error.error_code in NOT_REGISTERED_YET:
                return None
            raise
        return ChampionRecord(
            evaluation_model=self._load(EVALUATION_ALIAS),
            holdout_mape=float(version.tags["holdout_mape"]),
            promoted_at=version.tags["promoted_at"],
            training_end=self._training_end(version),
        )

    def _training_end(self, version: Any) -> dt.date | None:
        """Versions promoted before the tag existed: the promoting run's holdout start."""
        if TRAINING_END_TAG in version.tags:
            return dt.date.fromisoformat(version.tags[TRAINING_END_TAG])
        if version.run_id is None:
            return None
        holdout_start = self._client.get_run(version.run_id).data.params.get("holdout_start")
        return dt.date.fromisoformat(holdout_start) if holdout_start else None

    def load_serving_model(self) -> PreliminaryModel:
        return self._load(SERVING_ALIAS)

    def promote(self, evaluation_model: PreliminaryModel, serving_model: PreliminaryModel,
                holdout_mape: float, promoted_at: dt.datetime, training_end: dt.date) -> None:
        version = self._register(evaluation_model, "evaluation_model")
        self._client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "holdout_mape", str(holdout_mape))
        self._client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "promoted_at", promoted_at.isoformat())
        self._client.set_model_version_tag(REGISTERED_MODEL_NAME, version, TRAINING_END_TAG, training_end.isoformat())
        self._client.set_registered_model_alias(REGISTERED_MODEL_NAME, EVALUATION_ALIAS, version)
        self.replace_serving_model(serving_model)

    def replace_serving_model(self, serving_model: PreliminaryModel) -> None:
        version = self._register(serving_model, "serving_model")
        self._client.set_registered_model_alias(REGISTERED_MODEL_NAME, SERVING_ALIAS, version)

    def _register(self, model: PreliminaryModel, name: str) -> str:
        # MLflow prints its registration notices ("Created version '3' ...") straight to sys.stderr, not through
        # logging, so Kestra would show them as errors. Logging handlers keep the stderr they were created with.
        with contextlib.redirect_stderr(sys.stdout):
            info = mlflow.lightgbm.log_model(
                model.estimator,
                name=name,
                registered_model_name=REGISTERED_MODEL_NAME,
                serialization_format="skops",
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
        return str(info.registered_model_version)

    def _load(self, alias: str) -> PreliminaryModel:
        return PreliminaryModel.from_estimator(mlflow.lightgbm.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}"))
