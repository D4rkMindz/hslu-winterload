"""The champion on the MLflow Model Registry. winterload.forecast.lifecycle.champion says what the
two aliases mean. Run tracking lives here too, because `promote` logs its models into the active run.
"""

import datetime as dt
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from winterload.forecast.lifecycle.champion import ChampionRecord
from winterload.forecast.model.preliminary import PreliminaryModel

EXPERIMENT_NAME = "winterload"
REGISTERED_MODEL_NAME = "winterload-load-forecast"
EVALUATION_ALIAS = "champion_eval"
SERVING_ALIAS = "champion"
# MLflow's error codes for an unknown alias and an unknown registered model.
NOT_REGISTERED_YET = ("INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST")
SKOPS_TRUSTED_TYPES = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMRegressor"]


class ChampionRepository:
    def __init__(self, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        self._client = MlflowClient()

    def run(self, name: str) -> mlflow.ActiveRun:
        """Use as a context manager around `log`, `promote` and `replace_serving_model`."""
        return mlflow.start_run(run_name=name)

    def log(self, params: dict[str, Any], metrics: dict[str, float]) -> None:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

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
        )

    def load_serving_model(self) -> PreliminaryModel:
        return self._load(SERVING_ALIAS)

    def promote(self, evaluation_model: PreliminaryModel, serving_model: PreliminaryModel,
                holdout_mape: float, promoted_at: dt.datetime) -> None:
        version = self._register(evaluation_model, "evaluation_model")
        self._client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "holdout_mape", str(holdout_mape))
        self._client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "promoted_at", promoted_at.isoformat())
        self._client.set_registered_model_alias(REGISTERED_MODEL_NAME, EVALUATION_ALIAS, version)
        self.replace_serving_model(serving_model)

    def replace_serving_model(self, serving_model: PreliminaryModel) -> None:
        version = self._register(serving_model, "serving_model")
        self._client.set_registered_model_alias(REGISTERED_MODEL_NAME, SERVING_ALIAS, version)

    def _register(self, model: PreliminaryModel, name: str) -> str:
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
