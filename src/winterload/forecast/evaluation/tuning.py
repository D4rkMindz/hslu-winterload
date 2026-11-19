"""Hyperparameter search for LightGBM: the search space, the fold scoring and the code snippet a tuning run
leaves behind. These functions do no I/O; winterload.pipeline.training.tuning loads the data, runs Optuna and
logs to MLflow.

Trials score on chronological folds inside the data they may see (TRAIN before the final test, everything
before the champion/challenger holdout after it). VALIDATION only scores the best trial, once, so the search
never steers by it (docs/proposal.pdf, section 3.3).
"""

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd

from winterload.forecast.evaluation.feature_study import chronological_folds
from winterload.forecast.features.dataset import TARGET_COLUMN
from winterload.forecast.metrics import mape
from winterload.forecast.model.baseline import seasonal_naive_forecast
from winterload.forecast.model.preliminary import PRELIMINARY_SETTINGS, LightGBMSettings, PreliminaryModel

SEARCH_SEED = 42
DEFAULT_TRIALS = 100
# Bagging needs subsample_freq > 0. With subsample=1.0 it draws every row, so the defaults stay reachable.
SUBSAMPLE_FREQ = 1
# The first trial: LightGBM's defaults, i.e. the preliminary model. reg_lambda's log scale can't hold 0,
# so it starts at the lower bound, which changes next to nothing.
DEFAULT_PARAMS = {
    "n_estimators": PRELIMINARY_SETTINGS.n_estimators,
    "learning_rate": PRELIMINARY_SETTINGS.learning_rate,
    "num_leaves": PRELIMINARY_SETTINGS.num_leaves,
    "min_child_samples": PRELIMINARY_SETTINGS.min_child_samples,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1e-3,
}


class Trial(Protocol):
    """The part of optuna.Trial this module uses, so tests can pass a stub."""

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int: ...

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float: ...


def suggest_settings(trial: Trial) -> LightGBMSettings:
    return LightGBMSettings(
        n_estimators=trial.suggest_int("n_estimators", 100, 2000, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        max_depth=-1,
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=SUBSAMPLE_FREQ,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def settings_from_params(params: dict[str, Any]) -> LightGBMSettings:
    """Rebuilds the settings of a trial from its params (optuna's best_params)."""
    return LightGBMSettings(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=-1,
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        subsample_freq=SUBSAMPLE_FREQ,
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
    )


def settings_params(settings: LightGBMSettings) -> dict[str, Any]:
    """The settings as flat params, as tuning logs them with a `best_` prefix and final_test compares them."""
    return {field.name: getattr(settings, field.name) for field in dataclasses.fields(settings)}


def score_folds(settings: LightGBMSettings, folds: list[tuple[pd.DataFrame, pd.DataFrame]],
                report: Callable[[int, float], None] | None = None) -> list[float]:
    """The validation MAPE of each fold. After each fold, `report(fold, mean so far)` may raise to stop
    early (optuna's pruning)."""
    fold_mapes: list[float] = []
    for fold, (train, validation) in enumerate(folds):
        fold_mapes.append(PreliminaryModel(settings).fit(train).evaluate(validation))
        if report is not None:
            report(fold, float(np.mean(fold_mapes)))
    return fold_mapes


def baseline_fold_mapes(folds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> list[float]:
    """The seasonal-naive MAPE on each fold's validation rows, the bar every trial is reported against."""
    return [mape(validation[TARGET_COLUMN].to_numpy(), seasonal_naive_forecast(validation))
            for _, validation in folds]


def tuning_folds(history: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """The six 8-week folds at the end of `history` (see feature_study.chronological_folds)."""
    return chronological_folds(history.sort_values("event_timestamp", kind="stable").reset_index(drop=True))


def tuned_settings_snippet(settings: LightGBMSettings, run_id: str) -> str:
    """The block to paste into winterload/forecast/model/tuned.py, one setting per line so ruff accepts it."""
    fields = "".join(f"    {name}={value!r},\n" for name, value in settings_params(settings).items())
    return (f"TUNED_SETTINGS: LightGBMSettings | None = LightGBMSettings(\n{fields})\n"
            f"TUNING_RUN_ID: str | None = {run_id!r}\n")
