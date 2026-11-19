from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone

from winterload.forecast.features.dataset import TARGET_COLUMN
from winterload.forecast.metrics import mape


class ModelNotFitted(RuntimeError):
    """Raised when a caller asks an unfitted model for a forecast."""


@dataclass(frozen=True)
class LightGBMSettings:
    """The settings reviewer point A7 asks to document. `num_leaves`, `max_depth` and
    `min_child_samples` set the tree complexity."""
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    early_stopping_rounds: int | None = None
    # Row and column sampling and L2 regularisation. The defaults are LightGBM's, which switch all three off.
    # winterload.forecast.evaluation.tuning searches them.
    subsample: float = 1.0
    subsample_freq: int = 0
    colsample_bytree: float = 1.0
    reg_lambda: float = 0.0


# LightGBM's defaults, written out. The preliminary model stays untuned because the B1/A4 target
# measures tuning against it.
PRELIMINARY_SETTINGS = LightGBMSettings(n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1,
                                        min_child_samples=20)
# A candidate for tuning, compared in notebooks/training/feature_study.ipynb.
EARLY_STOPPING_SETTINGS = LightGBMSettings(n_estimators=2000, learning_rate=0.05, num_leaves=31, max_depth=-1,
                                           min_child_samples=20, early_stopping_rounds=50)
# Early stopping watches the newest rows of the training data, so it never reads validation rows.
EARLY_STOPPING_SHARE = 0.1
# LightGBM starts one thread per core by default. On ~10,000 rows the threads mostly wait for each other: on a
# busy 16-core host, 2000 trees with 255 leaves took 403 s with 16 threads and 4.4 s with 4.
THREADS = 4


class PreliminaryModel:
    _model: LGBMRegressor

    def __init__(self, settings: LightGBMSettings = PRELIMINARY_SETTINGS) -> None:
        self._settings = settings
        self._model: lgb.LGBMRegressor = lgb.LGBMRegressor(
            n_estimators=settings.n_estimators,
            learning_rate=settings.learning_rate,
            num_leaves=settings.num_leaves,
            max_depth=settings.max_depth,
            min_child_samples=settings.min_child_samples,
            subsample=settings.subsample,
            subsample_freq=settings.subsample_freq,
            colsample_bytree=settings.colsample_bytree,
            reg_lambda=settings.reg_lambda,
            n_jobs=THREADS,
            random_state=0,
            verbose=-1,
        )
        self._feature_columns: list[str] | None = None

    def unfitted_copy(self) -> PreliminaryModel:
        """A new, unfitted model with this model's estimator settings, for example to refit a champion
        loaded from the registry on more data."""
        model = PreliminaryModel(self._settings)
        model._model = clone(self._model)
        return model

    @classmethod
    def from_estimator(cls, estimator: lgb.LGBMRegressor) -> PreliminaryModel:
        """Wraps a fitted estimator, for example one loaded from the model registry."""
        model = cls()
        model._model = estimator
        model._feature_columns = list(estimator.feature_name_)
        return model

    @property
    def estimator(self) -> lgb.LGBMRegressor:
        return self._model

    def fit(self, train: pd.DataFrame) -> PreliminaryModel:
        self._feature_columns = [c for c in train.columns if c not in (TARGET_COLUMN, "event_timestamp")]
        features, target = train[self._feature_columns], train[TARGET_COLUMN]
        if self._settings.early_stopping_rounds is None:
            self._model.fit(features, target)
            return self

        cutoff = int(len(train) * (1 - EARLY_STOPPING_SHARE))
        self._model.fit(
            features.iloc[:cutoff], target.iloc[:cutoff],
            eval_X=features.iloc[cutoff:], eval_y=target.iloc[cutoff:],
            callbacks=[lgb.early_stopping(self._settings.early_stopping_rounds, verbose=False)],
        )
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self._feature_columns is None:
            raise ModelNotFitted("Call fit() before predict() or evaluate().")
        return self._model.predict(data[self._feature_columns])

    def evaluate(self, data: pd.DataFrame) -> float:
        """Mean absolute percentage error on `data`, as a fraction (0.0615, not 6.15)."""
        return mape(data[TARGET_COLUMN].to_numpy(), self.predict(data))
