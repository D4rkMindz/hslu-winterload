import lightgbm as lgb
import numpy as np
import pandas as pd
from winterload.forecast.dataset import TARGET_COLUMN


class PreliminaryModel:
    def __init__(self) -> None:
        self._model: lgb.LGBMRegressor = lgb.LGBMRegressor()
        self._feature_columns: list[str] | None = None

    def fit(self, train: pd.DataFrame) -> "PreliminaryModel":
        self._feature_columns = [c for c in train.columns if c not in (TARGET_COLUMN, "event_timestamp")]
        self._model.fit(train[self._feature_columns], train[TARGET_COLUMN])

        return self

    def evaluate(self, data: pd.DataFrame) -> float:
        """Mean absolute percentage error on `data`, as a fraction (0.0615, not 6.15)."""
        if not self._model:
            return 1 # 100% error rate if model was not fitted yet
        predictions = self._model.predict(data[self._feature_columns])
        actual = data[TARGET_COLUMN].to_numpy()
        return float(np.mean(np.abs((actual - predictions) / actual)))
