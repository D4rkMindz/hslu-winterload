"""Compares feature groups and model settings on chronological folds, reviewer point A7. These
functions do no I/O. notebooks/training/feature_study.ipynb loads the data and writes the results.

The folds must stay before the final test year, so callers pass rows up to VALIDATION_END.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from winterload.forecast.features.dataset import CALENDAR_COLUMNS, LAG_COLUMNS, TARGET_COLUMN, WEATHER_COLUMNS
from winterload.forecast.metrics import mae, mape
from winterload.forecast.model.preliminary import PRELIMINARY_SETTINGS, LightGBMSettings, PreliminaryModel

FEATURE_GROUPS = {"load": LAG_COLUMNS, "calendar": CALENDAR_COLUMNS, "weather": WEATHER_COLUMNS}
# Six 8-week folds cover the last 48 weeks before the test year, so every season validates once.
FOLD_COUNT = 6
FOLD_HOURS = 8 * 7 * 24


def feature_sets() -> dict[str, list[str]]:
    """Every combination of the feature groups, named like `load+calendar`."""
    names = list(FEATURE_GROUPS)
    return {
        "+".join(combo): [column for name in combo for column in FEATURE_GROUPS[name]]
        for size in range(1, len(names) + 1)
        for combo in combinations(names, size)
    }


def chronological_folds(frame: pd.DataFrame, fold_count: int = FOLD_COUNT,
                        fold_hours: int = FOLD_HOURS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """(train, validation) pairs from rows sorted by time. Each validation block follows its training rows."""
    splitter = TimeSeriesSplit(n_splits=fold_count, test_size=fold_hours)
    return [(frame.iloc[train], frame.iloc[validation]) for train, validation in splitter.split(frame)]


def out_of_fold_forecast(frame: pd.DataFrame, columns: list[str] | None = None,
                         settings: LightGBMSettings = PRELIMINARY_SETTINGS) -> pd.DataFrame:
    """The validation rows of every fold, with `fold` and `forecast` columns. The model trains on
    `columns`, or on every column if None."""
    scored = []
    for fold, (train, validation) in enumerate(chronological_folds(frame)):
        selected = train if columns is None else train[["event_timestamp", TARGET_COLUMN, *columns]]
        forecast = PreliminaryModel(settings).fit(selected).predict(validation)
        scored.append(validation.assign(fold=fold, forecast=forecast))
    return pd.concat(scored, ignore_index=True)


@dataclass(frozen=True)
class FoldSummary:
    name: str
    mean_mape: float
    mean_mae: float
    fold_mapes: tuple[float, ...]


def summarize_folds(name: str, scored: pd.DataFrame) -> FoldSummary:
    """Each fold counts the same, also if it holds fewer rows."""
    folds = [rows for _, rows in scored.groupby("fold")]
    fold_mapes = tuple(mape(rows[TARGET_COLUMN].to_numpy(), rows["forecast"].to_numpy()) for rows in folds)
    fold_maes = [mae(rows[TARGET_COLUMN].to_numpy(), rows["forecast"].to_numpy()) for rows in folds]
    return FoldSummary(name=name, mean_mape=float(np.mean(fold_mapes)), mean_mae=float(np.mean(fold_maes)),
                       fold_mapes=fold_mapes)
