"""What one inference run produces: the hours a model can forecast for a target day, and the records that
winterload.repository.forecast appends. Pure, no I/O."""

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from winterload.forecast.features.dataset import ZURICH_TZ

CHAMPION = "champion"
CHALLENGER = "challenger"
BASELINE = "baseline"
MODEL_ROLES = (CHAMPION, CHALLENGER, BASELINE)
# The seasonal-naive baseline has no registry version, so its model_id names the rule.
BASELINE_MODEL_ID = "seasonal_naive_168h"


@dataclass(frozen=True)
class ForecastRun:
    id: str
    started_at: dt.datetime
    completed_at: dt.datetime
    target_day: dt.date


@dataclass(frozen=True)
class ModelForecast:
    model_id: str
    model_role: str
    prediction_time: pd.Series
    result: np.ndarray


def tomorrow_in_zurich(now: dt.datetime) -> dt.date:
    """The Zurich local day after `now`, the day the 08:00 forecast is for. `now` must be timezone-aware."""
    return now.astimezone(ZoneInfo(ZURICH_TZ)).date() + dt.timedelta(days=1)


def split_complete(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """The rows with a value in each of `columns`, and the `event_timestamp` of every row that misses one."""
    complete = frame[columns].notna().all(axis=1)
    return frame.loc[complete].reset_index(drop=True), list(frame.loc[~complete, "event_timestamp"])
