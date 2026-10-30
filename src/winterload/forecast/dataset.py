import datetime as dt

import pandas as pd

ZURICH_TZ = "Europe/Zurich"

# Load lags: ">= 48h" per docs/proposal.pdf (the lag rule: at 08:00 on D-1, a 24h lag does not
# exist yet for the forecast horizon). 72h and 168h (one week) add weekly-pattern signal cheaply.
LAG_HOURS = [48, 72, 168]

TARGET_COLUMN = "quantity"

CALENDAR_COLUMNS = ["hour", "day_of_week", "month", "is_weekend"]
LAG_COLUMNS = [f"load_lag_{h}h" for h in LAG_HOURS]


def _load_lags(load: pd.Series) -> pd.DataFrame:
    """`load` must already be reindexed to one row per UTC hour (NaN where a reading is missing),
    so a `.shift(h)` lands exactly h hours back regardless of gaps in the raw data."""
    return pd.DataFrame({column: load.shift(hours) for column, hours in zip(LAG_COLUMNS, LAG_HOURS)})


def _calendar_features(event_timestamp: pd.Series) -> pd.DataFrame:
    """Basic calendar fields from the Swiss local time of each hour."""
    local = event_timestamp.dt.tz_convert(ZURICH_TZ)
    return pd.DataFrame({
        "event_timestamp": event_timestamp,
        "hour": local.dt.hour,
        "day_of_week": local.dt.dayofweek,
        "month": local.dt.month,
        "is_weekend": local.dt.dayofweek.isin([5, 6]).astype(int),
    })


def assemble_training_frame(load: pd.Series, weather: pd.DataFrame, start: dt.date) -> pd.DataFrame:
    """Builds one row per UTC hour in `[start, ...)`: the load target, its lags, weather forecast
    features and calendar fields, dropping rows with any missing value.

    `load` must cover enough lookback before `start` for the largest lag (reindexed to one row per
    UTC hour, as `_load_lags` requires); `weather` must have one row per `event_timestamp` already
    aggregated to a single value per feature (e.g. averaged across cities) -- both are
    winterload.pipeline.training's responsibility to fetch and shape. The model's input columns
    are simply every column here except `event_timestamp` and `TARGET_COLUMN` -- see
    winterload.forecast.model.PreliminaryModel.
    """
    lags = _load_lags(load)
    frame = lags.loc[load.index >= pd.Timestamp(start, tz="UTC")].copy()
    frame[TARGET_COLUMN] = load.loc[frame.index]
    frame = frame.rename_axis("event_timestamp").reset_index()

    frame = frame.merge(weather, on="event_timestamp", how="left")
    frame = frame.merge(_calendar_features(frame["event_timestamp"]), on="event_timestamp", how="left")

    return frame.dropna().reset_index(drop=True)
