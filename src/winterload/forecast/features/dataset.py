import datetime as dt

import pandas as pd
from winterload.forecast.features.holidays_ch import holiday_share_ch
from winterload.forecast.features.population import city_population_weights

ZURICH_TZ = "Europe/Zurich"

# Load lags: ">= 48h" per docs/proposal.pdf (the lag rule: at 08:00 on D-1, a 24h lag does not
# exist yet for the forecast horizon). 72h and 168h (one week) add weekly-pattern signal cheaply.
LAG_HOURS = [48, 72, 168]

TARGET_COLUMN = "quantity"

CALENDAR_COLUMNS = ["hour", "day_of_week", "month", "is_weekend", "holiday_share_ch"]
LAG_COLUMNS = [f"load_lag_{h}h" for h in LAG_HOURS]
# Matches winterload.source.openmeteo.HOURLY_VARIABLES, and a test pins that. The domain layer does
# not import the source layer.
WEATHER_COLUMNS = ["temperature_2m", "cloud_cover", "shortwave_radiation"]

# Reviewer point A1. Open-Meteo published the 00 UTC run at 06:51 UTC on 2026-10-31. That misses
# 08:00 Zurich time in summer (06:00 UTC) and leaves 9 minutes in winter (07:00 UTC). ECMWF delivers
# the 18 UTC run by about 00:30 UTC, so the run from the evening of D-2 reaches every issue time. The
# weather features then have lead times of 28 to 52 hours.
WEATHER_RUN_DAYS_BEFORE_TARGET = 2
WEATHER_RUN_TIME_UTC = dt.time(18)
# The Single Runs archive starts on 2024-03-14 with only the 00 and 12 UTC cycles, and adds 18 UTC on
# this day. Earlier days use 12 UTC, whose lead times reach 58 hours.
FIRST_18_UTC_RUN = dt.date(2024, 8, 6)
EARLY_WEATHER_RUN_TIME_UTC = dt.time(12)
FIRST_WEATHER_DAY = dt.date(2024, 3, 16)

EQUAL_CITY_WEIGHTS = {"basel": 0.25, "bern": 0.25, "geneva": 0.25, "zurich": 0.25}
POPULATION_CITY_WEIGHTS = city_population_weights()
# Reviewer point E2. Population weights count every canton to its nearest city, which gives Zurich 53%
# and Basel 6%. They scored worse than equal weights on six chronological folds
# (notebooks/training/feature_study.ipynb), so the unweighted mean stays.
DEFAULT_CITY_WEIGHTS = EQUAL_CITY_WEIGHTS


def weather_run(target_day: dt.date) -> dt.datetime:
    """The run behind the weather features of `target_day`, as a naive UTC datetime."""
    run_day = target_day - dt.timedelta(days=WEATHER_RUN_DAYS_BEFORE_TARGET)
    return dt.datetime.combine(run_day, WEATHER_RUN_TIME_UTC if run_day >= FIRST_18_UTC_RUN else EARLY_WEATHER_RUN_TIME_UTC)


def local_day_bounds(target_day: dt.date, tz: str = ZURICH_TZ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The `[start, end)` instants of a local calendar day.

    Builds both ends from the calendar date. A fixed 24h Timedelta overshoots or undershoots by an
    hour on clock-change days (23h in spring, 25h in autumn) and drops or double-counts that hour.
    """
    return pd.Timestamp(target_day, tz=tz), pd.Timestamp(target_day + dt.timedelta(days=1), tz=tz)


def aggregate_weather(weather: pd.DataFrame, weights: dict[str, float] = DEFAULT_CITY_WEIGHTS) -> pd.DataFrame:
    """One row per hour: the weighted mean of the cities' forecasts. A city without a value for an
    hour drops out of that hour's mean."""
    values = weather[WEATHER_COLUMNS]
    city_weights = weather["location_id"].map(weights)
    hours = weather["event_timestamp"]
    weighted_sum = values.mul(city_weights, axis=0).groupby(hours).sum(min_count=1)
    weight_sum = values.notna().mul(city_weights, axis=0).groupby(hours).sum()
    return (weighted_sum / weight_sum).reset_index()


def _load_lags(load: pd.Series) -> pd.DataFrame:
    """`load` must already be reindexed to one row per UTC hour (NaN where a reading is missing),
    so a `.shift(h)` lands exactly h hours back regardless of gaps in the raw data."""
    return pd.DataFrame({column: load.shift(hours) for column, hours in zip(LAG_COLUMNS, LAG_HOURS)})


def _calendar_features(event_timestamp: pd.Series) -> pd.DataFrame:
    """Calendar fields from the Swiss local time of each hour."""
    local = event_timestamp.dt.tz_convert(ZURICH_TZ)
    return pd.DataFrame({
        "event_timestamp": event_timestamp,
        "hour": local.dt.hour,
        "day_of_week": local.dt.dayofweek,
        "month": local.dt.month,
        "is_weekend": local.dt.dayofweek.isin([5, 6]).astype(int),
        "holiday_share_ch": local.dt.date.map(holiday_share_ch).astype(float),
    })


def assemble_training_frame(load: pd.Series, weather: pd.DataFrame, start: dt.date) -> pd.DataFrame:
    """Builds one row per UTC hour in `[start, ...)`: the load target, its lags, weather forecast
    features and calendar fields, dropping rows with any missing value.

    `load` must cover enough lookback before `start` for the largest lag (reindexed to one row per
    UTC hour, as `_load_lags` requires); `weather` must have one row per `event_timestamp` already
    aggregated to a single value per feature (see `aggregate_weather`) -- both are
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
