"""Replays past forecast days using only what was known at 08:00 Zurich time on D-1, reviewer point
A8. These functions take assembled frames and do no I/O. notebooks/backtest/replay.ipynb loads the
data and writes the logs.

A gap in the feature store is an expected result here, so check_day reports it instead of raising.
"""

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from winterload.forecast.features.dataset import LAG_HOURS, TARGET_COLUMN, ZURICH_TZ, local_day_bounds
from winterload.forecast.metrics import mae, mape
from winterload.forecast.model.baseline import seasonal_naive_forecast
from winterload.forecast.model.preliminary import PreliminaryModel

ISSUE_TIME_LOCAL = dt.time(8)

# Open-Meteo published the 00 and 06 UTC runs 6.1 to 7.4h after they started on 2026-10-31, and asks
# clients to wait 10 more minutes. The publication probe logs further observations.
WEATHER_RUN_DELAY = pd.Timedelta(8, unit="hours")
# The smallest publication_tolerance of any day, on the autumn clock change.
MIN_PUBLICATION_TOLERANCE = pd.Timedelta(8, unit="hours")


def issue_time(target_day: dt.date) -> pd.Timestamp:
    """08:00 Zurich time on the day before `target_day`."""
    return pd.Timestamp(dt.datetime.combine(target_day - dt.timedelta(days=1), ISSUE_TIME_LOCAL), tz=ZURICH_TZ)


def expected_target_hours(target_day: dt.date) -> pd.DatetimeIndex:
    """The UTC hours of a local day: 24 on normal days, 23 in spring, 25 in autumn."""
    start, end = local_day_bounds(target_day)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def publication_tolerance(target_day: dt.date) -> pd.Timedelta:
    """How late the newest load hour may arrive and still feed the shortest lag at issue time: 9h on
    normal days, 8h on the autumn clock-change day, 10h on the spring one."""
    newest_needed = expected_target_hours(target_day)[-1] - pd.Timedelta(min(LAG_HOURS), unit="hours")
    return issue_time(target_day) - newest_needed


def training_cutoff(target_day: dt.date) -> pd.Timestamp:
    """Local midnight on D-1. Labels before it had at least 7h to publish before issue time."""
    return local_day_bounds(target_day - dt.timedelta(days=1))[0]


def rows_of_local_day(frame: pd.DataFrame, target_day: dt.date) -> pd.DataFrame:
    start, end = local_day_bounds(target_day)
    return frame.loc[(frame["event_timestamp"] >= start) & (frame["event_timestamp"] < end)].reset_index(drop=True)


def rows_of_local_days(frame: pd.DataFrame, days: list[dt.date]) -> pd.DataFrame:
    return pd.concat([rows_of_local_day(frame, day) for day in days], ignore_index=True)


@dataclass(frozen=True)
class ReplayCheck:
    target_day: dt.date
    issue_time: pd.Timestamp
    expected_hours: int
    present_hours: int
    weather_run: pd.Timestamp | None
    weather_run_available: bool
    publication_tolerance: pd.Timedelta

    @property
    def complete(self) -> bool:
        return self.present_hours == self.expected_hours and self.weather_run_available


def check_day(target_day: dt.date, day_frame: pd.DataFrame, weather_runs: pd.Series) -> ReplayCheck:
    """`day_frame` holds the day's rows from `rows_of_local_day`, and `weather_runs` the run timestamps
    behind them. Both are empty on a gap day."""
    issued = issue_time(target_day)
    latest_run = weather_runs.max() if len(weather_runs) else None
    return ReplayCheck(
        target_day=target_day,
        issue_time=issued,
        expected_hours=len(expected_target_hours(target_day)),
        present_hours=len(day_frame),
        weather_run=latest_run,
        weather_run_available=latest_run is not None and latest_run + WEATHER_RUN_DELAY <= issued,
        publication_tolerance=publication_tolerance(target_day),
    )


@dataclass(frozen=True)
class ReplayResult:
    hours: int
    forecast: np.ndarray
    model_mape: float
    model_mae: float
    baseline_mape: float
    baseline_mae: float


def replay_day(model: PreliminaryModel, day_frame: pd.DataFrame) -> ReplayResult:
    """Scores `model` and the seasonal-naive baseline on the rows of one day."""
    actual = day_frame[TARGET_COLUMN].to_numpy()
    forecast = model.predict(day_frame)
    baseline = seasonal_naive_forecast(day_frame)
    return ReplayResult(
        hours=len(day_frame),
        forecast=forecast,
        model_mape=mape(actual, forecast),
        model_mae=mae(actual, forecast),
        baseline_mape=mape(actual, baseline),
        baseline_mae=mae(actual, baseline),
    )


@dataclass(frozen=True)
class RevisionReport:
    compared_hours: int
    changed_hours: int
    max_abs_change_mw: float


def compare_load_versions(before: pd.Series, after: pd.Series) -> RevisionReport:
    """Compares two versions of the load series, both indexed by `event_timestamp`. Only hours in both
    count, so a gap never reads as a revision."""
    joined = pd.concat([before.rename("before"), after.rename("after")], axis=1, join="inner").dropna()
    changes = (joined["before"] - joined["after"]).abs()
    return RevisionReport(
        compared_hours=len(joined),
        changed_hours=int((changes > 0).sum()),
        max_abs_change_mw=float(changes.max()) if len(joined) else 0.0,
    )


@dataclass(frozen=True)
class PublicationObservation:
    source: str
    observed_at: pd.Timestamp
    latest_published_hour: pd.Timestamp
    delay_hours: float
    tolerance_hours: float
    within_tolerance: bool


def observe_publication(source: str, observed_at: pd.Timestamp,
                        latest_published_hour: pd.Timestamp) -> PublicationObservation:
    """One observation of how far a load source lags behind real time, against the smallest tolerance
    of any day."""
    delay = observed_at - latest_published_hour
    return PublicationObservation(
        source=source,
        observed_at=observed_at,
        latest_published_hour=latest_published_hour,
        delay_hours=delay / pd.Timedelta(1, unit="hours"),
        tolerance_hours=MIN_PUBLICATION_TOLERANCE / pd.Timedelta(1, unit="hours"),
        within_tolerance=delay <= MIN_PUBLICATION_TOLERANCE,
    )


@dataclass(frozen=True)
class WeatherRunObservation:
    host: str
    observed_at: pd.Timestamp
    run: pd.Timestamp
    available_at: pd.Timestamp
    delay_hours: float
    within_assumed_delay: bool


def observe_weather_run(host: str, observed_at: pd.Timestamp, run: pd.Timestamp,
                        available_at: pd.Timestamp) -> WeatherRunObservation:
    """One observation of how long Open-Meteo took to publish a run, against WEATHER_RUN_DELAY."""
    delay = available_at - run
    return WeatherRunObservation(
        host=host,
        observed_at=observed_at,
        run=run,
        available_at=available_at,
        delay_hours=delay / pd.Timedelta(1, unit="hours"),
        within_assumed_delay=delay <= WEATHER_RUN_DELAY,
    )
