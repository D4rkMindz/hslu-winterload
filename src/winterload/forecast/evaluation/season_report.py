"""Model-versus-baseline results by season, reviewer point B3/A5. Winter (November to February) and
summer (June to August) follow the load comparison in docs/proposal.pdf, and the other months count
as transition.

Each result lists the hours and months behind it, because a segment can rest on a single month.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from winterload.forecast.features.dataset import TARGET_COLUMN, ZURICH_TZ
from winterload.forecast.features.holidays_ch import NATIONAL_HOLIDAY_SHARE
from winterload.forecast.metrics import mae, mape
from winterload.forecast.model.baseline import seasonal_naive_forecast
from winterload.forecast.model.preliminary import PreliminaryModel

SEASON_BY_MONTH = {
    11: "winter", 12: "winter", 1: "winter", 2: "winter",
    3: "transition", 4: "transition", 5: "transition", 9: "transition", 10: "transition",
    6: "summer", 7: "summer", 8: "summer",
}
SEASONS = ("winter", "transition", "summer")
MIN_TRAINING_HOURS = 90 * 24
# A season holds few hours per year, so a same-season model starts with 8 weeks of them.
MIN_SEASON_TRAINING_HOURS = 8 * 7 * 24
HOLIDAY_SEGMENTS = ("national holiday", "regional holiday", "other day")


@dataclass(frozen=True)
class SegmentResult:
    segment: str
    hours: int
    months: tuple[str, ...]
    model_mape: float
    model_mae: float
    baseline_mape: float
    baseline_mae: float


def local_months(frame: pd.DataFrame) -> pd.Series:
    return frame["event_timestamp"].dt.tz_convert(ZURICH_TZ).dt.strftime("%Y-%m")


def chronological_forecast(frame: pd.DataFrame, same_season_only: bool = False,
                           min_training_hours: int = MIN_TRAINING_HOURS) -> pd.DataFrame:
    """Scores each local calendar month with a model fit on the rows before it, so no month sees its
    own labels. With `same_season_only`, the model trains on earlier rows of the month's season. Skips
    months with under `min_training_hours` of history. Returns the scored rows with a `forecast`
    column."""
    months = local_months(frame)
    scored = []
    for month in sorted(months.unique()):
        rows = frame.loc[months == month]
        train = frame.loc[frame["event_timestamp"] < rows["event_timestamp"].min()]
        if same_season_only:
            train = train.loc[train["month"].map(SEASON_BY_MONTH) == SEASON_BY_MONTH[rows["month"].iloc[0]]]
        if len(train) >= min_training_hours:
            scored.append(rows.assign(forecast=PreliminaryModel().fit(train).predict(rows)))
    return pd.concat(scored, ignore_index=True) if scored else frame.iloc[0:0].assign(forecast=np.nan)


def report_by_segment(scored: pd.DataFrame, segments: pd.Series, order: tuple[str, ...]) -> list[SegmentResult]:
    """`scored` needs a `forecast` column, and `segments` names each row's segment. Segments without
    rows are left out."""
    results = []
    for segment in order:
        rows = scored.loc[segments == segment]
        if rows.empty:
            continue
        actual = rows[TARGET_COLUMN].to_numpy()
        forecast = rows["forecast"].to_numpy()
        baseline = seasonal_naive_forecast(rows)
        results.append(SegmentResult(
            segment=segment,
            hours=len(rows),
            months=tuple(sorted(local_months(rows).unique())),
            model_mape=mape(actual, forecast),
            model_mae=mae(actual, forecast),
            baseline_mape=mape(actual, baseline),
            baseline_mae=mae(actual, baseline),
        ))
    return results


def report_by_season(scored: pd.DataFrame) -> list[SegmentResult]:
    return report_by_segment(scored, scored["month"].map(SEASON_BY_MONTH), SEASONS)


def report_by_holiday(scored: pd.DataFrame) -> list[SegmentResult]:
    """Holidays apart from other days, as docs/proposal.pdf promises. See
    winterload.forecast.features.holidays_ch for the definitions."""
    segments = pd.Series("other day", index=scored.index)
    segments[scored["holiday_share_ch"] > 0] = "regional holiday"
    segments[scored["holiday_share_ch"] >= NATIONAL_HOLIDAY_SHARE] = "national holiday"
    return report_by_segment(scored, segments, HOLIDAY_SEGMENTS)


@dataclass(frozen=True)
class SeasonalModelComparison:
    season: str
    hours: int
    months: tuple[str, ...]
    single_mape: float
    single_mae: float
    seasonal_mape: float
    seasonal_mae: float
    baseline_mape: float


def compare_seasonal_models(frame: pd.DataFrame) -> list[SeasonalModelComparison]:
    """Reviewer A's check: does a model per season beat one model for the year? Both score the same
    rows, namely the months where both have enough history."""
    single = chronological_forecast(frame)
    seasonal = chronological_forecast(frame, same_season_only=True, min_training_hours=MIN_SEASON_TRAINING_HOURS)
    shared_months = set(local_months(single)) & set(local_months(seasonal))
    single_results = report_by_season(single.loc[local_months(single).isin(shared_months)])
    seasonal_results = {result.segment: result
                        for result in report_by_season(seasonal.loc[local_months(seasonal).isin(shared_months)])}
    return [
        SeasonalModelComparison(
            season=result.segment,
            hours=result.hours,
            months=result.months,
            single_mape=result.model_mape,
            single_mae=result.model_mae,
            seasonal_mape=seasonal_results[result.segment].model_mape,
            seasonal_mae=seasonal_results[result.segment].model_mae,
            baseline_mape=result.baseline_mape,
        )
        for result in single_results
    ]
