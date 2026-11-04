import datetime as dt

import numpy as np
import pandas as pd
import pytest

from winterload.forecast.evaluation.backtest import MIN_PUBLICATION_TOLERANCE, check_day, compare_load_versions, \
    expected_target_hours, issue_time, observe_publication, observe_weather_run, publication_tolerance, replay_day, \
    rows_of_local_day, rows_of_local_days, training_cutoff
from winterload.forecast.features.dataset import TARGET_COLUMN, weather_run
from winterload.forecast.model.preliminary import PreliminaryModel

NORMAL_DAY = dt.date(2025, 7, 2)
SPRING_DAY = dt.date(2025, 3, 30)
AUTUMN_DAY = dt.date(2024, 10, 27)


def _utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def test_issue_time_is_0800_local_on_the_day_before():
    assert issue_time(NORMAL_DAY) == _utc("2025-07-01 06:00")
    assert issue_time(SPRING_DAY) == _utc("2025-03-29 07:00")
    # D-1 is the autumn clock-change day, so 08:00 local is already back on winter time.
    assert issue_time(dt.date(2024, 10, 28)) == _utc("2024-10-27 07:00")


def test_expected_target_hours_follow_clock_changes():
    assert len(expected_target_hours(NORMAL_DAY)) == 24
    assert len(expected_target_hours(SPRING_DAY)) == 23
    assert len(expected_target_hours(AUTUMN_DAY)) == 25


def test_publication_tolerance_shrinks_on_the_autumn_clock_change():
    assert publication_tolerance(NORMAL_DAY) == pd.Timedelta(9, unit="hours")
    assert publication_tolerance(AUTUMN_DAY) == pd.Timedelta(8, unit="hours")
    assert publication_tolerance(SPRING_DAY) == pd.Timedelta(10, unit="hours")


def test_training_cutoff_is_local_midnight_on_d_minus_1():
    assert training_cutoff(NORMAL_DAY) == _utc("2025-06-30 22:00")
    assert issue_time(NORMAL_DAY) - training_cutoff(NORMAL_DAY) == pd.Timedelta(8, unit="hours")


def test_rows_of_local_day_keeps_only_that_local_day():
    frame = pd.DataFrame({"event_timestamp": pd.date_range("2025-03-28", "2025-04-01", freq="h", tz="UTC")})

    assert len(rows_of_local_day(frame, SPRING_DAY)) == 23


def test_check_day_passes_a_complete_day():
    frame = pd.DataFrame({"event_timestamp": expected_target_hours(NORMAL_DAY)})

    check = check_day(NORMAL_DAY, frame, pd.Series([_utc("2025-06-30 18:00")]))

    assert check.complete
    assert check.present_hours == 24
    assert check.weather_run_available


def test_check_day_reports_a_gap_instead_of_raising():
    frame = pd.DataFrame({"event_timestamp": expected_target_hours(NORMAL_DAY)}).iloc[0:0]

    check = check_day(NORMAL_DAY, frame, pd.Series([], dtype="datetime64[ns, UTC]"))

    assert not check.complete
    assert check.present_hours == 0
    assert check.expected_hours == 24
    assert check.weather_run is None
    assert not check.weather_run_available


def test_check_day_flags_a_weather_run_published_after_issue_time():
    frame = pd.DataFrame({"event_timestamp": expected_target_hours(NORMAL_DAY)})

    check = check_day(NORMAL_DAY, frame, pd.Series([_utc("2025-07-01 00:00")]))

    assert not check.weather_run_available
    assert not check.complete


def test_replay_day_scores_model_and_baseline_on_the_same_rows():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "event_timestamp": pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC"),
        "load_lag_48h": rng.uniform(100, 200, size=200),
    })
    frame[TARGET_COLUMN] = frame["load_lag_48h"] * 2
    frame["load_lag_168h"] = frame[TARGET_COLUMN] * 1.1
    model = PreliminaryModel().fit(frame)

    result = replay_day(model, frame.iloc[:24])

    assert result.hours == 24
    assert len(result.forecast) == 24
    assert result.model_mape < 0.05
    assert result.baseline_mape == pytest.approx(0.10)


def test_compare_load_versions_counts_changes_and_ignores_gaps():
    index = pd.date_range("2025-07-01", periods=3, freq="h", tz="UTC")
    before = pd.Series([100.0, 200.0, 300.0], index=index)
    after = pd.Series([100.0, 205.0], index=index[:2])

    report = compare_load_versions(before, after)

    assert report.compared_hours == 2
    assert report.changed_hours == 1
    assert report.max_abs_change_mw == pytest.approx(5.0)


def test_compare_load_versions_handles_no_overlap():
    index = pd.date_range("2025-07-01", periods=2, freq="h", tz="UTC")

    report = compare_load_versions(pd.Series([], dtype="float64"), pd.Series([1.0, 2.0], index=index))

    assert report.compared_hours == 0
    assert report.max_abs_change_mw == 0.0


def test_min_publication_tolerance_is_the_autumn_tolerance():
    assert MIN_PUBLICATION_TOLERANCE == publication_tolerance(AUTUMN_DAY)


def test_observe_publication_accepts_exactly_the_tolerance():
    published = _utc("2025-07-01 00:00")

    on_time = observe_publication("entsoe", published + pd.Timedelta(8, unit="hours"), published)
    late = observe_publication("entsoe", published + pd.Timedelta(8.5, unit="hours"), published)

    assert on_time.within_tolerance
    assert on_time.delay_hours == 8
    assert not late.within_tolerance
    assert on_time.tolerance_hours == 8


def test_observe_weather_run_compares_against_the_assumed_delay():
    run = _utc("2025-07-01 00:00")

    fast = observe_weather_run("host", run + pd.Timedelta(7, unit="hours"), run, run + pd.Timedelta(6.86, unit="hours"))
    slow = observe_weather_run("host", run + pd.Timedelta(9, unit="hours"), run, run + pd.Timedelta(8.5, unit="hours"))

    assert fast.within_assumed_delay
    assert not slow.within_assumed_delay


@pytest.mark.parametrize("day", [SPRING_DAY, AUTUMN_DAY])
def test_check_day_passes_on_clock_change_days(day):
    frame = pd.DataFrame({"event_timestamp": expected_target_hours(day)})
    run = pd.Timestamp(weather_run(day), tz="UTC")

    check = check_day(day, frame, pd.Series([run]))

    assert check.complete
    assert check.present_hours == check.expected_hours
    assert check.weather_run == run


def test_rows_of_local_days_concatenates_the_days():
    hours = pd.date_range("2025-03-29", "2025-04-01", freq="h", tz="UTC")
    frame = pd.DataFrame({"event_timestamp": hours})
    rows = rows_of_local_days(frame, [dt.date(2025, 3, 30), dt.date(2025, 3, 31)])
    assert len(rows) == 23 + 24
