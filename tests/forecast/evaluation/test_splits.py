import datetime as dt

import pandas as pd
from winterload.forecast.evaluation.splits import CHAMPION_CHALLENGER_HOLDOUT, TEST_END, TEST_START, TRAIN_START, \
    champion_challenger_holdout, challenger_train_end, training_windows


def test_test_window_holds_8760_hours():
    hours = pd.date_range(pd.Timestamp(TEST_START, tz="UTC"), pd.Timestamp(TEST_END, tz="UTC"), freq="h",
                          inclusive="left")

    assert len(hours) == 8760


def test_proposal_count_runs_to_the_last_local_hour_of_the_end_date():
    last_local_midnight = pd.Timestamp(TEST_END + dt.timedelta(days=1), tz="Europe/Zurich").tz_convert("UTC")

    hours = pd.date_range(pd.Timestamp(TEST_START, tz="UTC"), last_local_midnight, freq="h", inclusive="left")

    assert len(hours) == 8782


def test_challenger_train_end_is_eight_weeks_before_reference_date():
    reference_date = TEST_END + dt.timedelta(weeks=20)

    assert challenger_train_end(reference_date) == reference_date - dt.timedelta(weeks=8)


def test_holdout_is_none_while_it_would_overlap_the_final_test_year():
    # A holdout inside the test year would read it a second time outside TestYearAccessPolicy.
    assert champion_challenger_holdout(TEST_START) is None
    assert champion_challenger_holdout(TEST_END) is None
    assert champion_challenger_holdout(TEST_END + CHAMPION_CHALLENGER_HOLDOUT - dt.timedelta(days=1)) is None


def test_holdout_starts_exactly_at_test_end_once_far_enough_past_it():
    reference_date = TEST_END + CHAMPION_CHALLENGER_HOLDOUT

    assert champion_challenger_holdout(reference_date) == (TEST_END, reference_date)


def test_holdout_window_well_past_the_test_year():
    reference_date = TEST_END + dt.timedelta(weeks=20)

    assert champion_challenger_holdout(reference_date) == (reference_date - CHAMPION_CHALLENGER_HOLDOUT, reference_date)


def test_training_windows_skip_the_test_year_until_it_is_released():
    end = TEST_END + dt.timedelta(weeks=20)

    assert training_windows(end, test_year_released=False) == [(TRAIN_START, TEST_START), (TEST_END, end)]


def test_training_windows_include_the_test_year_once_released():
    end = TEST_END + dt.timedelta(weeks=20)

    assert training_windows(end, test_year_released=True) == [(TRAIN_START, end)]


def test_training_windows_ending_at_or_inside_the_test_year_stop_at_its_start():
    assert training_windows(TEST_END, test_year_released=False) == [(TRAIN_START, TEST_START)]
    assert training_windows(TEST_START + dt.timedelta(days=30), test_year_released=False) == [(TRAIN_START, TEST_START)]


def test_training_windows_before_the_test_year_stop_at_end():
    end = TEST_START - dt.timedelta(days=30)

    assert training_windows(end, test_year_released=False) == [(TRAIN_START, end)]
