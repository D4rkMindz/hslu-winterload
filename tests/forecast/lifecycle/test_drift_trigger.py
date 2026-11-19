import datetime as dt

from winterload.forecast.evaluation.splits import TEST_START, TRAIN_START
from winterload.forecast.lifecycle.retrain_trigger import (
    CHAMPION_TRAINING_FRAME,
    DRIFT_SHARE_THRESHOLD,
    SAME_WEEKS_LAST_YEAR,
    DriftTriggerPolicy,
    drift_reference,
)

START, END = dt.date(2026, 11, 5), dt.date(2026, 11, 12)


def test_half_of_the_columns_drifting_triggers_a_retrain():
    assert DRIFT_SHARE_THRESHOLD == 0.5
    assert DriftTriggerPolicy().decide(0.5).retrain
    assert not DriftTriggerPolicy().decide(0.49).retrain


def test_before_the_final_test_the_reference_is_the_champions_training_frame():
    reference = drift_reference(START, END, test_year_released=False, champion_training_end=dt.date(2026, 9, 30))

    assert reference.kind == CHAMPION_TRAINING_FRAME
    # The test year stays out of the reference, like out of the champion's training data.
    assert reference.windows[0] == (TRAIN_START, TEST_START)
    assert all(start >= dt.date(2026, 9, 23) for start, _ in reference.windows[1:])


def test_without_a_champion_and_before_the_final_test_there_is_no_reference():
    assert drift_reference(START, END, test_year_released=False, champion_training_end=None) is None


def test_after_the_final_test_the_reference_is_the_same_weeks_a_year_earlier():
    reference = drift_reference(START, END, test_year_released=True, champion_training_end=None)

    assert reference.kind == SAME_WEEKS_LAST_YEAR
    [(start, end)] = reference.windows
    assert (start.weekday(), end - start) == (START.weekday(), END - START)
    assert start.year == START.year - 1
