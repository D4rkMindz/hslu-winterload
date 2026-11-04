import datetime as dt

import pytest
from winterload.forecast.features.holidays_ch import holiday_share_ch, is_holiday_ch
from winterload.forecast.features.population import CANTON_POPULATION


@pytest.mark.parametrize("day", [dt.date(2025, 1, 1), dt.date(2025, 5, 29), dt.date(2025, 8, 1), dt.date(2025, 12, 25)])
def test_national_holidays_count_for_everyone(day):
    assert holiday_share_ch(day) == pytest.approx(1.0)
    assert is_holiday_ch(day)


@pytest.mark.parametrize("day", [dt.date(2025, 4, 18), dt.date(2025, 4, 21), dt.date(2025, 6, 9), dt.date(2025, 12, 26)])
def test_widely_observed_holidays_are_flagged(day):
    assert 0.5 < holiday_share_ch(day) < 1
    assert is_holiday_ch(day)


@pytest.mark.parametrize("day", [dt.date(2025, 1, 2), dt.date(2025, 5, 1), dt.date(2025, 6, 19), dt.date(2025, 11, 1)])
def test_regional_holidays_are_not_flagged(day):
    assert 0 < holiday_share_ch(day) < 0.5
    assert not is_holiday_ch(day)


def test_an_ordinary_day_has_no_share():
    assert holiday_share_ch(dt.date(2025, 7, 15)) == 0.0
    assert not is_holiday_ch(dt.date(2025, 7, 15))


def test_a_geneva_only_holiday_weighs_geneva_alone():
    expected = CANTON_POPULATION["GE"] / sum(CANTON_POPULATION.values())

    assert holiday_share_ch(dt.date(2025, 9, 11)) == pytest.approx(expected)


def test_eight_days_of_2025_are_flagged():
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(365)]

    assert sum(is_holiday_ch(day) for day in days) == 8
