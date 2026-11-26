import datetime as dt

import pandas as pd
import pytest

from winterload.forecast.features.completeness import implausible_load, is_final, local_hours


@pytest.mark.parametrize("day,hours", [
    (dt.date(2026, 3, 29), 23), (dt.date(2026, 10, 25), 25), (dt.date(2026, 6, 1), 24),
])
def test_local_hours(day, hours):
    assert local_hours(day) == hours


def test_is_final_from_two_days_back():
    today = dt.date(2026, 11, 26)
    assert is_final(dt.date(2026, 11, 24), today)
    assert not is_final(dt.date(2026, 11, 25), today)


def test_implausible_load_flags_198_but_not_2816_mw():
    flags = implausible_load(pd.DataFrame({"quantity": [198.0, 2816.0]}))
    assert list(flags) == [True, False]
