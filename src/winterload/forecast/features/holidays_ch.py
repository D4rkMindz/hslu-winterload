"""Swiss public holidays as national calendar features, reviewer point A6. Cantons set most public
holidays themselves, and the `holidays` package lists them per canton. Only New Year, Ascension,
1 August and Christmas hold in all 26 cantons.

A national forecast needs one signal, so each canton counts by its population. `holiday_share_ch` is
the share of Swiss residents whose canton has a public holiday on a local date, and the model trains on
it. `is_holiday_ch` marks the dates where that share reaches half: those four days plus Good Friday,
Easter Monday, Whit Monday and 26 December. The model does not train on the flag, because a tree can
split the share at the same point and the flag alone scored worse (notebooks/training/feature_study.ipynb).
"""

import datetime as dt
from functools import lru_cache

import holidays
from winterload.forecast.features.population import CANTON_POPULATION

NATIONAL_HOLIDAY_SHARE = 0.5


@lru_cache(maxsize=None)
def _shares_of_year(year: int) -> dict[dt.date, float]:
    total = sum(CANTON_POPULATION.values())
    shares: dict[dt.date, float] = {}
    for canton, population in CANTON_POPULATION.items():
        for day in holidays.country_holidays("CH", subdiv=canton, years=year):
            shares[day] = shares.get(day, 0.0) + population / total
    return shares


def holiday_share_ch(day: dt.date) -> float:
    return _shares_of_year(day.year).get(day, 0.0)


def is_holiday_ch(day: dt.date) -> bool:
    return holiday_share_ch(day) >= NATIONAL_HOLIDAY_SHARE
