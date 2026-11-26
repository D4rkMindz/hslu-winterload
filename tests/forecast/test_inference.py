import datetime as dt

import pandas as pd

from winterload.forecast.inference import split_complete, tomorrow_in_zurich


def test_tomorrow_is_the_zurich_day_after_now():
    assert tomorrow_in_zurich(dt.datetime(2026, 11, 26, 12, tzinfo=dt.UTC)) == dt.date(2026, 11, 27)
    # 23:30 UTC is already 00:30 of the next day in Zurich (CET)
    assert tomorrow_in_zurich(dt.datetime(2026, 11, 26, 23, 30, tzinfo=dt.UTC)) == dt.date(2026, 11, 28)


def test_split_complete_names_the_hours_with_a_missing_input():
    hours = pd.date_range("2026-11-26 23:00", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"event_timestamp": hours, "a": [1.0, None, 3.0], "b": [1.0, 2.0, None]})

    complete, missing = split_complete(frame, ["a"])
    assert list(complete["event_timestamp"]) == [hours[0], hours[2]]
    assert missing == [hours[1]]

    complete, missing = split_complete(frame, ["a", "b"])
    assert list(complete["event_timestamp"]) == [hours[0]]
    assert missing == [hours[1], hours[2]]
