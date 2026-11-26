"""When a backfilled day counts as complete, reviewer point A4. Pure: the pipelines pass in the frame and today."""
import datetime as dt

import pandas as pd

from winterload.forecast.features.dataset import local_day_bounds

# Below this the Swiss load is a broken reading, not a quiet night: the lowest real hour since 2024 is ~2,800 MW.
LOAD_FLOOR_MW = 2000.0
# ENTSO-E and ERA5 fill in recent days later. Only a day this old must be complete.
FINAL_AFTER = dt.timedelta(days=2)


def local_hours(day: dt.date) -> int:
    """23, 24 or 25: the hours of a Zurich local day."""
    start, end = local_day_bounds(day)
    return int((end - start) / pd.Timedelta(hours=1))


def is_final(day: dt.date, today: dt.date) -> bool:
    return day <= today - FINAL_AFTER


def implausible_load(frame: pd.DataFrame) -> pd.Series:
    """True for each row whose `quantity` is below LOAD_FLOOR_MW."""
    return frame["quantity"] < LOAD_FLOOR_MW
