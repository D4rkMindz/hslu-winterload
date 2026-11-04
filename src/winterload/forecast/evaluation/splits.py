import datetime as dt

from winterload import DEFAULT_START_DATE, DEFAULT_END_DATE

TRAIN_START = DEFAULT_START_DATE
TRAIN_END = DEFAULT_END_DATE - dt.timedelta(days=365) - dt.timedelta(weeks=12)

VALIDATION_START = TRAIN_END
VALIDATION_END = DEFAULT_END_DATE - dt.timedelta(days=365)

# Reviewer point A2. This window holds 8,760 UTC hours (365 x 24), because the 23h and 25h
# clock-change days cancel out. The proposal's 8,782 hours run on to the last local hour of
# 2026-09-23, which adds 22 UTC hours. The feature store first held weather for exactly those 8,782
# hours. The final test scores the 8,760 hours of this half-open window.
TEST_START = VALIDATION_END
TEST_END = DEFAULT_END_DATE

# Reviewer point B2/A3: a fair comparison needs 8 weeks that neither model trained on. This split
# serves live retraining and stays separate from the one-time TRAIN/VALIDATION/TEST split above.
CHAMPION_CHALLENGER_HOLDOUT = dt.timedelta(weeks=8)

# Both models score the same rows, so a gap in the holdout keeps the comparison fair. Below one week
# of hours the result says little, so the pipeline skips the comparison.
MIN_HOLDOUT_HOURS = 7 * 24


def challenger_train_end(reference_date: dt.date) -> dt.date:
    """The challenger may only train on data strictly before this date."""
    return reference_date - CHAMPION_CHALLENGER_HOLDOUT


def champion_challenger_holdout(reference_date: dt.date) -> tuple[dt.date, dt.date] | None:
    """The `[start, end)` window both champion and challenger score on.

    Returns None while that window would overlap the final test year. Scoring models there would
    read the year a second time outside TestYearAccessPolicy (reviewer point B1/A4). The 168h lag of
    the window's first week still takes the year's last week as an input, as any live forecast does,
    but no metric scores those hours.
    """
    start = challenger_train_end(reference_date)
    if start < TEST_END:
        return None
    return start, reference_date


def training_windows(end: dt.date, test_year_released: bool) -> list[tuple[dt.date, dt.date]]:
    """The `[start, end)` windows a live model may train on, up to `end`.

    Reviewer point B1/A4 keeps the test year untouched until the end, so the windows skip it until
    the final test has run (`test_year_released`).
    """
    if test_year_released or end <= TEST_START:
        return [(TRAIN_START, end)]
    if end <= TEST_END:
        return [(TRAIN_START, TEST_START)]
    return [(TRAIN_START, TEST_START), (TEST_END, end)]
