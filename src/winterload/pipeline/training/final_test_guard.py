"""Orchestrates the single legitimate read of the final held-out test year: reads the access
marker, asks winterload.forecast.test_year_access whether this access is allowed, loads the data
via winterload.pipeline.training.repository, writes the marker, logs. No business rule lives here
-- see winterload.forecast.test_year_access.TestYearAccessPolicy for that.
"""

import datetime as dt
import logging

import pandas as pd
from winterload.forecast.splits import TEST_END, TEST_START
from winterload.forecast.test_year_access import TestYearAccessPolicy
from winterload.pipeline.training.repository import load_dataframe
from winterload.storage import Storage, get_storage

logger = logging.getLogger(__name__)

MARKER_KEY = "final_test_year_access"


def load_final_test_year(storage: Storage | None = None, *, force: bool = False) -> pd.DataFrame:
    """Loads the final test year's dataframe. Raises TestYearAlreadyAccessed if it was already
    loaded once and `force` isn't set. Every successful call is logged loudly and recorded as a
    marker -- `force=True` included -- so even a deliberate re-run leaves an audit trail of how
    many times the test year was actually looked at."""
    storage = storage or get_storage()
    previous = storage.read_marker(MARKER_KEY)
    accessed_at = dt.datetime.now(dt.UTC)

    decision = TestYearAccessPolicy().decide(previous, force=force, accessed_at=accessed_at)

    if decision.is_first_access:
        logger.warning(
            "FINAL TEST YEAR ACCESSED for the first time (%s to %s) at %s. This should happen "
            "exactly once, at the very end.", TEST_START, TEST_END, accessed_at,
        )
    else:
        logger.warning(
            "FINAL TEST YEAR ACCESSED AGAIN (%s to %s) at %s -- force=True, previous access was "
            "at %s. Make sure this is intentional.", TEST_START, TEST_END, accessed_at,
            previous["accessed_at"],
        )

    frame = load_dataframe(storage, TEST_START, TEST_END)
    storage.write_marker(MARKER_KEY, decision.record)
    return frame
