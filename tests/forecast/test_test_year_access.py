import datetime as dt

import pytest
from winterload.forecast.test_year_access import TestYearAccessPolicy, TestYearAlreadyAccessed

NOW = dt.datetime(2026, 10, 29, tzinfo=dt.UTC)


def test_first_access_is_allowed_and_recorded():
    decision = TestYearAccessPolicy().decide(None, force=False, accessed_at=NOW)

    assert decision.is_first_access
    assert decision.record["access_count"] == 1
    assert decision.record["previous_access_at"] is None
    assert decision.record["accessed_at"] == NOW.isoformat()


def test_second_access_without_force_raises():
    previous = {"accessed_at": "2026-10-01T00:00:00+00:00", "access_count": 1}

    with pytest.raises(TestYearAlreadyAccessed, match="already accessed"):
        TestYearAccessPolicy().decide(previous, force=False, accessed_at=NOW)


def test_second_access_with_force_is_allowed_and_increments_count():
    previous = {"accessed_at": "2026-10-01T00:00:00+00:00", "access_count": 1}

    decision = TestYearAccessPolicy().decide(previous, force=True, accessed_at=NOW)

    assert not decision.is_first_access
    assert decision.record["access_count"] == 2
    assert decision.record["previous_access_at"] == previous["accessed_at"]
