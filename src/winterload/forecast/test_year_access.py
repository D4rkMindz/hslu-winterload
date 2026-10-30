import datetime as dt
from dataclasses import dataclass
from typing import Any


class TestYearAlreadyAccessed(RuntimeError):
    """Raised when the final test year is accessed a second time without force=True."""
    __test__ = False  # not a pytest test case -- the name just starts with "Test"


@dataclass(frozen=True)
class TestYearAccessDecision:
    __test__ = False  # not a pytest test case -- the name just starts with "Test"

    #: What to persist as the new access marker.
    record: dict[str, Any]
    #: Whether this is the first-ever access (vs. a force=True re-access).
    is_first_access: bool


class TestYearAccessPolicy:
    def decide(self, previous: dict[str, Any] | None, *, force: bool,
               accessed_at: dt.datetime) -> TestYearAccessDecision:
        """`previous` is the last-persisted access record, or None if the test year was never
        accessed. Raises TestYearAlreadyAccessed if `previous` is set and `force` isn't."""
        if previous is not None and not force:
            raise TestYearAlreadyAccessed(
                f"The final test year was already accessed at {previous['accessed_at']}. "
                "Evaluating on it again turns it into a validation set -- see reviewer point "
                "B1/A4. If this is a deliberate, final re-run, pass force=True."
            )

        record = {
            "accessed_at": accessed_at.isoformat(),
            "previous_access_at": previous["accessed_at"] if previous else None,
            "access_count": (previous.get("access_count", 1) + 1) if previous else 1,
        }
        return TestYearAccessDecision(record=record, is_first_access=previous is None)
