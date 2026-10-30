import datetime as dt
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from feast.data_source import DataSource


class Storage(ABC):
    """The offline store backend of the feature tables. A table holds one feature view's rows and
    is written one day at a time, so a backfill can skip the days it already has."""

    @abstractmethod
    def setup(self) -> None:
        """Creates whatever the backend needs before the first write or `feast apply`."""

    @abstractmethod
    def day_exists(self, table: str, target_day: dt.date) -> bool:
        """Whether `target_day` was already written to `table`."""

    @abstractmethod
    def write_day(self, table: str, df: pd.DataFrame, target_day: dt.date) -> str:
        """Writes `df` as `target_day`'s rows of `table`, replacing any earlier write of that day.
        Returns where it was written to, for logging."""

    @abstractmethod
    def source(self, table: str) -> DataSource:
        """The Feast data source reading `table`, for the feature views in definitions.py."""

    @abstractmethod
    def read_range(self, table: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Reads `table`'s rows with `event_timestamp` in `[start, end)` as one DataFrame. Used by
        winterload.pipeline.training to assemble train/validation/test dataframes directly from
        the feature tables, without going through Feast's point-in-time join."""

    @abstractmethod
    def read_marker(self, key: str) -> dict[str, Any] | None:
        """Reads a small named JSON marker (e.g. `final_test_year_access`), or None if it was
        never written."""

    @abstractmethod
    def write_marker(self, key: str, data: dict[str, Any]) -> None:
        """Writes `data` as a small named JSON marker, replacing any earlier value."""
