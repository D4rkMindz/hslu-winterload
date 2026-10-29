from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod

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
