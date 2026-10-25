"""The feature pipelines: one `BackfillPipeline` per feature view, each writing one Parquet file
per day so a long backfill is resumable (see pipeline.weather_backfill for the layout this
mirrors). `run_all` discovers and runs every pipeline defined in this package; `python -m
winterload.pipeline` is its CLI entry point (see __main__.py).
"""

from __future__ import annotations

import datetime as dt
import importlib
import inspect
import logging
import pkgutil
import time
from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from winterload import COORDINATES
from winterload.storage import get_filesystem
from winterload.source.openmeteo import OpenMeteo

# The full span the proposal trains and evaluates on: "used from 01.2022 (start of the forecast
# archive)" through the end of the held-out test year (23.09.2026) -- docs/proposal.pdf, §3 Volume
# and §3.3 Leakage and split.
DEFAULT_START_DATE = dt.date(2025, 9, 23)
DEFAULT_END_DATE = dt.date(2026, 9, 23)

# Source APIs (ENTSO-E, Open-Meteo) rate-limit requests. One day is one batch: on failure (most
# likely a rate limit) it's retried twice, 30s apart, and if it still fails the day is rejected --
# no file is written for it, so it's neither counted as done nor skipped, and simply rerunning the
# same backfill command later retries only that missing day instead of the whole range.
RETRY_ATTEMPTS = 2
RETRY_WAIT_SECONDS = 30

__all__ = [
    "BackfillPipeline", "DEFAULT_START_DATE", "DEFAULT_END_DATE", "RETRY_ATTEMPTS", "RETRY_WAIT_SECONDS",
    "discover_pipelines", "run_all", "retry",
]

logger = logging.getLogger(__name__)


class BackfillPipeline(ABC):
    """A feature pipeline that (re)builds a date range of per-day Parquet files for the feature
    store. Subclass this once per feature view; `discover_pipelines`/`run_all` pick up every
    concrete subclass defined anywhere in this package automatically, so a new pipeline module
    only needs to exist under `winterload.pipeline` to be included.
    """

    #: Human-readable name used in progress output; defaults to the class name if not overridden.
    name: str = ""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__

    @abstractmethod
    def build_range(self, start_date: dt.date, end_date: dt.date, force: bool = False) -> tuple[int, int, int]:
        """Writes one Parquet file per day in [start_date, end_date], skipping days that already
        have a file unless `force`. Returns (written, skipped, failed)."""


class OpenMeteoBackfillPipeline(BackfillPipeline):
    @abstractmethod
    def load_day(self, client: OpenMeteo, target_day: dt.date, cities: list[dict]) -> dict:
        """Load the open meteo data for a day, return a list of dicts per city"""

    @abstractmethod
    def transform_day(self, client: OpenMeteo, target_day: dt.date) -> list[dict]:
        """Transform the open meteo data for a day"""

    @abstractmethod
    def day_path(self, target_day: dt.date) -> str:
        """Returns the path for a day. e.g.
        return f"{settings.feast_store_path}/weather_forecast/{target_day.isoformat()}.parquet" """

    @abstractmethod
    def feature_columns(self) -> list[str]:
        """Returns a list of all feature columns"""

    def day_exists(self, target_day: dt.date) -> bool:
        path = self.day_path(target_day)
        return get_filesystem(path).exists(path)

    def write_day(self, df: pd.DataFrame, target_day: dt.date) -> str:
        path = self.day_path(target_day)
        filesystem = get_filesystem(path)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, filesystem=filesystem)
        return path

    def build_day(self, client: OpenMeteo, target_day: dt.date, cities: list[dict] = COORDINATES) -> pd.DataFrame:
        city_rows = self.load_day(client, target_day, cities)
        rows = []
        for city, city_data in city_rows.items():
            for model_name, model_data in city_data.items():
                day_hours = self.transform_day(target_day, model_data)
                day_hours["location_id"] = city
                day_hours["model"] = model_name
                rows.append(day_hours)

        result = pd.concat(rows, ignore_index=True).rename(columns={"date": "event_timestamp"})
        return result[self.feature_columns()]

    def build_range(self, start_date: dt.date, end_date: dt.date,
                    cities: list[dict] = COORDINATES, force: bool = False) -> tuple[int, int, int]:
        # single-run data never changes once published -- cache it for the lifetime of a long backfill
        client = OpenMeteo(cache_expire_after=30 * 86400)
        written = skipped = failed = 0
        day = start_date
        while day <= end_date:
            logger.info("%s: processing %s (%s to %s)", self.name, day, start_date, end_date)
            exists = self.day_exists(day)
            if exists is None:
                failed += 1
                day += dt.timedelta(days=1)
                continue
            if not force and exists:
                skipped += 1
                day += dt.timedelta(days=1)
                continue
            if retry(lambda: self.write_day(self.build_day(client, day, cities), day),
                     f"Backfilling {day} ({len(cities)} cities)"):
                written += 1
            else:
                failed += 1
            day += dt.timedelta(days=1)
        return written, skipped, failed


def discover_pipelines() -> list[type[BackfillPipeline]]:
    for module_info in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
        if module_info.name.rsplit(".", 1)[-1] == "__main__":
            continue
        importlib.import_module(module_info.name)

    def all_subclasses(cls: type) -> set[type]:
        return set(cls.__subclasses__()).union(
            sub for subclass in cls.__subclasses__() for sub in all_subclasses(subclass)
        )

    return [cls for cls in all_subclasses(BackfillPipeline) if not inspect.isabstract(cls)]


def run_all(
        start_date: dt.date = DEFAULT_START_DATE, end_date: dt.date = DEFAULT_END_DATE, force: bool = False
) -> None:
    """Runs every discovered pipeline's `build_range` over [start_date, end_date]."""
    pipelines = discover_pipelines()
    if not pipelines:
        logger.warning("No BackfillPipeline subclasses found under winterload.pipeline")
        return

    for cls in pipelines:
        pipeline = cls()
        logger.info("=== %s: %s to %s ===", pipeline.name, start_date, end_date)
        written, skipped, failed = pipeline.build_range(start_date, end_date, force=force)
        logger.info(
            "%s: wrote %s new day(s), skipped %s already-backfilled day(s), "
            "rejected %s day(s) after repeated failures",
            pipeline.name, written, skipped, failed,
        )


def retry(callable: Callable[[], None], identifier: str = "__retry__", retry_attempts: int = RETRY_ATTEMPTS,
          retry_wait_seconds: int = RETRY_WAIT_SECONDS) -> bool:
    """Runs `callable` retrying up to retry_attempts times (retry_wait_seconds
    apart) if it raises. Returns False once all attempts are exhausted, rejecting the day."""
    for attempt in range(retry_attempts + 1):
        try:
            callable()
            return True
        except Exception as e:
            if attempt == RETRY_ATTEMPTS:
                logger.info("[%s]: giving up after %s attempt(s): %s", identifier, attempt + 1, e)
                return False
            logger.info("[%s]: attempt %s failed (%s), retrying in %ss", identifier, attempt + 1, e,
                        RETRY_WAIT_SECONDS)
            time.sleep(RETRY_WAIT_SECONDS)
    return False
