"""The feature pipelines: one `BackfillPipeline` per feature view, each writing one Parquet file
per day so a long backfill is resumable (see pipeline.weather_backfill for the layout this
mirrors). `run_all` discovers and runs every pipeline defined in this package; `python -m
winterload.pipeline` is its CLI entry point (see __main__.py).
"""

import datetime as dt
import importlib
import inspect
import logging
import pkgutil
import time
from abc import abstractmethod
from typing import Callable, TypedDict, Any

import pandas as pd
from winterload.pipeline.base import Pipeline
from winterload.storage import Storage, get_storage

DEFAULT_START_DATE = dt.date(2022, 1, 1)
DEFAULT_END_DATE = dt.date(2026, 9, 23)

RETRY_ATTEMPTS = 2
RETRY_WAIT_SECONDS = 30

__all__ = [
    "BackfillPipeline", "DEFAULT_START_DATE", "DEFAULT_END_DATE",
    "RETRY_ATTEMPTS", "RETRY_WAIT_SECONDS",
    "discover_pipelines", "retry", "main",
]

class FeaturePipeline(Pipeline):
    name: str = "Feature Pipeline"
    order: int = 100

    def run(self, start_date: dt.date, end_date: dt.date, force: bool = False) -> None:
        """"Run all pipelines in the right order and discover them, therefore override the run method here"""
        self.logger.info("== FeaturePipeline ==")
        self.logger.info("Starting LiveFeaturePipeline")
        LiveFeaturePipeline.run_all()
        self.logger.info("Starting Backfill")
        BackfillPipeline.run_all(start_date, end_date, force=force)

class LiveFeaturePipeline(FeaturePipeline):
    name: str = "LiveFeaturePipeline"
    order: int = 0

    def run(self) -> None:
        self.logger.info("TODO")


class BackfillPipeline(FeaturePipeline):
    """A feature pipeline that (re)builds a date range of per-day Parquet files for the feature
    store. Subclass this once per feature view; `discover_pipelines`/`run_all` pick up every
    concrete subclass defined anywhere in this package automatically, so a new pipeline module
    only needs to exist under `winterload.pipeline` to be included.
    """

    #: Human-readable name used in progress output; defaults to the class name if not overridden.
    name: str = ""
    #: The table this pipeline writes to: a directory of per-day Parquet files on S3, a BigQuery
    #: table in BigQuery -- see winterload.storage.
    table: str

    @abstractmethod
    def run(self, start_date: dt.date, end_date: dt.date, force: bool = False) -> None:
        """Writes one Parquet file per day in [start_date, end_date], skipping days that already
        have a file unless `force`. Returns (written, skipped, failed)."""

    def day_exists(self, target_day: dt.date) -> bool:
        return self.storage.day_exists(self.table, target_day)

    def write_day(self, df: pd.DataFrame, target_day: dt.date) -> str:
        return self.storage.write_day(self.table, df, target_day)



def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s", "--start", type=dt.date.fromisoformat, default=DEFAULT_START_DATE,
        help=f"default: {DEFAULT_START_DATE.isoformat()} (start of the forecast archive, docs/proposal.pdf)",
    )
    parser.add_argument(
        "-e", "--end", type=dt.date.fromisoformat, default=DEFAULT_END_DATE,
        help=f"default: {DEFAULT_END_DATE.isoformat()} (end of the held-out test year, docs/proposal.pdf)",
    )
    parser.add_argument(
        "-p", "--pipeline", type=dt.date.fromisoformat, default=DEFAULT_END_DATE,
        help=f"default: run all",
    )
    parser.add_argument("-f", "--force", action="store_true", help="re-fetch days that already have a file")
    args = parser.parse_args()

    FeaturePipeline.run_all(args.start, args.end, force=args.force)


def retry(callable: Callable[[], Any], identifier: str = "__retry__", retry_attempts: int = RETRY_ATTEMPTS,
          retry_wait_seconds: int = RETRY_WAIT_SECONDS) -> bool:
    """Runs `callable` retrying up to retry_attempts times (retry_wait_seconds
    apart) if it raises. Returns False once all attempts are exhausted, rejecting the day."""
    logger = logging.getLogger("retry")
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
