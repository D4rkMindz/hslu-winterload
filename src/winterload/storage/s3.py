"""Parquet files on SeaweedFS/S3 (or a local directory): one directory per table, one file per day,
read by Feast's `file` offline store (feature_store.yaml).
"""

import datetime as dt
import json
import logging

from typing import Any
from urllib.parse import urlparse

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from feast import FileSource
from fsspec import AbstractFileSystem
from winterload.config import settings
from winterload.storage.base import Storage

logger = logging.getLogger(__name__)


def get_filesystem(path: str) -> AbstractFileSystem:
    """Returns the fsspec filesystem for `path`: s3fs against the configured SeaweedFS/S3 endpoint
    for s3:// paths, the local filesystem otherwise -- so every pipeline resolves feast_store_path
    the same way whether it points at SeaweedFS or a local directory.
    """
    if path.startswith("s3://"):
        return s3fs.S3FileSystem()
    return fsspec.filesystem("local")


class S3Storage(Storage):
    def day_path(self, table: str, target_day: dt.date) -> str:
        return f"{settings.feast_store_path}/{table}/{target_day.isoformat()}.parquet"

    def day_exists(self, table: str, target_day: dt.date) -> bool:
        path = self.day_path(table, target_day)
        return get_filesystem(path).exists(path)

    def write_day(self, table: str, df: pd.DataFrame, target_day: dt.date) -> str:
        path = self.day_path(table, target_day)
        filesystem = get_filesystem(path)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, filesystem=filesystem)
        return path

    def source(self, table: str) -> FileSource:
        # A FileSource reads the directory of per-day Parquet files as one dataset.
        return FileSource(
            path=f"{settings.feast_store_path}/{table}",
            timestamp_field="event_timestamp",
            s3_endpoint_override=settings.s3_endpoint,
        )

    def read_range(self, table: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        path = f"{settings.feast_store_path}/{table}"
        filesystem = get_filesystem(path)
        if not filesystem.exists(path):
            return pd.DataFrame()
        # Reads every day's Parquet file under `table/` as one dataset (pyarrow treats a directory
        # path as a dataset), then filters to the requested window -- same per-day layout Feast's
        # FileSource reads, just without going through a Feast point-in-time join.
        df = pq.read_table(path, filesystem=filesystem).to_pandas()
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        in_range = (df["event_timestamp"] >= start_ts) & (df["event_timestamp"] < end_ts)
        return df.loc[in_range].sort_values("event_timestamp").reset_index(drop=True)

    def marker_path(self, key: str) -> str:
        return f"{settings.feast_store_path}/_markers/{key}.json"

    def read_marker(self, key: str) -> dict[str, Any] | None:
        path = self.marker_path(key)
        filesystem = get_filesystem(path)
        if not filesystem.exists(path):
            return None
        with filesystem.open(path, "r") as f:
            return json.load(f)

    def write_marker(self, key: str, data: dict[str, Any]) -> None:
        path = self.marker_path(key)
        filesystem = get_filesystem(path)
        with filesystem.open(path, "w") as f:
            json.dump(data, f, default=str)

    def setup(self) -> None:
        """Creates the S3 bucket backing the Feast offline store on SeaweedFS, if it doesn't exist yet.

        Unlike AWS S3, SeaweedFS's S3 gateway still requires the bucket to exist before anything can
        be written to it -- s3fs's S3FileSystem.mkdir() on a bucket-only path does that.
        """
        if not settings.feast_store_path.startswith("s3://"):
            logger.info("FEAST_OFFLINE_STORE_PATH=%r is not an s3:// path, nothing to do", settings.feast_store_path)
            return
        bucket = urlparse(settings.feast_store_path).netloc
        fs = get_filesystem(settings.feast_store_path)
        if not fs.exists(bucket):
            fs.mkdir(bucket)
        logger.info("Bucket %r ready on %s", bucket, settings.s3_endpoint)
