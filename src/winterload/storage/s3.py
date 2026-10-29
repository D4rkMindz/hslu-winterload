"""Parquet files on SeaweedFS/S3 (or a local directory): one directory per table, one file per day,
read by Feast's `file` offline store (feature_store.yaml).
"""

import datetime as dt
import logging

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
