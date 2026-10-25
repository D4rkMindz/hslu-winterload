import logging

from urllib.parse import urlparse

import fsspec
import s3fs
from fsspec import AbstractFileSystem
from winterload.config import settings

logger = logging.getLogger(__name__)


def get_filesystem(path: str) -> AbstractFileSystem:
    """Returns the fsspec filesystem for `path`: s3fs against the configured SeaweedFS/S3 endpoint
    for s3:// paths, the local filesystem otherwise -- so every pipeline resolves feast_store_path
    the same way whether it points at SeaweedFS or a local directory.
    """
    if path.startswith("s3://"):
        return s3fs.S3FileSystem(endpoint_url=settings.s3_endpoint)
    return fsspec.filesystem("local")


def setup() -> None:
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
