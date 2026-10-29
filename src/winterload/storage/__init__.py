"""Where the backfill pipelines write their feature tables, and where Feast reads them from: one
`Storage` adapter per backend, picked by OFFLINE_STORAGE (see .env.example). Adding a backend means
implementing `Storage` and registering it in `get_storage`, plus a feature_store.<backend>.yaml for
the matching Feast offline store.
"""

from __future__ import annotations

from winterload.config import settings
from winterload.storage.base import Storage
from winterload.storage.s3 import S3Storage, get_filesystem

__all__ = ["Storage", "get_filesystem", "get_storage", "setup"]


def get_storage(backend: str | None = None) -> Storage:
    """The adapter for `backend`, defaulting to the configured OFFLINE_STORAGE."""
    backend = backend or settings.offline_storage
    if backend == "s3":
        return S3Storage()
    if backend == "bigquery":
        from winterload.storage.bigquery import BigQueryStorage
        return BigQueryStorage()
    raise ValueError(f"Unknown OFFLINE_STORAGE {backend!r}, expected 's3' or 'bigquery'")


def setup() -> None:
    get_storage().setup()
