import logging
import sys

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# pydantic-settings' env_file= below only parses .env into this Settings object -- it doesn't
# touch os.environ. pyarrow and fsspec read their S3 credentials (AWS_*, FSSPEC_S3_*) straight
# from os.environ, not through Settings, so they need this too. Must run before `fsspec` is
# imported by anything downstream of this module (incl. transitively via feast/dask): fsspec
# snapshots its env-var config at first import, so loading .env any later leaves the SeaweedFS
# S3 credentials unset and historical retrieval fails with NoCredentialsError/ACCESS_DENIED.
load_dotenv(PROJECT_ROOT / ".env")

# Configured here, at the module everything else imports `settings` from, so every entry point
# (pipeline CLIs, notebooks) gets a working handler just by importing winterload.config -- without
# this, logger.info/.debug calls anywhere in the app are silently dropped (Python's logging has no
# handler by default). Set at INFO until `settings.log_level` is known below, so a config error
# itself (logged, not printed) is still visible even if LOG_LEVEL ends up invalid.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    entsoe_api_key: str = Field(alias="ENTSOE_API_KEY")

    feast_project: str = Field(alias="FEAST_PROJECT")
    # feast_store_path: str = Field(default="s3://feast/", alias="FEAST_S3_STORE_PATH") # online
    feast_store_path: str = Field(default="data/offline", alias="FEAST_OFFLINE_STORE_PATH")
    gcp_project_id: str = Field(alias="GCP_PROJECT_ID")

    # Where the backfills write and Feast reads (see winterload.storage): "s3" or "bigquery"
    offline_storage: Literal["s3", "bigquery"] = Field(default="s3", alias="OFFLINE_STORAGE")
    bigquery_dataset: str = Field(default="winterload", alias="BIGQUERY_DATASET")
    bigquery_location: str = Field(default="europe-west6", alias="BIGQUERY_LOCATION")

    s3_endpoint: str | None = Field(default=None, alias="S3_ENDPOINT")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


try:
    settings = Settings()
except ValidationError as e:
    logger.critical("Configuration error:\n%s", e)
    sys.exit(1)

logging.getLogger().setLevel(settings.log_level.upper())
